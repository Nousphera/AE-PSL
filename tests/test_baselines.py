"""Baseline regressions use small real ViTs; public entrypoints retain ViT-B/32."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from ae_psl.models import adc, c3_sl, quantization
from ae_psl.models.random_top_k import RandomTopKModifier


def base_model():
    torch.manual_seed(71)
    vit = VisionTransformer(image_size=16, patch_size=8, num_layers=3,
                            num_heads=2, hidden_dim=8, mlp_dim=16, num_classes=3)
    # torchvision's zero-initialized head would hide omitted transformer layers.
    nn.init.normal_(vit.heads.head.weight)
    return SimpleNamespace(vit=vit)


def pair(method, *, rate=.4, ratio=2, compress_cls=False):
    base = base_model()
    if method == 'rand_top_k':
        client = quantization.ClientModel(base, quantization.Quantization_Layer(32),
            RandomTopKModifier(rate), 'cpu', 1, compress_cls, 3, rate, 32, False)
        server = quantization.ServerModel(base, 'cpu', 1, compress_cls)
    elif method == 'c3_sl':
        client = c3_sl.ClientModel(base, ratio, 5 if compress_cls else 4, 8, 'cpu', 1, compress_cls)
        server = c3_sl.ServerModel(base, 'cpu', 1)
    else:
        client = adc.ClientModel(base, 'cpu', 1, .5, .6, 3)
        server = adc.ServerModel(base, 'cpu', 1)
    return base, client, server


@pytest.mark.parametrize('method', ['rand_top_k', 'c3_sl', 'adc'])
def test_baseline_split_bypass_matches_every_unsplit_transformer_layer(method):
    base, client, server = pair(method)
    base.vit.eval(); client.eval(); server.eval()
    images = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(server.forward_uncompressed(client(images)), base.vit(images))


def test_rand_topk_preserves_cls_and_exact_selected_patch_gradients():
    _, client, _ = pair('rand_top_k', rate=.25)
    x = torch.arange(1, 81, dtype=torch.float32).reshape(2, 5, 8).requires_grad_()
    result, _, _ = client.compress_decompress(x)
    torch.testing.assert_close(result[:, :1], x[:, :1], rtol=0, atol=0)
    assert torch.count_nonzero(result[:, 1:]).item() == 16
    result.sum().backward()
    torch.testing.assert_close(x.grad[:, :1], torch.ones(2, 1, 8))
    torch.testing.assert_close(x.grad[:, 1:] != 0, result[:, 1:] != 0)


def test_rand_topk_full_rate_is_identity():
    x = torch.randn(2, 3, 4, requires_grad=True)
    torch.testing.assert_close(RandomTopKModifier(1)(x), x)


@pytest.mark.parametrize('value', [0., 3.])
def test_quantization_constant_values_and_gradients_remain_finite(value):
    x = torch.full((2, 4), value, requires_grad=True)
    result = quantization.Quantization_Layer(8)(x)
    torch.testing.assert_close(result, x)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_c3_keys_survive_checkpoint_reload_and_module_dtype_movement():
    _, client, _ = pair('c3_sl')
    restored = copy.deepcopy(client)
    with torch.no_grad():
        restored.encoder.keys.zero_()
    restored.load_state_dict(client.state_dict())
    x = torch.randn(3, 5, 8)
    torch.testing.assert_close(restored.compress_decompress(x)[0], client.compress_decompress(x)[0])
    restored.to(dtype=torch.float64)
    assert restored.encoder.keys.dtype == torch.float64
    assert restored.decoder.keys.dtype == torch.float64


def test_c3_padding_has_nonzero_actual_encoded_size_below_group_size():
    _, client, _ = pair('c3_sl', ratio=5)
    x = torch.randn(2, 5, 8)
    _, _, byte_count = client.compress_decompress(x)
    # One padded patch superposition (4*8 floats) plus both CLS tokens.
    assert byte_count == (4 * 8 + 2 * 8) * 4


def test_adc_remains_active_during_local_inference_and_averages_labels():
    _, client, _ = pair('adc')
    client.eval()
    features = client(torch.randn(3, 3, 16, 16))
    compressed, _, _ = client.compress_decompress(features)
    labels = client.compress_labels(torch.tensor([0, 1, 2]))
    assert compressed.shape == (1, 3, 8)
    torch.testing.assert_close(labels, torch.tensor([[1/3, 1/3, 1/3]]))


@pytest.mark.parametrize('method', ['rand_top_k', 'c3_sl', 'adc'])
def test_encoded_baselines_cross_detached_wire_and_match_connected_gradients(method):
    from ae_psl.transport import transmit
    _, client, _ = pair(method)
    client.eval()
    # ADC needs attention from its actual backbone pass.
    images = torch.randn(3, 3, 16, 16)
    features = client(images).detach().requires_grad_()
    torch.manual_seed(7)
    direct = client.compress_decompress(features)[0]
    upstream = torch.randn_like(direct)
    expected = torch.autograd.grad((direct * upstream).sum(), features)[0]
    torch.manual_seed(7)
    payload, context = client.encode_for_transport(features)
    wire = [transmit(part, requires_grad=part.is_floating_point())[0] for part in payload]
    decoded = client.decode_from_transport(wire, context)
    torch.testing.assert_close(decoded, direct)
    (decoded * upstream).sum().backward()
    assert features.grad is None
    gradients = [transmit(part.grad)[0] for part in wire if part.grad is not None]
    outputs = [part for part in payload if part.requires_grad]
    torch.autograd.backward(outputs, gradients)
    torch.testing.assert_close(features.grad, expected)
    # Indices are counted honestly; small fixtures may have net framing overhead.
    assert sum(part.numel() for part in payload if part.is_floating_point()) < features.numel()


def test_integer_transport_keeps_indices_exact_and_independent():
    from ae_psl.transport import transmit
    indices = torch.tensor([[0, 1000, 2**24 + 1]], dtype=torch.int64)
    restored, count = transmit(indices)
    torch.testing.assert_close(restored, indices)
    assert restored.data_ptr() != indices.data_ptr()
    assert count > indices.numel() * indices.element_size()


@pytest.mark.parametrize('method', ['rand_top_k', 'c3_sl', 'adc'])
def test_baseline_entrypoint_eval_flags_and_local_macro_average(method):
    from ae_psl.trainers.implementations.classification.sasl_trainer_heuristic import SASLTrainerHeuristic
    _, client, server = pair(method)
    trainer = SASLTrainerHeuristic()
    data = [(torch.randn(1, 3, 16, 16), torch.tensor([0])),
            (torch.randn(3, 3, 16, 16), torch.tensor([1, 2, 0]))]
    expected = trainer.test_epoch_global_evaluation(client, server, data, 'cpu', None, 0, False, False)
    with torch.no_grad():
        predictions = server.forward_uncompressed(client(torch.cat([x for x, _ in data])))
        labels = torch.cat([y for _, y in data])
        assert expected[0] == pytest.approx(nn.functional.cross_entropy(predictions, labels).item())
        assert expected[1] == pytest.approx((predictions.argmax(1) == labels).float().mean().item())
    result = trainer.test_epoch_distributed({0:client, 1:copy.deepcopy(client)}, server,
        {0:data[:1], 1:data[1:]}, 'cpu', None, 0, False, True)
    assert result[0] == pytest.approx((result[2][0]['loss'] + result[2][1]['loss']) / 2)
    assert result[1] == pytest.approx((result[2][0]['acc'] + result[2][1]['acc']) / 2)
    assert result[2][1]['samples'] == 3


@pytest.mark.parametrize('method', ['rand_top_k', 'c3_sl', 'adc'])
def test_baseline_training_matches_original_sample_weighted_connected_graph(method):
    from ae_psl.trainers.implementations.classification.sasl_trainer_heuristic import SASLTrainerHeuristic
    _, client, server = pair(method)
    clients = {0:client, 1:copy.deepcopy(client)}
    references, reference_server = copy.deepcopy(clients), copy.deepcopy(server)
    batches = {0:(torch.randn(2, 3, 16, 16), torch.tensor([0, 1])),
               1:(torch.randn(3, 3, 16, 16), torch.tensor([1, 2, 0]))}
    torch.manual_seed(39)
    outputs, targets = [], []
    for cid, model in references.items():
        features = model(batches[cid][0])
        reconstructed = model.compress_decompress(features)[0]
        prediction = reference_server.forward_uncompressed(reconstructed)
        if method == 'adc':
            _, inverse = torch.unique(model.compression_layer.cluster_ids, sorted=True, return_inverse=True)
            prediction = prediction[inverse]
        outputs.append(prediction)
        targets.append(batches[cid][1])
    loss = nn.functional.cross_entropy(torch.cat(outputs), torch.cat(targets))
    loss.backward()
    for model in [*references.values(), reference_server]:
        torch.optim.SGD(model.parameters(), lr=.01).step()
    torch.manual_seed(39)
    result = SASLTrainerHeuristic()._single_epoch('cpu', server,
        torch.optim.SGD(server.parameters(), lr=.01), [batches], clients, True,
        {cid:torch.optim.SGD(model.parameters(), lr=.01) for cid, model in clients.items()},
        {}, 1, 0, {'nr_of_clients':2, 'dataset':'fixture', 'small_test_run':False, 'freeze_clientside':False})
    assert result[0] / 5 == pytest.approx(loss.item())
    assert result[7] == {0:2, 1:3}
    assert result[2] == sum(result[5].values())
    assert result[3] == sum(result[4].values())
    for model, expected in zip([*clients.values(),server], [*references.values(),reference_server]):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize('identical', [False, True])
def test_adc_multicluster_handles_real_package_api_and_empty_clusters(identical):
    _, client, _ = pair('adc')
    client.eval()
    features = client(torch.randn(6, 3, 16, 16))
    if identical:
        client.compression_layer.attn_wrapper.class_token_attention.fill_(.2)
    reconstructed, _, _ = client.compress_decompress(features)
    predictions = torch.randn(len(reconstructed), 3)
    expanded = client.expand_predictions(predictions)
    assert torch.isfinite(reconstructed).all()
    assert expanded.shape == (6, 3)
    assert 1 <= reconstructed.shape[0] <= 3


@pytest.mark.parametrize('rate', [0, -0.1, 1.1])
def test_randtopk_rejects_invalid_retained_fraction(rate):
    with pytest.raises(ValueError):
        RandomTopKModifier(rate)


@pytest.mark.parametrize('ratio', [0, -2, 1.5])
def test_c3_rejects_invalid_group_size(ratio):
    with pytest.raises(ValueError):
        pair('c3_sl', ratio=ratio)
