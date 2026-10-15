import pytest
import torch
from ae_psl.transport import transmit


def test_tensor_transport_is_bytes_and_detaches_autograd():
    x = torch.randn(2, 5, 8, requires_grad=True)
    y, size = transmit(x, requires_grad=True)
    torch.testing.assert_close(x, y)
    assert y.is_leaf and y.data_ptr() != x.data_ptr()
    assert size > x.numel() * x.element_size()
    y.sum().backward()
    assert x.grad is None


def test_label_local_logit_gradients_equal_centralized_batch():
    torch.manual_seed(6)
    z = torch.randn(5, 3, requires_grad=True)
    y = torch.tensor([0, 2, 1, 1, 2])
    expected = torch.autograd.grad(torch.nn.functional.cross_entropy(z, y), z)[0]
    gradients = []
    for start, stop in [(0, 2), (2, 5)]:
        local, _ = transmit(z[start:stop], requires_grad=True)
        loss = torch.nn.functional.cross_entropy(local, y[start:stop])
        gradient = torch.autograd.grad(loss, local)[0] * (stop-start) / len(y)
        returned, _ = transmit(gradient)
        gradients.append(returned)
    torch.testing.assert_close(torch.cat(gradients), expected)

import copy
from ae_psl.smoke import create_system, run
from ae_psl.models.vision_transformer.implementations.unimodal.image_classification.models import get_split_model
from ae_psl.trainers.implementations.classification.sasl_trainer import SASLTrainer


def split_models():
    args, base, ae = create_system(42)
    client, server, _ = get_split_model(base, ae, 5, 100, 'cpu', False)
    return args, {i: copy.deepcopy(client) for i in range(2)}, server


def test_cls_identity_and_gradient_through_frozen_ae():
    _, _, ae = create_system()
    ae.freeze(True, True)
    ae.eval()
    x = torch.randn(1, 50, 768, requires_grad=True)
    latent, cls = ae.encode(x)
    reconstructed = ae.decode(latent, cls)
    assert torch.equal(reconstructed[:, :1], x[:, :1])
    reconstructed.sum().backward()
    torch.testing.assert_close(x.grad[:, :1], torch.ones_like(x[:, :1]))
    assert x.grad[:, 1:].abs().sum() > 0
    assert all(p.grad is None for p in ae.parameters())


def test_serialized_split_updates_equal_unsplit_autograd():
    torch.set_num_threads(1)
    args, clients, server = split_models()
    for c in clients.values():
        c.eval(); c.set_lora_trainable(); c.ae_module.freeze(True, True)
    server.eval(); server.ae_module.freeze(True, True)
    reference_clients, reference_server = copy.deepcopy(clients), copy.deepcopy(server)
    data = {i: (torch.randn(n,3,224,224), torch.arange(n)%100) for i,n in enumerate((1,2))}
    main = {i: torch.optim.SGD(c.parameters(),lr=.01) for i,c in clients.items()}
    server_opt = torch.optim.SGD(server.parameters(),lr=.01)
    zs, cs = [], []
    for i,c in reference_clients.items():
        z, cls, _ = c.compress(c(data[i][0])); zs.append(z); cs.append(cls)
    predictions = reference_server.forward_uncompressed(reference_server.decompress(torch.cat(zs),torch.cat(cs)))
    loss = torch.nn.functional.cross_entropy(predictions,torch.cat([v[1] for v in data.values()]))
    loss.backward()
    for model in [*reference_clients.values(), reference_server]:
        torch.optim.SGD(model.parameters(),lr=.01).step()
    result = SASLTrainer()._single_epoch('cpu',server,server_opt,[data],clients,True,main,{},1,1,args)
    assert result[0]/3 == pytest.approx(loss.item())
    for actual, expected in zip([*clients.values(),server],[*reference_clients.values(),reference_server]):
        for k,v in actual.state_dict().items():
            torch.testing.assert_close(v,expected.state_dict()[k],rtol=1e-5,atol=1e-7)


def test_global_eval_bypasses_ae_and_local_eval_uses_it():
    _, clients, server = split_models()
    data = [(torch.randn(1,3,224,224),torch.tensor([0]))]
    calls = []
    handle = clients[0].ae_module.patch_encoder.register_forward_hook(lambda *args: calls.append('encoder'))
    decoder_handle = server.ae_module.patch_decoder.register_forward_hook(lambda *args: calls.append('decoder'))
    trainer = SASLTrainer()
    trainer.test_epoch_global_evaluation(clients[0], server, data, 'cpu',None,1,False,False)
    assert calls == []
    trainer.test_epoch_distributed({0:clients[0]},server,{0:data},'cpu',None,1,False,True)
    assert calls == ['encoder','decoder']
    handle.remove(); decoder_handle.remove()


def test_three_stages_are_reproducible_and_preserve_invariants():
    first,_ = run(42)
    second,_ = run(42)
    assert all(first['checks'].values())
    assert first['state_sha256'] == second['state_sha256']
    assert first['sample_counts'] == {0:1,1:2}
    assert first['communication']['outgoing_client_communication_overhead_in_mb']['0'] > (49 * 64 + 768 + 100) * 4 / 2**20
