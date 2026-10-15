"""The offline integration entrypoint must execute the published ViT-B/32."""

import torch
import pytest
from ae_psl.smoke import create_system


def test_smoke_uses_full_paper_vit_b32_without_weight_downloads(monkeypatch):
    import torch.hub

    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *a, **k: pytest.fail("Offline model requested weights"),
    )
    torch.set_num_threads(1)
    args, base, ae = create_system()
    assert base.vit.image_size == 224
    assert base.vit.patch_size == 32
    assert base.vit.hidden_dim == 768
    assert len(base.vit.encoder.layers) == 12
    assert base.vit.encoder.layers[0].self_attention.num_heads == 12
    assert args["split_layer"] == 5
    assert args["ae_latent_dim"] == 64
    with torch.no_grad():
        features = base.retrieve_split_layer_activations(torch.randn(1, 3, 224, 224), 5)
        latent, cls = ae.encode(features)
    assert features.shape == (1, 50, 768)
    assert latent.shape == (1, 49, 64)
    assert cls.shape == (1, 1, 768)


def test_pretrained_checkpoint_errors_are_not_replaced_by_random_weights(monkeypatch):
    import torchvision
    from ae_psl.models.vision_transformer.base.vision_transformer_base import (
        VisionTransformerBase,
    )

    def unavailable(*args, **kwargs):
        assert kwargs["weights"] is torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr(torchvision.models, "vit_b_32", unavailable)
    with pytest.raises(OSError, match="checkpoint unavailable"):
        VisionTransformerBase("vit_b_32", True, 16, 32, 100, "cpu")
