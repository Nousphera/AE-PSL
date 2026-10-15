import torch
import torchvision
from torch import nn

from ae_psl.models.lora_layer import LoRALinear
from ae_psl.utils.model_utils import SupportedModel


class VisionTransformerBase(nn.Module):
    def __init__(
        self,
        vit_type: str,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: int,
        num_classes: int,
        device,
        *,
        pretrained=True,
    ):
        super().__init__()
        self.use_lora = use_lora
        self.num_classes = num_classes

        if vit_type != SupportedModel.VIT_B_32.value:
            raise ValueError(f"Unsupported paper backbone: {vit_type}; use vit_b_32")
        # An explicit offline run uses the identical architecture from scratch.
        # Download errors in pretrained runs propagate; there is no fallback.
        weights = (
            torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.vit = torchvision.models.vit_b_32(weights=weights)
        self.pretrained = pretrained

        self.device = device

        # Replace with new head for the desired number of classes
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, num_classes)

        if self.use_lora:
            # MODE A: LoRA Adaptation
            # 1. Freeze EVERYTHING first
            for param in self.vit.parameters():
                param.requires_grad = False

            # 2. Swap Encoder Linear layers to LoRALinear (adds trainable params)
            self._apply_lora_to_encoder(rank=lora_rank, alpha=lora_alpha)

            # 3. Unfreeze Head (Standard practice)
            for param in self.vit.heads.parameters():
                param.requires_grad = True

        else:
            # MODE B: Regular Fine-Tuning
            # Ensure everything is trainable (default behavior, but being explicit is safe)
            for param in self.vit.parameters():
                param.requires_grad = True

        self.vit.to(device)

    def _apply_lora_to_encoder(self, rank, alpha):
        """Recursively swaps Linear layers for LoRALinear in the encoder."""

        def replace_linear_recursion(module):
            for name, child in module.named_children():
                if isinstance(child, nn.MultiheadAttention):
                    # MHA calls functional attention with out_proj.weight directly;
                    # wrapping out_proj.forward would create unused LoRA parameters.
                    continue
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, rank, alpha))
                else:
                    replace_linear_recursion(child)

        replace_linear_recursion(self.vit.encoder)

    def retrieve_split_layer_activations(self, x, split_layer: int):

        with torch.no_grad():
            x = x.to(self.device)

            # 1. Patch Embedding Logic

            # (n, c, h, w) -> (n, hidden_dim, out_h, out_w)
            x = self.vit.conv_proj(x)
            n, hidden_dim, out_h, out_w = x.shape

            # (n, hidden_dim, out_h, out_w) -> (n, hidden_dim, (out_h * out_w))
            x = x.reshape(n, hidden_dim, out_h * out_w)

            # (n, hidden_dim, (out_h * out_w)) -> (n, (out_h * out_w), hidden_dim)
            x = x.permute(0, 2, 1)

            # 2. Add Tokens & Position Embeddings
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            torch._assert(
                x.dim() == 3,
                f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}",
            )

            x = x + self.vit.encoder.pos_embedding
            x = self.vit.encoder.dropout(x)

            # 3. Forward Pass through Transformer Encoder up to split_layer
            for i in range(split_layer):
                x = self.vit.encoder.layers[i](x)

        return x

    def get_hidden_dim(self):
        return self.vit.hidden_dim
