import copy

import torch.nn as nn
import torch
import torch.nn.functional as F

from ae_psl.utils.mpsl_utils import get_communication_size


# Simple Uniform Quantization
class _QuantizeFunction(torch.autograd.Function):

    @staticmethod
    def quantize(x: torch.tensor, n_bits: int):
        # Handle the case of complex tensor applying the quantization to both real and imaginary parts
        if torch.is_complex(x):
            real_q = _QuantizeFunction.quantize(x.real, n_bits)
            imag_q = _QuantizeFunction.quantize(x.imag, n_bits)
            return torch.complex(real_q, imag_q)

        if n_bits == 32:
            return x
        # Get min and max
        x_min = x.min()
        x_max = x.max()

        if x_max == x_min:
            return x.clone()

        # Get levels
        levels = 2 ** n_bits

        # Get the step
        scale = (x_max - x_min) / (levels - 1)

        # Quantize: map x to integers [0, levels-1]
        q_x = torch.round((x - x_min) / scale).clamp(0, levels - 1)

        # Dequantize: map back to float domain
        quantized_x = q_x * scale + x_min

        return quantized_x



    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int):
        ctx.quantize_n_bits = n_bits
        return _QuantizeFunction.quantize(x, n_bits)

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        return _QuantizeFunction.quantize(grad_outputs, ctx.quantize_n_bits), None


class Quantization_Layer(nn.Module):

    def __init__(self, n_bits: int):
        super().__init__()
        self.n_bits = n_bits


    def forward(self, x: torch.Tensor):
        return _QuantizeFunction.apply(x, self.n_bits)





class ClientModel(nn.Module):
    def __init__(self, centralized_base_model, quant_module, sparse_module, device, split_layer, compress_cls_token, num_classes, rand_top_k_sparsity, bit_width, entropy_coding=False):
        super().__init__()
        self.device = device
        self.split_layer = split_layer

        # --- Deepcopy components from Centralized Model ---
        # 1. Input Processing
        self.patch_size = centralized_base_model.vit.patch_size
        self.hidden_dim = centralized_base_model.vit.hidden_dim
        self.conv_proj = copy.deepcopy(centralized_base_model.vit.conv_proj)
        self.image_size = centralized_base_model.vit.image_size

        # 2. Embeddings & Tokens
        self.class_token = copy.deepcopy(centralized_base_model.vit.class_token)
        self.pos_embedding = copy.deepcopy(centralized_base_model.vit.encoder.pos_embedding)
        self.dropout = copy.deepcopy(centralized_base_model.vit.encoder.dropout)

        # 3. Transformer Blocks (up to split)
        # Note: centralized_model.vit.encoder.layers has [Blocks ... AE ... Blocks]
        # We need everything BEFORE the AE.
        client_layers = []
        for i in range(self.split_layer):
            client_layers.append(centralized_base_model.vit.encoder.layers[i])
        self.blocks = copy.deepcopy(nn.Sequential(*client_layers))

        # 4. AE Encoder

        self.quant_module = copy.deepcopy(quant_module)
        self.sparse_module = copy.deepcopy(sparse_module)


        self.compress_cls_token = compress_cls_token

        self.to(device)

    def forward(self, x):
        # 1. Pre-processing (Client-side logic)
        x = x.to(self.device)

        # Access the underlying ViT components
        n = x.shape[0]

        # Patching and Embedding
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.blocks(x)

        return x

    def encode_for_transport(self, x):
        patches = x if self.compress_cls_token else x[:, 1:]
        flat = patches.flatten(1)
        if self.sparse_module is None:
            indices = torch.arange(flat.shape[1], device=x.device).expand(len(x), -1)
        else:
            indices = self.sparse_module.selected_indices(patches)
        values = flat.gather(1, indices)
        if self.quant_module is not None:
            values = self.quant_module(values)
        payload = (values, indices) if self.compress_cls_token else (values, indices, x[:, :1])
        return payload, {'shape': list(patches.shape)}

    def decode_from_transport(self, payload, context):
        values, indices = payload[:2]
        batch, tokens, width = context['shape']
        patches = values.new_zeros(batch, tokens * width).scatter(1, indices, values)
        patches = patches.reshape(batch, tokens, width)
        return patches if self.compress_cls_token else torch.cat((payload[2], patches), dim=1)

    def compress_decompress(self, x):
        payload, context = self.encode_for_transport(x)
        reconstructed = self.decode_from_transport(payload, context)
        # Raw payload bytes. Training adds framing, logits and returned gradients.
        size = sum(part.numel() * part.element_size() for part in payload)
        return reconstructed, F.mse_loss(reconstructed, x).item(), size

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self

class ServerModel(nn.Module):
    def __init__(self, centralized_base_model, device, split_layer, compress_cls_token):
        super().__init__()
        self.device = device

        # Get all encoder layers after the split_layer
        self.server_blocks = copy.deepcopy(nn.Sequential(*list(centralized_base_model.vit.encoder.layers)[split_layer:]))
        self.final_layer_norm = copy.deepcopy(centralized_base_model.vit.encoder.ln)
        self.heads = copy.deepcopy(centralized_base_model.vit.heads)

        self.compress_cls_token = compress_cls_token



    def forward_uncompressed(self, x):
        # Run remaining blocks
        x = self.server_blocks(x)

        # Final Layer Norm and Head
        x = self.final_layer_norm(x)
        x = x[:, 0]
        x = self.heads(x)

        return x

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self