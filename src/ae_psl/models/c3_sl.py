import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae_psl.utils.mpsl_utils import get_communication_size


# adapted from https://github.com/Federico-Alvetreti/Attention-based-Double-Compression/blob/main/methods/c3_sl.py
# original paper: https://arxiv.org/abs/2207.12397


def compute_fft_flops(batch_size, signal_length):
    """Compute FLOPS for FFT operations"""
    return batch_size * signal_length * math.log2(signal_length) * 5


def compute_circular_convolution_flops(batch_size, num_signals, signal_length):
    """
    Compute FLOPS for FFT-based circular convolution
    Total: ~15*N*log2(N) + N per operation
    """
    n = signal_length
    fft_forward = compute_fft_flops(batch_size * num_signals, n)
    fft_inverse = compute_fft_flops(batch_size * num_signals, n)
    element_wise_mult = batch_size * num_signals * n
    total_flops = fft_forward + element_wise_mult + fft_inverse
    return total_flops


def batch_circular_convolution_fft(x, h):
    # x: [B//R, R, D]
    # h: [R, D] -> need to expand to [1, R, D] for broadcasting
    h = h.unsqueeze(0)
    x_fft = torch.fft.fft(x, dim=-1)
    h_fft = torch.fft.fft(h, dim=-1)
    result_fft = x_fft * h_fft
    return torch.fft.ifft(result_fft, dim=-1).real


def batch_circular_correlation_fft(x, h):
    # x: [B/R, 1, D]
    # h: [R, D]
    h = h.unsqueeze(0)  # [1, R, D]
    X = torch.fft.fft(x, dim=-1)  # [B/R, 1, D]
    H = torch.fft.fft(h, dim=-1).conj()  # [1, R, D]
    Y = X * H
    return torch.fft.ifft(Y, dim=-1).real


class Encoder(nn.Module):
    def __init__(self, R, keys, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(R, int) or R < 1:
            raise ValueError("C3 group size must be a positive integer")
        self.R = R
        self.register_buffer("keys", keys.clone())
        self.conv_flops = 0

    def forward(self, x, *args, **kwargs):
        # Flatten into B x (P*d)
        x = torch.flatten(x, start_dim=1)
        batch_dim, features_dim = x.shape

        # Reshape into (B//R) x R x (P*d)
        x = x.reshape(batch_dim // self.R, self.R, features_dim)

        batch_over_r = batch_dim // self.R
        conv_flops = compute_circular_convolution_flops(
            batch_size=batch_over_r,
            num_signals=self.R,
            signal_length=features_dim
        )
        self.conv_flops += conv_flops

        x = batch_circular_convolution_fft(x, self.keys)
        x = x.sum(dim=1)  # shape: [B//R, P*d]

        return x


class Decoder(nn.Module):
    def __init__(self, keys, num_tokens, token_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("keys", keys.clone())
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.corr_flops = 0

    def forward(self, x, *args, **kwargs):
        # from (B//R) x (P*d) -> (B//R) x 1 x (P*d)
        x = x.unsqueeze(1)

        batch_over_r = x.shape[0]
        num_signals = self.keys.shape[0]  # R
        signal_length = x.shape[-1]  # P*d

        corr_flops = compute_circular_convolution_flops(
            batch_size=batch_over_r,
            num_signals=num_signals,
            signal_length=signal_length
        )
        self.corr_flops += corr_flops

        # Decode into a (B//R) x R x (P*d) tensor
        x = batch_circular_correlation_fft(x, self.keys)

        batch_over_R, R, features_dim = x.shape
        batch_size = batch_over_R * R

        # Reshape into B x (P*d)
        x = x.reshape(batch_size, features_dim)

        # Reshape into B x P x d
        x = x.reshape(batch_size, self.num_tokens, self.token_dim)

        return x


# --- Client & Server Models ---

class ClientModel(nn.Module):
    def __init__(self, centralized_base_model, R, num_patches, hidden_dim, device, split_layer, compress_cls_token, **kwargs):
        super().__init__()
        self.device = device
        self.split_layer = split_layer
        if not isinstance(R, int) or R < 1:
            raise ValueError("C3 group size must be a positive integer")
        self.R = R
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim


        self.patch_size = centralized_base_model.vit.patch_size
        self.hidden_dim = centralized_base_model.vit.hidden_dim
        self.conv_proj = copy.deepcopy(centralized_base_model.vit.conv_proj)
        self.image_size = centralized_base_model.vit.image_size


        self.class_token = copy.deepcopy(centralized_base_model.vit.class_token)
        self.pos_embedding = copy.deepcopy(centralized_base_model.vit.encoder.pos_embedding)
        self.dropout = copy.deepcopy(centralized_base_model.vit.encoder.dropout)


        client_layers = [centralized_base_model.vit.encoder.layers[i] for i in range(self.split_layer)]
        self.blocks = copy.deepcopy(nn.Sequential(*client_layers))

        self.flat_activation_size = self.num_patches * self.hidden_dim

        # Instantiate normal distribution keys and normalize
        keys = torch.normal(0, 1 / self.flat_activation_size,
                                 size=(self.R, self.flat_activation_size)).to(device)
        keys = keys / keys.norm(dim=1, keepdim=True)

        self.encoder = Encoder(self.R, keys)
        self.decoder = Decoder(keys, self.num_patches, self.hidden_dim)

        self.compress_cls_token = compress_cls_token

        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        n = x.shape[0]

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.blocks(x)
        return x

    def encode_for_transport(self, x):
        patches = x if self.compress_cls_token else x[:, 1:]
        batch, tokens, width = patches.shape
        padding = (-batch) % self.R
        if padding:
            patches = torch.cat((patches, patches.new_zeros(padding, tokens, width)))
        encoded = self.encoder(patches)
        payload = (encoded,) if self.compress_cls_token else (encoded, x[:, :1])
        return payload, {'batch': batch}

    def decode_from_transport(self, payload, context):
        # Keys are fixed, pre-shared codec state and are saved in both modules.
        patches = self.decoder(payload[0])[:context['batch']]
        return patches if self.compress_cls_token else torch.cat((payload[1], patches), dim=1)

    def compress_decompress(self, x):
        payload, context = self.encode_for_transport(x)
        reconstructed = self.decode_from_transport(payload, context)
        size = sum(part.numel() * part.element_size() for part in payload)
        return reconstructed, F.mse_loss(reconstructed, x).item(), size

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self


class ServerModel(nn.Module):
    def __init__(self, centralized_base_model, device, split_layer):
        super().__init__()
        self.device = device
        self.server_blocks = copy.deepcopy(nn.Sequential(*list(centralized_base_model.vit.encoder.layers)[split_layer:]))
        self.final_layer_norm = copy.deepcopy(centralized_base_model.vit.encoder.ln)
        self.heads = copy.deepcopy(centralized_base_model.vit.heads)
        self.to(device)

    def forward_uncompressed(self, x):
        x = self.server_blocks(x)
        x = self.final_layer_norm(x)
        x = x[:, 0]
        x = self.heads(x)
        return x

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self
