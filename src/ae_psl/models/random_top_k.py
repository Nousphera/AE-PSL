import torch.nn as nn
import numpy as np
import torch


# From https://github.com/Federico-Alvetreti/Split-Learning/blob/main/methods/random_top_k.py

# Selects the Top-K of a tensor with a random portion
class RandomTopKModifier(nn.Module):
    def __init__(self, rate: float, random_portion: float = 0.1):
        super(RandomTopKModifier, self).__init__()
        if not 0 < rate <= 1:
            raise ValueError("Retained fraction must lie in (0, 1]")
        self.rate = rate
        self.random_portion = random_portion

    def selected_indices(self, x: torch.Tensor):
        flat = x.reshape(x.shape[0], -1)
        size = flat.shape[1]
        k = max(1, round(self.rate * size))
        if k == size:
            return torch.arange(size, device=x.device).expand(len(x), -1)
        top = flat.abs().topk(k, sorted=False).indices
        probs = torch.full_like(flat, self.random_portion / (size - k))
        probs.scatter_(1, top, (1 - self.random_portion) / k)
        return torch.multinomial(probs, k)

    def select_top_k(self, x: torch.Tensor):
        flat = x.reshape(x.shape[0], -1)
        indices = self.selected_indices(x)
        return torch.zeros_like(flat).scatter(1, indices, flat.gather(1, indices)).reshape_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.select_top_k(x)


    # Function to get the compression level of this method
    def get_compression_level(self, batch_elements):


        # lower rate means more compression
        # Compute forward and backward compression   (following https://arxiv.org/pdf/2305.18469)
        if self.rate == 1:
            forward_compression = 1
        else:
            forward_compression = self.rate * (1 + np.ceil(np.log2(batch_elements)) / 32)

        backward_compression = self.rate

        # Return average compression
        compression = (forward_compression + backward_compression) / 2
        compression_ratio = 1.0 / compression

        # print(f"RandomTopKModifier: forward_compression={forward_compression:.4f}, backward_compression={backward_compression:.4f}, average_compression={compression:.4f}, compression_ratio={compression_ratio:.4f}")
        return compression_ratio

