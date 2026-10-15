import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ae_psl.utils.mpsl_utils import get_communication_size


class StoreAttentionWrapper(nn.Module):
    """
    Wraps a torchvision nn.MultiheadAttention module to force it to return
    attention weights and stores the class token attention.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.class_token_attention = None

    def forward(self, *args, **kwargs):
        kwargs['need_weights'] = True

        attn_output, attn_weights = self.mha(*args, **kwargs)

        # Extract attention from the class token (index 0) to all other tokens
        # PyTorch MHA with need_weights=True already averages across heads by default
        self.class_token_attention = attn_weights[:, 0, :].detach()

        return attn_output, attn_weights


class AttentionCompressionLayer(nn.Module):
    """
    Compresses tokens and batches based on attention centroids.
    """

    def __init__(self, prev_block: nn.Module, batch_compression: float, token_compression: float):
        super().__init__()
        self.batch_compression = batch_compression
        self.token_compression = token_compression
        self.cluster_ids = None

        # Explicitly store these to sync with label compression later
        self.n_new_batches = 0
        self.n_new_tokens = 0

        # Dynamically wrap the previous block's attention module
        self.attn_wrapper = StoreAttentionWrapper(prev_block.self_attention)
        prev_block.self_attention = self.attn_wrapper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batches, n_tokens, _ = x.size()

        self.n_new_tokens = max(1, int(self.token_compression * n_tokens))
        self.n_new_batches = max(1, int(self.batch_compression * n_batches))
        device = x.device

        class_token_attention = self.attn_wrapper.class_token_attention.to(device)

        # Retain individual samples when batch compression cannot form groups.
        if self.n_new_batches >= n_batches or n_batches < 2:
            # Bypass K-Means: treat each individual image as its own cluster.
            # This ensures token compression still occurs so sequence lengths match across clients.
            self.n_new_batches = n_batches
            self.cluster_ids = torch.arange(n_batches, device=device)
            centroids = class_token_attention
        elif self.n_new_batches == 1:
            self.cluster_ids = torch.zeros(n_batches, dtype=torch.long, device=device)
            centroids = class_token_attention.mean(dim=0, keepdim=True)
        else:
            # Bounded Lloyd iterations. Keep empty centroids unchanged instead
            # of taking an empty mean (NaN); singleton clusters stay rank two.
            with torch.no_grad():
                centroids = class_token_attention[
                    torch.randperm(n_batches, device=device)[:self.n_new_batches]
                ].clone()
                for _ in range(500):
                    distances = (class_token_attention[:, None] - centroids[None]).square().sum(-1)
                    ids = distances.argmin(dim=1)
                    updated = centroids.clone()
                    for index in range(self.n_new_batches):
                        members = class_token_attention[ids == index]
                        if len(members):
                            updated[index] = members.mean(dim=0)
                    shift = (updated - centroids).square().sum(dim=1).sqrt().sum()
                    centroids = updated
                    if shift.square() < 1e-4:
                        break
                self.cluster_ids = (class_token_attention[:, None] - centroids[None]).square().sum(-1).argmin(1)

        clustered_activations = []

        for cluster_id in range(self.n_new_batches):
            mask = self.cluster_ids == cluster_id

            if not mask.any():
                continue

            cluster_activations = x[mask]
            average_activation = cluster_activations.mean(dim=0)
            cluster_class_token_attention = centroids[cluster_id, 1:]

            top_k_tokens = torch.topk(
                cluster_class_token_attention,
                k=self.n_new_tokens - 1,
                largest=True,
                sorted=False
            ).indices

            top_k_tokens_indexes = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                top_k_tokens + 1
            ])

            clustered_activations.append(average_activation[top_k_tokens_indexes, :])

        if len(clustered_activations) == 0:
            return x

        return torch.stack(clustered_activations, dim=0)

class ClientModel(nn.Module):
    def __init__(self, centralized_base_model, device, split_layer, batch_compression: float, token_compression: float,
                 num_classes):
        super().__init__()
        self.device = device
        self.split_layer = split_layer

        # 1. Input Processing
        self.hidden_dim = centralized_base_model.vit.hidden_dim
        self.conv_proj = copy.deepcopy(centralized_base_model.vit.conv_proj)
        self.image_size = centralized_base_model.vit.image_size

        # 2. Embeddings & Tokens
        self.class_token = copy.deepcopy(centralized_base_model.vit.class_token)
        self.pos_embedding = copy.deepcopy(centralized_base_model.vit.encoder.pos_embedding)
        self.dropout = copy.deepcopy(centralized_base_model.vit.encoder.dropout)

        # 3. Transformer Blocks (up to split)
        client_layers = []
        for i in range(self.split_layer):
            client_layers.append(centralized_base_model.vit.encoder.layers[i])
        self.blocks = copy.deepcopy(nn.Sequential(*client_layers))

        # 4. Attach Attention Compression to the last client block
        if self.split_layer > 0:
            last_client_block = self.blocks[-1]
            self.compression_layer = AttentionCompressionLayer(
                prev_block=last_client_block,
                batch_compression=batch_compression,
                token_compression=token_compression
            )
        else:
            self.compression_layer = None

        self.num_classes = num_classes

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

        # Standard forward pass through client blocks
        x = self.blocks(x)

        return x

    def encode_for_transport(self, x):
        compressed = self.compression_layer(x) if self.compression_layer is not None else x
        return (compressed,), {}

    def decode_from_transport(self, payload, context):
        return payload[0]

    def compress_decompress(self, x):
        payload, context = self.encode_for_transport(x)
        result = self.decode_from_transport(payload, context)
        return result, 0.0, get_communication_size(result)

    def expand_predictions(self, predictions):
        """Return one prediction per original sample using client-local cluster IDs."""
        if self.compression_layer is None or self.compression_layer.cluster_ids is None:
            return predictions
        _, inverse = torch.unique(self.compression_layer.cluster_ids, sorted=True, return_inverse=True)
        return predictions[inverse]

    def compress_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.compression_layer is None or self.compression_layer.cluster_ids is None:
            return labels

        cluster_ids = self.compression_layer.cluster_ids

        n_new_batches = self.compression_layer.n_new_batches
        new_labels = []

        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        for cluster_id in range(n_new_batches):
            mask = cluster_ids == cluster_id

            if not mask.any():
                continue

            cluster_labels = one_hot_labels[mask]
            cluster_average_label = cluster_labels.mean(dim=0)
            new_labels.append(cluster_average_label)

        if len(new_labels) == 0:
            return one_hot_labels

        return torch.stack(new_labels, dim=0)

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self


class ServerModel(nn.Module):
    def __init__(self, centralized_base_model, device, split_layer):
        super().__init__()
        self.device = device

        # Get all encoder layers from split_layer onwards
        server_layers = list(centralized_base_model.vit.encoder.layers)[split_layer:]
        self.server_blocks = copy.deepcopy(nn.Sequential(*server_layers))

        self.final_layer_norm = copy.deepcopy(centralized_base_model.vit.encoder.ln)
        self.heads = copy.deepcopy(centralized_base_model.vit.heads)

        self.to(device)

    def forward_uncompressed(self, x):
        x = x.to(self.device)

        # Run remaining blocks
        x = self.server_blocks(x)

        # Final Layer Norm and Head
        x = self.final_layer_norm(x)
        x = x[:, 0]
        x = self.heads(x)

        return x

    def forward(self, x):
        return self.forward_uncompressed(x)

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self