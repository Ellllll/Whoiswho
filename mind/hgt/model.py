import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv, Linear


class HGTEncoder(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=769, num_layers=2, heads=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_lin = nn.ModuleDict()
        for node_type, input_dim in in_dims.items():
            self.node_lin[node_type] = Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        hidden = {
            node_type: F.relu(self.node_lin[node_type](node_features))
            for node_type, node_features in x_dict.items()
        }
        for layer in self.layers:
            hidden = layer(hidden, edge_index_dict)
            hidden = {node_type: self.dropout(F.relu(node_features)) for node_type, node_features in hidden.items()}
        return hidden


def edge_label_loss(embeddings, edge_label_index, edge_label, source_type, target_type, temperature=0.2):
    source_embeddings = F.normalize(embeddings[source_type], p=2, dim=-1)
    target_embeddings = F.normalize(embeddings[target_type], p=2, dim=-1)
    source_index, target_index = edge_label_index

    if source_index.numel() == 0 or target_index.numel() == 0:
        return source_embeddings.new_zeros(())

    logits = (source_embeddings[source_index] * target_embeddings[target_index]).sum(dim=-1)
    logits = logits / temperature
    labels = edge_label.to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels)


def edge_reconstruction_loss(embeddings, edge_index, source_type, target_type, temperature=0.2):
    source_embeddings = embeddings[source_type]
    target_embeddings = embeddings[target_type]
    source_index, target_index = edge_index

    if source_index.numel() == 0 or target_embeddings.shape[0] == 0:
        return source_embeddings.new_zeros(())

    source_embeddings = F.normalize(source_embeddings, p=2, dim=-1)
    target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)

    positive_scores = (source_embeddings[source_index] * target_embeddings[target_index]).sum(dim=-1)

    negative_target_index = torch.randint(
        low=0,
        high=target_embeddings.shape[0],
        size=target_index.shape,
        device=target_index.device,
    )
    if target_embeddings.shape[0] > 1:
        collision_mask = negative_target_index == target_index
        while collision_mask.any():
            replacement = torch.randint(
                low=0,
                high=target_embeddings.shape[0],
                size=(int(collision_mask.sum().item()),),
                device=target_index.device,
            )
            negative_target_index[collision_mask] = replacement
            collision_mask = negative_target_index == target_index
    negative_scores = (source_embeddings[source_index] * target_embeddings[negative_target_index]).sum(dim=-1)

    positive_scores = positive_scores / temperature
    negative_scores = negative_scores / temperature

    positive_loss = F.binary_cross_entropy_with_logits(positive_scores, torch.ones_like(positive_scores))
    negative_loss = F.binary_cross_entropy_with_logits(negative_scores, torch.zeros_like(negative_scores))
    return positive_loss + negative_loss