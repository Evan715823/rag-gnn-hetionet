from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero


def _safe_key(node_type: str) -> str:
    return node_type.replace(" ", "_")


class HomoGNN(nn.Module):
    def __init__(self, hidden: int, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv((-1, -1), hidden) for _ in range(n_layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class HeteroGNN(nn.Module):
    def __init__(self, data: HeteroData, hidden: int = 128, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.node_types = list(data.node_types)
        self._keys = {nt: _safe_key(nt) for nt in self.node_types}
        self.embs = nn.ModuleDict({
            self._keys[nt]: nn.Embedding(data[nt].num_nodes, hidden)
            for nt in self.node_types
        })
        for emb in self.embs.values():
            nn.init.xavier_uniform_(emb.weight)
        self.gnn = to_hetero(HomoGNN(hidden, n_layers, dropout), data.metadata(), aggr="sum")

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {nt: self.embs[self._keys[nt]].weight for nt in self.node_types}
        return self.gnn(x_dict, data.edge_index_dict)
