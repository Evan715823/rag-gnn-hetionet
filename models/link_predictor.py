from __future__ import annotations

import torch
from torch import nn


class DotLinkPredictor(nn.Module):
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = z_src[edge_index[0]]
        dst = z_dst[edge_index[1]]
        return (src * dst).sum(dim=-1)


class MLPLinkPredictor(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = z_src[edge_index[0]]
        dst = z_dst[edge_index[1]]
        return self.net(torch.cat([src, dst], dim=-1)).squeeze(-1)
