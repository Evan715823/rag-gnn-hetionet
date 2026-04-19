from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

TARGET = ("Compound", "treats", "Disease")
REVERSE = ("Disease", "treats", "Compound")


@dataclass
class LinkSplit:
    train_pos: torch.Tensor
    val_pos: torch.Tensor
    test_pos: torch.Tensor
    num_compounds: int
    num_diseases: int


def split_ctd(data: HeteroData, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> LinkSplit:
    ei = data[TARGET].edge_index
    n = ei.size(1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)

    val_idx = perm[:n_val]
    test_idx = perm[n_val : n_val + n_test]
    train_idx = perm[n_val + n_test :]

    split = LinkSplit(
        train_pos=ei[:, train_idx].contiguous(),
        val_pos=ei[:, val_idx].contiguous(),
        test_pos=ei[:, test_idx].contiguous(),
        num_compounds=data["Compound"].num_nodes,
        num_diseases=data["Disease"].num_nodes,
    )

    data[TARGET].edge_index = split.train_pos
    data[REVERSE].edge_index = split.train_pos.flip(0).contiguous()

    return split


def sample_negatives(num_compounds: int, num_diseases: int, num_samples: int,
                     positive_set: set[tuple[int, int]] | None = None) -> torch.Tensor:
    c = torch.randint(0, num_compounds, (num_samples * 2,))
    d = torch.randint(0, num_diseases, (num_samples * 2,))
    if positive_set is None:
        return torch.stack([c[:num_samples], d[:num_samples]])
    kept_c, kept_d = [], []
    for i in range(c.size(0)):
        if (int(c[i]), int(d[i])) not in positive_set:
            kept_c.append(int(c[i]))
            kept_d.append(int(d[i]))
            if len(kept_c) == num_samples:
                break
    return torch.tensor([kept_c, kept_d], dtype=torch.long)


def positive_pair_set(*edge_indices: torch.Tensor) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for ei in edge_indices:
        for i in range(ei.size(1)):
            result.add((int(ei[0, i]), int(ei[1, i])))
    return result
