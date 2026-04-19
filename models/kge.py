from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData


@dataclass
class FlatTriples:
    head: torch.Tensor
    rel: torch.Tensor
    tail: torch.Tensor
    num_nodes: int
    num_rels: int
    node_type_offsets: dict[str, int]
    rel_to_id: dict[tuple[str, str, str], int]

    def global_id(self, node_type: str, idx: int) -> int:
        return self.node_type_offsets[node_type] + idx


def flatten_hetero(data: HeteroData) -> FlatTriples:
    node_type_offsets: dict[str, int] = {}
    offset = 0
    for nt in data.node_types:
        node_type_offsets[nt] = offset
        offset += data[nt].num_nodes
    num_nodes = offset

    rel_to_id: dict[tuple[str, str, str], int] = {
        et: i for i, et in enumerate(data.edge_types)
    }
    num_rels = len(rel_to_id)

    heads, rels, tails = [], [], []
    for et in data.edge_types:
        src, _, dst = et
        ei = data[et].edge_index.cpu()
        rid = rel_to_id[et]
        heads.append(ei[0] + node_type_offsets[src])
        rels.append(torch.full((ei.size(1),), rid, dtype=torch.long))
        tails.append(ei[1] + node_type_offsets[dst])

    return FlatTriples(
        head=torch.cat(heads),
        rel=torch.cat(rels),
        tail=torch.cat(tails),
        num_nodes=num_nodes,
        num_rels=num_rels,
        node_type_offsets=node_type_offsets,
        rel_to_id=rel_to_id,
    )
