from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from retrieval.metapath import METAPATHS, EdgeType, Metapath


@dataclass
class Path:
    node_types: tuple[str, ...]
    node_indices: tuple[int, ...]
    edge_types: tuple[EdgeType, ...]
    score: float
    metapath_name: str


def build_adjacency(data: HeteroData) -> dict[EdgeType, dict[int, list[int]]]:
    adj: dict[EdgeType, dict[int, list[int]]] = {}
    for et in data.edge_types:
        ei = data[et].edge_index.cpu()
        bucket: dict[int, list[int]] = defaultdict(list)
        for i in range(ei.size(1)):
            bucket[int(ei[0, i])].append(int(ei[1, i]))
        adj[et] = bucket
    return adj


def _enumerate_paths(
    adj: dict[EdgeType, dict[int, list[int]]],
    mp: Metapath,
    src_idx: int,
    dst_idx: int,
    max_branch: int = 200,
) -> list[tuple[int, ...]]:
    frontier: list[tuple[int, ...]] = [(src_idx,)]
    for et in mp.edges:
        next_frontier: list[tuple[int, ...]] = []
        for partial in frontier:
            last = partial[-1]
            for nb in adj.get(et, {}).get(last, []):
                next_frontier.append(partial + (nb,))
                if len(next_frontier) >= max_branch * len(frontier):
                    break
        frontier = next_frontier
        if not frontier:
            return []
    return [p for p in frontier if p[-1] == dst_idx]


def _path_score(z: dict[str, torch.Tensor], node_types: tuple[str, ...],
                node_indices: tuple[int, ...]) -> float:
    vecs = []
    for t, i in zip(node_types, node_indices):
        vecs.append(z[t][i])
    sims = []
    for a, b in zip(vecs[:-1], vecs[1:]):
        sims.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return float(sum(sims) / max(1, len(sims)))


def extract_paths(
    data: HeteroData,
    adj: dict[EdgeType, dict[int, list[int]]],
    z: dict[str, torch.Tensor],
    compound_idx: int,
    disease_idx: int,
    metapaths: Iterable[Metapath] = METAPATHS,
    top_k: int = 5,
    max_branch: int = 200,
) -> list[Path]:
    candidates: list[Path] = []
    for mp in metapaths:
        raw = _enumerate_paths(adj, mp, compound_idx, disease_idx, max_branch=max_branch)
        for nodes in raw:
            score = _path_score(z, mp.node_types, nodes)
            candidates.append(Path(
                node_types=mp.node_types,
                node_indices=nodes,
                edge_types=mp.edges,
                score=score,
                metapath_name=mp.name,
            ))
    candidates.sort(key=lambda p: p.score, reverse=True)
    return candidates[:top_k]
