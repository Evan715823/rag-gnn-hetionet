from __future__ import annotations

import bz2
import json
from pathlib import Path

import requests
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

HETIONET_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "cache" / "hetionet-v1.0.json.bz2"


def download_hetionet(path: Path = DEFAULT_CACHE) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(HETIONET_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="hetionet") as pbar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                pbar.update(len(chunk))
    return path


def load_raw(path: Path = DEFAULT_CACHE) -> dict:
    with bz2.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_hetero_data(raw: dict) -> tuple[HeteroData, dict]:
    nodes_by_kind: dict[str, list] = {}
    id_to_index: dict[tuple, int] = {}

    for node in raw["nodes"]:
        kind = node["kind"]
        bucket = nodes_by_kind.setdefault(kind, [])
        id_to_index[(kind, node["identifier"])] = len(bucket)
        bucket.append(node)

    data = HeteroData()
    for kind, nodes in nodes_by_kind.items():
        data[kind].num_nodes = len(nodes)
        data[kind].name = [n.get("name", "") for n in nodes]
        data[kind].identifier = [n["identifier"] for n in nodes]

    edges_by_type: dict[tuple, list] = {}
    for edge in raw["edges"]:
        src_kind, src_id = edge["source_id"]
        dst_kind, dst_id = edge["target_id"]
        rel = edge["kind"]
        src_idx = id_to_index.get((src_kind, src_id))
        dst_idx = id_to_index.get((dst_kind, dst_id))
        if src_idx is None or dst_idx is None:
            continue
        edges_by_type.setdefault((src_kind, rel, dst_kind), []).append((src_idx, dst_idx))
        if edge.get("direction") == "both":
            edges_by_type.setdefault((dst_kind, rel, src_kind), []).append((dst_idx, src_idx))

    for edge_type, pairs in edges_by_type.items():
        ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[edge_type].edge_index = ei

    return data, id_to_index


def load_hetionet(cache_path: Path = DEFAULT_CACHE) -> tuple[HeteroData, dict]:
    path = download_hetionet(cache_path)
    raw = load_raw(path)
    return build_hetero_data(raw)


if __name__ == "__main__":
    data, _ = load_hetionet()
    print(data)
