from __future__ import annotations

from torch_geometric.data import HeteroData

from retrieval.subgraph_extractor import Path

EDGE_TEMPLATES: dict[str, str] = {
    "binds": "{src} binds to {dst}",
    "upregulates": "{src} upregulates {dst}",
    "downregulates": "{src} downregulates {dst}",
    "associates": "{src} is associated with {dst}",
    "treats": "{src} treats {dst}",
    "palliates": "{src} palliates {dst}",
    "interacts": "{src} interacts with {dst}",
    "regulates": "{src} regulates {dst}",
    "participates": "{src} participates in {dst}",
    "causes": "{src} causes {dst}",
    "resembles": "{src} is structurally similar to {dst}",
    "expresses": "{src} expresses {dst}",
    "localizes": "{src} localizes to {dst}",
    "presents": "{src} presents with {dst}",
    "includes": "{src} belongs to {dst}",
    "covaries": "{src} covaries with {dst}",
}


def _node_label(data: HeteroData, node_type: str, idx: int) -> str:
    name = data[node_type].name[idx]
    return f"{name} [{node_type}]"


def verbalize_edge(data: HeteroData, edge_type: tuple[str, str, str],
                   src_idx: int, dst_idx: int) -> str:
    _, rel, _ = edge_type
    template = EDGE_TEMPLATES.get(rel, "{src} " + rel + " {dst}")
    return template.format(
        src=_node_label(data, edge_type[0], src_idx),
        dst=_node_label(data, edge_type[2], dst_idx),
    )


def verbalize_path(data: HeteroData, path: Path) -> str:
    parts = []
    for i, et in enumerate(path.edge_types):
        parts.append(verbalize_edge(data, et, path.node_indices[i], path.node_indices[i + 1]))
    return " → ".join(parts)


def verbalize_paths(data: HeteroData, paths: list[Path]) -> str:
    if not paths:
        return "(no reasoning paths found between the two entities)"
    lines = []
    for i, p in enumerate(paths, 1):
        lines.append(f"{i}. [{p.metapath_name}, score={p.score:.3f}]  {verbalize_path(data, p)}")
    return "\n".join(lines)
