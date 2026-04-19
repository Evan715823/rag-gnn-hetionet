from __future__ import annotations

from dataclasses import dataclass

EdgeType = tuple[str, str, str]


@dataclass(frozen=True)
class Metapath:
    name: str
    edges: tuple[EdgeType, ...]

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def node_types(self) -> tuple[str, ...]:
        out = [self.edges[0][0]]
        for _, _, dst in self.edges:
            out.append(dst)
        return tuple(out)


METAPATHS: list[Metapath] = [
    Metapath("CpD", (
        ("Compound", "palliates", "Disease"),
    )),
    Metapath("CbGaD", (
        ("Compound", "binds", "Gene"),
        ("Gene", "associates", "Disease"),
    )),
    Metapath("CuGuD", (
        ("Compound", "upregulates", "Gene"),
        ("Gene", "upregulates", "Disease"),
    )),
    Metapath("CdGdD", (
        ("Compound", "downregulates", "Gene"),
        ("Gene", "downregulates", "Disease"),
    )),
    Metapath("CuGdD", (
        ("Compound", "upregulates", "Gene"),
        ("Gene", "downregulates", "Disease"),
    )),
    Metapath("CdGuD", (
        ("Compound", "downregulates", "Gene"),
        ("Gene", "upregulates", "Disease"),
    )),
    Metapath("CrCtD", (
        ("Compound", "resembles", "Compound"),
        ("Compound", "treats", "Disease"),
    )),
    Metapath("CbGiGaD", (
        ("Compound", "binds", "Gene"),
        ("Gene", "interacts", "Gene"),
        ("Gene", "associates", "Disease"),
    )),
    Metapath("CbGpPpG-free", (
        ("Compound", "binds", "Gene"),
        ("Gene", "participates", "Pathway"),
        ("Pathway", "participates", "Gene"),
    )),
]
