from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.load_hetionet import load_hetionet


def main() -> None:
    data, id_to_index = load_hetionet()

    print("=" * 60)
    print("Hetionet loaded")
    print("=" * 60)
    print(data)

    print("\nNode counts by kind:")
    for kind in data.node_types:
        print(f"  {kind:25s} {data[kind].num_nodes:>7d}")

    print("\nEdge counts by relation:")
    for etype in data.edge_types:
        src, rel, dst = etype
        n = data[etype].edge_index.size(1)
        print(f"  {src:10s} -[{rel:20s}]-> {dst:10s} {n:>7d}")

    print("\nSample Compound nodes:")
    for i in range(5):
        print(f"  [{i}] {data['Compound'].name[i]}  (id={data['Compound'].identifier[i]})")

    print("\nSample Disease nodes:")
    for i in range(5):
        print(f"  [{i}] {data['Disease'].name[i]}  (id={data['Disease'].identifier[i]})")

    ctd_types = [et for et in data.edge_types if et[0] == "Compound" and et[2] == "Disease"]
    print(f"\nCompound->Disease edge types: {ctd_types}")

    if ("Compound", "Gene", "Gene") in id_to_index:
        pass

    ddr1 = [i for i, n in enumerate(data["Gene"].name) if n == "DDR1"]
    kras = [i for i, n in enumerate(data["Gene"].name) if n == "KRAS"]
    print(f"\nDDR1 gene index: {ddr1}")
    print(f"KRAS gene index: {kras}")


if __name__ == "__main__":
    main()
