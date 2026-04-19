from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_records(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def bucket(records: list[dict]) -> dict[str, list[dict]]:
    buckets = {
        "gnn_right_llm_wrong": [],
        "gnn_wrong_llm_right": [],
        "both_wrong": [],
        "both_right": [],
        "retrieval_helped": [],
        "retrieval_hurt": [],
    }
    for r in records:
        label = r["label"]
        gnn = r["gnn"]["pred"]
        llm_only = r["llm_only"]["pred"]
        llm_rag = r["llm_rag"]["pred"]
        if gnn == label and llm_rag != label:
            buckets["gnn_right_llm_wrong"].append(r)
        elif gnn != label and llm_rag == label:
            buckets["gnn_wrong_llm_right"].append(r)
        elif gnn != label and llm_rag != label:
            buckets["both_wrong"].append(r)
        else:
            buckets["both_right"].append(r)
        if llm_only != label and llm_rag == label:
            buckets["retrieval_helped"].append(r)
        elif llm_only == label and llm_rag != label:
            buckets["retrieval_hurt"].append(r)
    return buckets


def print_case(r: dict, show_paths: bool = True, show_rationale: bool = True) -> None:
    c = r["compound"]
    d = r["disease"]
    print(f"  [{c['name']} → {d['name']}]  label={r['label']}")
    print(f"    GNN pred={r['gnn']['pred']}  score={r['gnn']['score']:.3f}")
    print(f"    LLM-only pred={r['llm_only']['pred']}  conf={r['llm_only']['confidence']}")
    print(f"    LLM-RAG  pred={r['llm_rag']['pred']}  conf={r['llm_rag']['confidence']}")
    if show_paths and r.get("paths_block"):
        lines = r["paths_block"].split("\n")
        shown = lines[:3]
        print("    retrieved paths:")
        for line in shown:
            print(f"      {line}")
        if len(lines) > 3:
            print(f"      ... (+{len(lines) - 3} more)")
    if show_rationale:
        rr = r["llm_rag"].get("rationale") or ""
        print(f"    LLM-RAG rationale: {rr[:200]}")
        if r.get("judge") and isinstance(r["judge"].get("faithful"), bool):
            print(f"    judge: faithful={r['judge']['faithful']}  invented={r['judge'].get('invented_entities')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="runs/main_results.jsonl")
    p.add_argument("--per-bucket", type=int, default=5)
    p.add_argument("--bucket", type=str, default=None,
                   help="only print this bucket (gnn_right_llm_wrong, gnn_wrong_llm_right, both_wrong, both_right, retrieval_helped, retrieval_hurt)")
    args = p.parse_args()

    records = load_records(Path(args.file))
    print(f"loaded {len(records)} records from {args.file}\n")

    b = bucket(records)
    print("=== bucket counts ===")
    for name, items in b.items():
        print(f"  {name:26s} {len(items):4d}")

    print()
    targets = [args.bucket] if args.bucket else list(b.keys())
    for name in targets:
        items = b.get(name, [])
        if not items:
            continue
        print(f"\n=== {name}  (showing {min(args.per_bucket, len(items))} of {len(items)}) ===")
        for r in items[: args.per_bucket]:
            print_case(r)
            print()


if __name__ == "__main__":
    main()
