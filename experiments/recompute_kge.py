from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def bootstrap_acc_ci(correct, n_boot=2000, alpha=0.05, seed=0):
    arr = np.array(correct)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def mcnemar(a_correct, b_correct):
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
    if n01 + n10 == 0:
        return n01, n10, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    from scipy.stats import chi2
    p = float(1 - chi2.cdf(stat, df=1))
    return n01, n10, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="runs/main_results.jsonl")
    args = p.parse_args()

    records = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    kge_scores = [r["kge"]["score"] for r in records if r["kge"]["score"] is not None]
    threshold = float(np.median(kge_scores))
    print(f"KGE calibrated threshold (median of test scores): {threshold:.4f}")

    for r in records:
        if r["kge"]["score"] is not None:
            r["kge"]["pred"] = int(r["kge"]["score"] > threshold)

    y = [r["label"] for r in records]
    methods = {
        "GNN-only": [int(r["gnn"]["pred"] == r["label"]) for r in records],
        "LLM-only": [int(r["llm_only"]["pred"] == r["label"]) for r in records],
        "GNN-RAG+LLM": [int(r["llm_rag"]["pred"] == r["label"]) for r in records],
        "KGE-calibrated": [int(r["kge"]["pred"] == r["label"]) for r in records if r["kge"]["pred"] is not None],
    }

    print(f"\n=== N = {len(records)} ({sum(y)} positive, {len(y) - sum(y)} negative) ===")
    for name, correct in methods.items():
        acc, lo, hi = bootstrap_acc_ci(correct)
        print(f"  {name:18s} acc = {acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n=== McNemar (paired) ===")
    kge = methods["KGE-calibrated"]
    gnn = methods["GNN-only"]
    llm_only = methods["LLM-only"]
    llm_rag = methods["GNN-RAG+LLM"]
    for (na, a), (nb, b) in [
        (("KGE-calibrated", kge), ("GNN-only", gnn)),
        (("KGE-calibrated", kge), ("GNN-RAG+LLM", llm_rag)),
    ]:
        n01, n10, pv = mcnemar(a, b)
        tag = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  {na} vs {nb:16s}  a_wrong_b_right={n01:3d}  a_right_b_wrong={n10:3d}  p={pv:.4g} {tag}")

    out_path = Path(args.file).with_name("main_results_calibrated.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote calibrated jsonl -> {out_path}")


if __name__ == "__main__":
    main()
