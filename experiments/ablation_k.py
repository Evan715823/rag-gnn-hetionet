from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from data.load_hetionet import load_hetionet
from data.splits import positive_pair_set, sample_negatives, split_ctd
from llm.client import LLMClient
from models.hetero_gnn import HeteroGNN
from models.link_predictor import DotLinkPredictor
from retrieval.subgraph_extractor import build_adjacency, extract_paths
from retrieval.verbalizer import verbalize_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/gnn.pt")
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 3, 5, 10])
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="runs/ablation_k.json")
    args = p.parse_args()

    data, _ = load_hetionet()
    split = split_ctd(data, seed=42)
    all_pos = positive_pair_set(split.train_pos, split.val_pos, split.test_pos)
    data = data.to(args.device)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    saved = ckpt.get("args", {})
    model = HeteroGNN(data, hidden=saved.get("hidden", 128), n_layers=saved.get("layers", 2)).to(args.device)
    predictor = DotLinkPredictor().to(args.device)
    with torch.no_grad():
        _ = model(data)
    model.load_state_dict(ckpt["model"])
    predictor.load_state_dict(ckpt["predictor"])
    model.eval()
    predictor.eval()
    with torch.no_grad():
        z = model(data)

    adj = build_adjacency(data)
    llm = LLMClient()

    pos_pairs = [(int(split.test_pos[0, i]), int(split.test_pos[1, i]))
                 for i in range(split.test_pos.size(1))][: args.n_test]
    neg_ei = sample_negatives(split.num_compounds, split.num_diseases, args.n_neg, positive_set=all_pos)
    neg_pairs = [(int(neg_ei[0, i]), int(neg_ei[1, i])) for i in range(neg_ei.size(1))]
    cases = [(c, d, 1) for c, d in pos_pairs] + [(c, d, 0) for c, d in neg_pairs]

    results = {}
    for k in args.ks:
        n_correct, n_total = 0, 0
        for c_idx, d_idx, label in tqdm(cases, desc=f"k={k}"):
            c_name = data["Compound"].name[c_idx]
            c_id = data["Compound"].identifier[c_idx]
            d_name = data["Disease"].name[d_idx]
            d_id = data["Disease"].identifier[d_idx]
            if k == 0:
                paths_block = "(no retrieval performed — answer from your own knowledge)"
            else:
                paths = extract_paths(data, adj, z, c_idx, d_idx, top_k=k)
                paths_block = verbalize_paths(data, paths)
            try:
                resp = llm.predict(c_name, c_id, d_name, d_id, paths_block)
                pred = 1 if resp.prediction == "yes" else 0
            except Exception:
                pred = 0
            n_total += 1
            if pred == label:
                n_correct += 1
        acc = n_correct / max(1, n_total)
        results[str(k)] = {"accuracy": acc, "n": n_total}
        print(f"k={k:3d}  accuracy={acc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
