from __future__ import annotations

import argparse
import json
import sys
import time
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


def load_gnn(data, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = HeteroGNN(data, hidden=saved.get("hidden", 128), n_layers=saved.get("layers", 2)).to(device)
    predictor = DotLinkPredictor().to(device)
    with torch.no_grad():
        _ = model(data)
    model.load_state_dict(ckpt["model"])
    predictor.load_state_dict(ckpt["predictor"])
    model.eval()
    predictor.eval()
    return model, predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/gnn.pt")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--n-neg", type=int, default=30)
    p.add_argument("--out", type=str, default="runs/rag_results.jsonl")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model", type=str, default="grok-4-fast-reasoning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", action="store_true", help="also run faithfulness judge")
    args = p.parse_args()

    data, _ = load_hetionet()
    split = split_ctd(data, seed=42)
    all_pos = positive_pair_set(split.train_pos, split.val_pos, split.test_pos)
    data = data.to(args.device)

    model, predictor = load_gnn(data, args.ckpt, args.device)
    with torch.no_grad():
        z = model(data)

    adj = build_adjacency(data)
    llm = LLMClient(model=args.model)

    pos_pairs = [(int(split.test_pos[0, i]), int(split.test_pos[1, i]))
                 for i in range(split.test_pos.size(1))][: args.n_test]

    neg_ei = sample_negatives(split.num_compounds, split.num_diseases, args.n_neg, positive_set=all_pos)
    neg_pairs = [(int(neg_ei[0, i]), int(neg_ei[1, i])) for i in range(neg_ei.size(1))]

    cases = [(c, d, 1) for c, d in pos_pairs] + [(c, d, 0) for c, d in neg_pairs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct_llm = 0
    n_correct_gnn = 0
    n_total = 0
    n_faithful = 0
    n_judged = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for c_idx, d_idx, label in tqdm(cases, desc="pipeline"):
            c_name = data["Compound"].name[c_idx]
            c_id = data["Compound"].identifier[c_idx]
            d_name = data["Disease"].name[d_idx]
            d_id = data["Disease"].identifier[d_idx]

            paths = extract_paths(data, adj, z, c_idx, d_idx, top_k=args.top_k)
            paths_block = verbalize_paths(data, paths)

            with torch.no_grad():
                gnn_score = predictor(z["Compound"], z["Disease"],
                                      torch.tensor([[c_idx], [d_idx]], device=args.device)).sigmoid().item()
            gnn_pred = 1 if gnn_score > 0.5 else 0

            try:
                resp = llm.predict(c_name, c_id, d_name, d_id, paths_block)
            except Exception as e:
                resp = None
                err = str(e)
            else:
                err = None

            record = {
                "compound": {"idx": c_idx, "name": c_name, "id": c_id},
                "disease": {"idx": d_idx, "name": d_name, "id": d_id},
                "label": label,
                "gnn_score": gnn_score,
                "gnn_pred": gnn_pred,
                "n_paths": len(paths),
                "paths_block": paths_block,
                "llm_prediction": resp.prediction if resp else None,
                "llm_confidence": resp.confidence if resp else None,
                "llm_rationale": resp.rationale if resp else None,
                "error": err,
            }

            if resp and args.judge and paths:
                judge = llm.judge_faithfulness(paths_block, resp.rationale)
                record["judge"] = judge
                n_judged += 1
                if judge.get("faithful"):
                    n_faithful += 1

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            llm_pred = 1 if (resp and resp.prediction == "yes") else 0
            n_total += 1
            if llm_pred == label:
                n_correct_llm += 1
            if gnn_pred == label:
                n_correct_gnn += 1

    print(f"\nN = {n_total}")
    print(f"GNN-only accuracy: {n_correct_gnn / max(1, n_total):.4f}")
    print(f"GNN-RAG+LLM accuracy: {n_correct_llm / max(1, n_total):.4f}")
    if n_judged:
        print(f"LLM faithfulness rate: {n_faithful / n_judged:.4f}  (n={n_judged})")
    print(f"\nper-case results written to: {out_path}")


if __name__ == "__main__":
    main()
