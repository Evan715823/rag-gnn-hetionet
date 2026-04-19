from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.nn import ComplEx, DistMult
from tqdm import tqdm

from data.load_hetionet import load_hetionet
from data.splits import positive_pair_set, sample_negatives, split_ctd
from llm.client import LLMClient
from models.hetero_gnn import HeteroGNN
from models.kge import flatten_hetero
from models.link_predictor import DotLinkPredictor
from retrieval.subgraph_extractor import build_adjacency, extract_paths
from retrieval.verbalizer import verbalize_paths

KGE_CLS = {"distmult": DistMult, "complex": ComplEx}


def mcnemar(a_correct: list[int], b_correct: list[int]) -> tuple[int, int, float]:
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
    if n01 + n10 == 0:
        return n01, n10, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    try:
        from scipy.stats import chi2
        p = float(1 - chi2.cdf(stat, df=1))
    except ImportError:
        p = float(math.exp(-stat / 2))
    return n01, n10, p


def bootstrap_acc_ci(correct: list[int], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    arr = np.array(correct)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


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


def load_kge(data, ckpt_path: str | None, device: str):
    if ckpt_path is None or not Path(ckpt_path).exists():
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    kge_name = args.get("kge", "distmult")
    hidden = args.get("hidden", 128)
    flat = flatten_hetero(data.cpu() if data.is_cuda else data)
    model = KGE_CLS[kge_name](
        num_nodes=flat.num_nodes,
        num_relations=flat.num_rels,
        hidden_channels=hidden,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, flat


def kge_score(model, flat, c_idx: int, d_idx: int, device: str) -> float:
    ctd_rel = flat.rel_to_id[("Compound", "treats", "Disease")]
    h = torch.tensor([c_idx + flat.node_type_offsets["Compound"]], device=device)
    r = torch.tensor([ctd_rel], device=device)
    t = torch.tensor([d_idx + flat.node_type_offsets["Disease"]], device=device)
    with torch.no_grad():
        return float(model(h, r, t).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn-ckpt", type=str, default="checkpoints/gnn.pt")
    p.add_argument("--kge-ckpt", type=str, default="checkpoints/distmult.pt")
    p.add_argument("--n-pos", type=int, default=75)
    p.add_argument("--n-neg", type=int, default=75)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="runs/main_results.jsonl")
    p.add_argument("--model", type=str, default="grok-4-fast-reasoning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", action="store_true")
    args = p.parse_args()

    data, _ = load_hetionet()
    split = split_ctd(data, seed=42)
    all_pos = positive_pair_set(split.train_pos, split.val_pos, split.test_pos)

    kge_model, kge_flat = load_kge(data, args.kge_ckpt, args.device)
    data = data.to(args.device)
    gnn, predictor = load_gnn(data, args.gnn_ckpt, args.device)
    with torch.no_grad():
        z = gnn(data)
    adj = build_adjacency(data)
    llm = LLMClient(model=args.model)

    n_pos = min(args.n_pos, split.test_pos.size(1))
    pos_pairs = [(int(split.test_pos[0, i]), int(split.test_pos[1, i])) for i in range(n_pos)]
    neg_ei = sample_negatives(split.num_compounds, split.num_diseases, args.n_neg, positive_set=all_pos)
    neg_pairs = [(int(neg_ei[0, i]), int(neg_ei[1, i])) for i in range(neg_ei.size(1))]
    cases = [(c, d, 1) for c, d in pos_pairs] + [(c, d, 0) for c, d in neg_pairs]

    NO_RETRIEVAL = "(no reasoning paths provided — answer using your internal knowledge only)"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(out_path, "w", encoding="utf-8") as f:
        for c_idx, d_idx, label in tqdm(cases, desc="main"):
            c_name = data["Compound"].name[c_idx]
            c_id = data["Compound"].identifier[c_idx]
            d_name = data["Disease"].name[d_idx]
            d_id = data["Disease"].identifier[d_idx]

            with torch.no_grad():
                gnn_s = predictor(z["Compound"], z["Disease"],
                                  torch.tensor([[c_idx], [d_idx]], device=args.device)).sigmoid().item()
            gnn_pred = int(gnn_s > 0.5)

            kge_s = None
            kge_pred = None
            if kge_model is not None:
                kge_s = kge_score(kge_model, kge_flat, c_idx, d_idx, args.device)
                kge_pred = int(kge_s > 0.0)

            paths = extract_paths(data, adj, z, c_idx, d_idx, top_k=args.top_k)
            paths_block = verbalize_paths(data, paths) if paths else NO_RETRIEVAL

            llm_only_resp = None
            llm_rag_resp = None
            try:
                llm_only_resp = llm.predict(c_name, c_id, d_name, d_id, NO_RETRIEVAL)
            except Exception as e:
                llm_only_err = str(e)
            else:
                llm_only_err = None
            try:
                llm_rag_resp = llm.predict(c_name, c_id, d_name, d_id, paths_block)
            except Exception as e:
                llm_rag_err = str(e)
            else:
                llm_rag_err = None

            llm_only_pred = int(llm_only_resp.prediction == "yes") if llm_only_resp else 0
            llm_rag_pred = int(llm_rag_resp.prediction == "yes") if llm_rag_resp else 0

            judge = None
            if args.judge and llm_rag_resp and paths:
                try:
                    judge = llm.judge_faithfulness(paths_block, llm_rag_resp.rationale)
                except Exception as e:
                    judge = {"faithful": None, "error": str(e)}

            rec = {
                "compound": {"idx": c_idx, "name": c_name, "id": c_id},
                "disease": {"idx": d_idx, "name": d_name, "id": d_id},
                "label": label,
                "gnn": {"score": gnn_s, "pred": gnn_pred},
                "kge": {"score": kge_s, "pred": kge_pred},
                "paths_block": paths_block,
                "n_paths": len(paths),
                "llm_only": {
                    "pred": llm_only_pred,
                    "confidence": llm_only_resp.confidence if llm_only_resp else None,
                    "rationale": llm_only_resp.rationale if llm_only_resp else None,
                    "error": llm_only_err,
                },
                "llm_rag": {
                    "pred": llm_rag_pred,
                    "confidence": llm_rag_resp.confidence if llm_rag_resp else None,
                    "rationale": llm_rag_resp.rationale if llm_rag_resp else None,
                    "error": llm_rag_err,
                },
                "judge": judge,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    y = [r["label"] for r in records]
    gnn_c = [int(r["gnn"]["pred"] == r["label"]) for r in records]
    kge_c = [int(r["kge"]["pred"] == r["label"]) if r["kge"]["pred"] is not None else None for r in records]
    kge_c = [c for c in kge_c if c is not None]
    llm_only_c = [int(r["llm_only"]["pred"] == r["label"]) for r in records]
    llm_rag_c = [int(r["llm_rag"]["pred"] == r["label"]) for r in records]

    print(f"\n=== N = {len(records)} ({sum(y)} positive, {len(y) - sum(y)} negative) ===")
    for name, correct in [("GNN-only", gnn_c), ("LLM-only", llm_only_c), ("GNN-RAG+LLM", llm_rag_c)] + (
        [("KGE (DistMult)", kge_c)] if kge_c else []
    ):
        acc, lo, hi = bootstrap_acc_ci(correct)
        print(f"  {name:18s} acc = {acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n=== McNemar paired tests ===")
    comparisons = [
        ("GNN-only", gnn_c, "LLM-only", llm_only_c),
        ("GNN-only", gnn_c, "GNN-RAG+LLM", llm_rag_c),
        ("LLM-only", llm_only_c, "GNN-RAG+LLM", llm_rag_c),
    ]
    for name_a, a, name_b, b in comparisons:
        n01, n10, p = mcnemar(a, b)
        tag = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {name_a} vs {name_b:18s}  a_wrong_b_right={n01:3d}  a_right_b_wrong={n10:3d}  p={p:.4g} {tag}")

    if args.judge:
        judged = [r for r in records if r.get("judge") and isinstance(r["judge"].get("faithful"), bool)]
        if judged:
            faithful_rate = sum(1 for r in judged if r["judge"]["faithful"]) / len(judged)
            print(f"\nLLM faithfulness: {faithful_rate:.4f}  (n={len(judged)})")

    print(f"\nper-case results -> {out_path}")


if __name__ == "__main__":
    main()
