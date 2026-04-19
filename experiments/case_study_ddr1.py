from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.load_hetionet import load_hetionet
from data.splits import split_ctd
from llm.client import LLMClient
from models.hetero_gnn import HeteroGNN
from models.link_predictor import DotLinkPredictor
from retrieval.subgraph_extractor import build_adjacency, extract_paths
from retrieval.verbalizer import verbalize_paths


def find_gene_idx(data, name: str) -> int | None:
    for i, n in enumerate(data["Gene"].name):
        if n == name:
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/gnn.pt")
    p.add_argument("--gene", type=str, default="DDR1")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-llm", action="store_true")
    args = p.parse_args()

    data, _ = load_hetionet()
    split_ctd(data, seed=42)
    data = data.to(args.device)

    gene_idx = find_gene_idx(data, args.gene)
    if gene_idx is None:
        print(f"Gene {args.gene} not found")
        return
    print(f"{args.gene} gene index: {gene_idx}")

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

    gene_emb = z["Gene"][gene_idx]
    comp_emb = z["Compound"]
    dis_emb = z["Disease"]

    comp_sim = torch.nn.functional.cosine_similarity(gene_emb.unsqueeze(0), comp_emb)
    dis_sim = torch.nn.functional.cosine_similarity(gene_emb.unsqueeze(0), dis_emb)
    top_comp = torch.topk(comp_sim, k=5).indices.tolist()
    top_dis = torch.topk(dis_sim, k=5).indices.tolist()

    print(f"\nCompounds most similar to {args.gene} in embedding space:")
    for i in top_comp:
        print(f"  {data['Compound'].name[i]:30s}  sim={comp_sim[i].item():.3f}")

    print(f"\nDiseases most similar to {args.gene}:")
    for i in top_dis:
        print(f"  {data['Disease'].name[i]:40s}  sim={dis_sim[i].item():.3f}")

    adj = build_adjacency(data)
    c_idx = top_comp[0]
    d_idx = top_dis[0]
    c_name = data["Compound"].name[c_idx]
    d_name = data["Disease"].name[d_idx]
    print(f"\n--- Case: {c_name} → {d_name} ---")
    paths = extract_paths(data, adj, z, c_idx, d_idx, top_k=args.top_k)
    paths_block = verbalize_paths(data, paths)
    print(paths_block)

    if args.skip_llm:
        return
    try:
        llm = LLMClient()
    except RuntimeError as e:
        print(f"\nskip LLM ({e})")
        return
    c_id = data["Compound"].identifier[c_idx]
    d_id = data["Disease"].identifier[d_idx]
    resp = llm.predict(c_name, c_id, d_name, d_id, paths_block)
    print(f"\nLLM prediction: {resp.prediction}  (confidence={resp.confidence:.2f})")
    print(f"Rationale: {resp.rationale}")


if __name__ == "__main__":
    main()
