from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data.load_hetionet import load_hetionet
from data.splits import positive_pair_set, split_ctd
from models.hetero_gnn import HeteroGNN
from models.link_predictor import DotLinkPredictor


@torch.no_grad()
def hits_at_k(scores_pos: torch.Tensor, scores_neg_per_pos: torch.Tensor, k: int = 10) -> float:
    pos = scores_pos.unsqueeze(1)
    ranks = (scores_neg_per_pos >= pos).sum(dim=1) + 1
    return float((ranks <= k).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/gnn.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-neg-per-pos", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data, _ = load_hetionet()
    split = split_ctd(data, seed=42)
    all_pos = positive_pair_set(split.train_pos, split.val_pos, split.test_pos)
    data = data.to(args.device)
    test_pos = split.test_pos.to(args.device)

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

    z = model(data)
    pos_scores = predictor(z["Compound"], z["Disease"], test_pos).sigmoid().cpu()

    n_pos = test_pos.size(1)
    g = torch.Generator().manual_seed(args.seed)
    neg_per_pos = []
    for i in range(n_pos):
        c = int(test_pos[0, i])
        chosen = []
        while len(chosen) < args.n_neg_per_pos:
            d = int(torch.randint(0, split.num_diseases, (1,), generator=g))
            if (c, d) not in all_pos:
                chosen.append(d)
        c_col = torch.full((args.n_neg_per_pos,), c, dtype=torch.long)
        d_col = torch.tensor(chosen, dtype=torch.long)
        neg_per_pos.append(torch.stack([c_col, d_col]))
    neg_per_pos = torch.stack(neg_per_pos).to(args.device)

    neg_scores = torch.zeros(n_pos, args.n_neg_per_pos)
    for i in range(n_pos):
        neg_scores[i] = predictor(z["Compound"], z["Disease"], neg_per_pos[i]).sigmoid().cpu()

    h10 = hits_at_k(pos_scores, neg_scores, k=10)
    h3 = hits_at_k(pos_scores, neg_scores, k=3)
    h1 = hits_at_k(pos_scores, neg_scores, k=1)

    y_true = torch.cat([torch.ones(n_pos), torch.zeros(n_pos * args.n_neg_per_pos)])
    y_pred = torch.cat([pos_scores, neg_scores.flatten()])
    auroc = roc_auc_score(y_true.numpy(), y_pred.numpy())
    auprc = average_precision_score(y_true.numpy(), y_pred.numpy())

    print(f"AUROC  {auroc:.4f}")
    print(f"AUPRC  {auprc:.4f}")
    print(f"Hits@1  {h1:.4f}")
    print(f"Hits@3  {h3:.4f}")
    print(f"Hits@10 {h10:.4f}")


if __name__ == "__main__":
    main()
