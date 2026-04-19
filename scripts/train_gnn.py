from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from data.load_hetionet import load_hetionet
from data.splits import positive_pair_set, sample_negatives, split_ctd
from models.hetero_gnn import HeteroGNN
from models.link_predictor import DotLinkPredictor


def score_edges(z_dict, predictor, edge_index):
    return predictor(z_dict["Compound"], z_dict["Disease"], edge_index)


@torch.no_grad()
def evaluate(model, predictor, data, pos_edges, num_compounds, num_diseases, n_neg_mult=10, seed=0):
    model.eval()
    predictor.eval()
    z = model(data)
    pos_score = score_edges(z, predictor, pos_edges).sigmoid().cpu()
    g = torch.Generator().manual_seed(seed)
    n_neg = pos_edges.size(1) * n_neg_mult
    neg_c = torch.randint(0, num_compounds, (n_neg,), generator=g)
    neg_d = torch.randint(0, num_diseases, (n_neg,), generator=g)
    neg_ei = torch.stack([neg_c, neg_d])
    neg_score = score_edges(z, predictor, neg_ei).sigmoid().cpu()

    y_true = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).numpy()
    y_pred = torch.cat([pos_score, neg_score]).numpy()
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    return {"auroc": float(auroc), "auprc": float(auprc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--neg-ratio", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", type=str, default="checkpoints/gnn.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    print(f"device: {args.device}")
    data, _ = load_hetionet()
    split = split_ctd(data, seed=args.seed)
    print(f"train/val/test positives: {split.train_pos.size(1)} / {split.val_pos.size(1)} / {split.test_pos.size(1)}")

    all_pos = positive_pair_set(split.train_pos, split.val_pos, split.test_pos)

    data = data.to(args.device)
    split.train_pos = split.train_pos.to(args.device)
    split.val_pos = split.val_pos.to(args.device)
    split.test_pos = split.test_pos.to(args.device)

    model = HeteroGNN(data, hidden=args.hidden, n_layers=args.layers).to(args.device)
    predictor = DotLinkPredictor().to(args.device)

    with torch.no_grad():
        _ = model(data)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    best_val = -1.0
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        predictor.train()
        n_pos = split.train_pos.size(1)
        neg_ei = sample_negatives(split.num_compounds, split.num_diseases,
                                   n_pos * args.neg_ratio, positive_set=all_pos).to(args.device)

        z = model(data)
        pos_score = score_edges(z, predictor, split.train_pos)
        neg_score = score_edges(z, predictor, neg_ei)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, predictor, data, split.val_pos, split.num_compounds, split.num_diseases)
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val AUROC {val['auroc']:.4f} AUPRC {val['auprc']:.4f}")
            if val["auroc"] > best_val:
                best_val = val["auroc"]
                torch.save({
                    "model": model.state_dict(),
                    "predictor": predictor.state_dict(),
                    "args": vars(args),
                    "val_auroc": best_val,
                    "epoch": epoch,
                }, ckpt_path)

    print(f"\nbest val AUROC: {best_val:.4f}")
    print(f"ckpt saved to: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    predictor.load_state_dict(ckpt["predictor"])
    test = evaluate(model, predictor, data, split.test_pos, split.num_compounds, split.num_diseases)
    print(f"TEST  AUROC {test['auroc']:.4f}  AUPRC {test['auprc']:.4f}")


if __name__ == "__main__":
    main()
