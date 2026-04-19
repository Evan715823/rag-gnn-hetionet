from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import ComplEx, DistMult
from tqdm import tqdm

from data.load_hetionet import load_hetionet
from data.splits import positive_pair_set, split_ctd
from models.kge import flatten_hetero

MODEL_CLS = {"distmult": DistMult, "complex": ComplEx}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kge", choices=list(MODEL_CLS.keys()), default="distmult")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.ckpt is None:
        args.ckpt = f"checkpoints/{args.kge}.pt"

    torch.manual_seed(args.seed)
    data, _ = load_hetionet()
    split = split_ctd(data, seed=args.seed)
    all_pos = positive_pair_set(split.train_pos, split.val_pos, split.test_pos)
    flat = flatten_hetero(data)

    print(f"device: {args.device}")
    print(f"model: {args.kge}  hidden={args.hidden}")
    print(f"triples: {flat.head.size(0)}  nodes: {flat.num_nodes}  rels: {flat.num_rels}")

    model = MODEL_CLS[args.kge](
        num_nodes=flat.num_nodes,
        num_relations=flat.num_rels,
        hidden_channels=args.hidden,
    ).to(args.device)

    loader = model.loader(
        head_index=flat.head.to(args.device),
        rel_type=flat.rel.to(args.device),
        tail_index=flat.tail.to(args.device),
        batch_size=args.batch,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ctd_rel = flat.rel_to_id[("Compound", "treats", "Disease")]
    comp_offset = flat.node_type_offsets["Compound"]
    dis_offset = flat.node_type_offsets["Disease"]

    def eval_split(pos_edges: torch.Tensor, n_neg_mult: int = 10, seed: int = 0):
        model.eval()
        with torch.no_grad():
            n_pos = pos_edges.size(1)
            h = (pos_edges[0] + comp_offset).to(args.device)
            t = (pos_edges[1] + dis_offset).to(args.device)
            r = torch.full((n_pos,), ctd_rel, dtype=torch.long, device=args.device)
            pos_s = model(h, r, t).cpu()

            g = torch.Generator().manual_seed(seed)
            n_neg = n_pos * n_neg_mult
            neg_c = torch.randint(0, split.num_compounds, (n_neg,), generator=g)
            neg_d = torch.randint(0, split.num_diseases, (n_neg,), generator=g)
            h = (neg_c + comp_offset).to(args.device)
            t = (neg_d + dis_offset).to(args.device)
            r = torch.full((n_neg,), ctd_rel, dtype=torch.long, device=args.device)
            neg_s = model(h, r, t).cpu()

            y_true = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).numpy()
            y_pred = torch.cat([pos_s, neg_s]).numpy()
            return {
                "auroc": float(roc_auc_score(y_true, y_pred)),
                "auprc": float(average_precision_score(y_true, y_pred)),
            }

    best_val = -1.0
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for head, rel, tail in loader:
            optimizer.zero_grad()
            loss = model.loss(head, rel, tail)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        val = eval_split(split.val_pos)
        print(f"epoch {epoch:2d} | loss {total / n_batches:.4f} | val AUROC {val['auroc']:.4f} AUPRC {val['auprc']:.4f}")
        if val["auroc"] > best_val:
            best_val = val["auroc"]
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "rel_to_id": flat.rel_to_id,
                        "node_type_offsets": flat.node_type_offsets}, args.ckpt)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = eval_split(split.test_pos)
    print(f"\nTEST ({args.kge}) AUROC {test['auroc']:.4f}  AUPRC {test['auprc']:.4f}")
    print(f"ckpt: {args.ckpt}")


if __name__ == "__main__":
    main()
