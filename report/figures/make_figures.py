"""Publication-quality figures for the COMP 559 final report.

Generates (as vector PDFs under report/figures/):
    fig_pipeline.pdf  - 6-stage end-to-end pipeline diagram
    fig_bars.pdf      - 4-method accuracy bar chart with 95% CI
    fig_ablation.pdf  - top-K retrieval saturation curve

Run from the `report/figures/` directory:
    python make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

# ---------- shared typography ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # TrueType, not Type-3 (NeurIPS requires this)
    "ps.fonttype": 42,
})

CLR_INPUT = "#F5A962"
CLR_GNN   = "#4E8ABF"
CLR_RETR  = "#58B36E"
CLR_VERB  = "#E6C463"
CLR_LLM   = "#8E6FB0"
CLR_OUT   = "#C0504D"

C_LLM   = "#F77B72"
C_GNN   = "#6BA4D3"
C_KGE   = "#65C38A"
C_OURS  = "#2C3E8C"
ACCENT  = "#1F4E79"


# ============================================================================
#  Figure 1: Pipeline
# ============================================================================
def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # (x_center, y_center, title, subtitle, colour)
    stages = [
        (1.5,  4.5, "Query",          r"$(c,d)\in\mathcal{C}\!\times\!\mathcal{D}$",  CLR_INPUT),
        (5.0,  4.5, "Hetero.\\ GNN",  r"SAGEConv $\circ$ to_hetero",                   CLR_GNN),
        (8.5,  4.5, "Retrieval",      r"top-$K$ meta-paths",                           CLR_RETR),
        (12.0, 4.5, "Verbaliser",     "template-based",                                CLR_VERB),
        (12.0, 1.5, "LLM reasoner",   "grok-4-fast-reasoning",                         CLR_LLM),
        (5.75, 1.5, "Output + judge", r"$\{$pred, conf, rationale$\}$",                CLR_OUT),
    ]
    node_by_label = {}
    for xc, yc, title, sub, color in stages:
        width, height = 2.7, 1.55
        box = FancyBboxPatch(
            (xc - width / 2, yc - height / 2), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.2",
            linewidth=0.9, edgecolor="black",
            facecolor=color, alpha=0.85,
        )
        # drop-shadow
        shadow = FancyBboxPatch(
            (xc - width / 2 + 0.06, yc - height / 2 - 0.06), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.2",
            linewidth=0, edgecolor="none",
            facecolor="black", alpha=0.15, zorder=box.get_zorder() - 1,
        )
        ax.add_patch(shadow)
        ax.add_patch(box)
        ax.text(xc, yc + 0.25, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="black")
        ax.text(xc, yc - 0.30, sub, ha="center", va="center",
                fontsize=8.5, color="black")
        node_by_label[title] = (xc, yc, width, height)

    def arrow(a: str, b: str, label: str | None = None, label_offset=(0, 0)) -> None:
        xa, ya, wa, ha = node_by_label[a]
        xb, yb, wb, hb = node_by_label[b]
        # choose connection points on the facing edges
        if abs(xa - xb) > abs(ya - yb):     # horizontal arrow
            start = (xa + wa / 2 * (1 if xb > xa else -1), ya)
            end   = (xb - wb / 2 * (1 if xb > xa else -1), yb)
        else:                               # vertical arrow
            start = (xa, ya - ha / 2 * (1 if yb < ya else -1))
            end   = (xb, yb + hb / 2 * (1 if yb < ya else -1))
        a_patch = FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.1, color="#555555",
            shrinkA=2, shrinkB=2,
        )
        ax.add_patch(a_patch)
        if label:
            mx = (start[0] + end[0]) / 2 + label_offset[0]
            my = (start[1] + end[1]) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=8, fontfamily="sans-serif",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    arrow("Query", "Hetero.\\ GNN")
    arrow("Hetero.\\ GNN", "Retrieval", r"$\{\mathbf{h}_v\}$", (0, 0.25))
    arrow("Retrieval", "Verbaliser", "paths", (0.35, 0))
    arrow("Verbaliser", "LLM reasoner", "prompt", (0.45, 0))
    arrow("LLM reasoner", "Output + judge", "JSON", (0, 0.25))

    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ============================================================================
#  Figure 2: Main results bar chart
# ============================================================================
def figure_bars() -> None:
    methods = ["LLM-only", "GNN-only", "KGE\n(calibrated)", "GNN-RAG+LLM\n(ours)"]
    accs    = np.array([0.553, 0.727, 0.787, 0.873])
    ci_lo   = np.array([0.080, 0.074, 0.067, 0.060])
    ci_hi   = np.array([0.080, 0.066, 0.066, 0.047])
    colors  = [C_LLM, C_GNN, C_KGE, C_OURS]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(methods))
    bars = ax.bar(x, accs, width=0.65,
                  color=colors, edgecolor="black", linewidth=0.9,
                  yerr=[ci_lo, ci_hi], capsize=5,
                  error_kw=dict(ecolor="#444444", elinewidth=1.0))
    # hatch ours
    bars[-1].set_hatch("////")
    bars[-1].set_edgecolor("white")
    bars[-1].set_linewidth(1.2)

    # chance line
    ax.axhline(0.5, color="#B03030", linestyle="--", linewidth=1.0, alpha=0.85, zorder=0)
    ax.text(-0.45, 0.505, "chance", color="#B03030", fontsize=8.5,
            ha="left", va="bottom", alpha=0.9)

    # value labels
    for bar, v, hi in zip(bars, accs, ci_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, v + hi + 0.015,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT / "fig_bars.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ============================================================================
#  Figure 3: Top-K ablation
# ============================================================================
def figure_ablation() -> None:
    Ks   = np.array([0, 1, 3, 5, 10])
    accs = np.array([0.525, 0.750, 0.800, 0.875, 0.825])

    fig, ax = plt.subplots(figsize=(6.8, 3.3))

    # fill area
    ax.fill_between(Ks, accs, 0.40, alpha=0.18, color=ACCENT, zorder=1)
    # main curve
    ax.plot(Ks, accs, color=ACCENT, linewidth=1.8, zorder=2)
    ax.plot(Ks, accs, "o", color=ACCENT, markersize=7,
            markerfacecolor=ACCENT, markeredgecolor="white", markeredgewidth=1.1, zorder=3)

    # peak star
    ax.plot(5, 0.875, marker="*", markersize=24,
            markerfacecolor="#D03030", markeredgecolor="#701515", markeredgewidth=1.0, zorder=5)

    # per-point callouts
    ax.text(0,  0.525 - 0.025, "0.525", ha="center", va="top", fontsize=8.5, color=ACCENT)
    ax.text(1,  0.750 - 0.025, "0.750", ha="center", va="top", fontsize=8.5, color=ACCENT)
    ax.text(3,  0.800 - 0.025, "0.800", ha="center", va="top", fontsize=8.5, color=ACCENT)
    ax.text(5,  0.905, r"$K^\star=5$   0.875", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#B02020")
    ax.text(10, 0.825 - 0.025, "0.825", ha="center", va="top", fontsize=8.5, color=ACCENT)

    # drop annotation (curved arrow + label)
    drop = FancyArrowPatch(
        (5.7, 0.875), (9.5, 0.835),
        arrowstyle="-|>", mutation_scale=12,
        connectionstyle="arc3,rad=-0.28",
        color="#B02020", linewidth=1.2,
    )
    ax.add_patch(drop)
    ax.text(7.6, 0.925, r"$-5$ pts drop",
            ha="center", va="bottom", fontsize=9, color="#B02020", style="italic")

    ax.set_xlabel(r"Number of retrieved paths $K$", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks([0, 1, 3, 5, 10])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(0.45, 0.96)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    figure_pipeline()
    figure_bars()
    figure_ablation()
    for name in ("fig_pipeline.pdf", "fig_bars.pdf", "fig_ablation.pdf"):
        print(f"  wrote {(OUT / name).resolve()}")
