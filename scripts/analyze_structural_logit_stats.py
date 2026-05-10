#!/usr/bin/env python3
"""
Experiment A: Structural Token Confidence Analysis – Analysis Script

Loads per-structural-step logit stats saved by inference.py
(structural_logit_stats.pt), computes aggregate statistics, and generates
publication-quality figures stratified by token type
(NOTE_ON / NOTE_OFF / TIME_SHIFT).

Usage
-----
python scripts/analyze_structural_logit_stats.py \
    --stats_path outputs/<run>/structural_logit_stats.pt \
    [--output_dir outputs/<run>/structural_logit_analysis]

Outputs (in --output_dir)
------------------------
  summary.json                    – aggregate stats dict
  fig_prob_on_correct.png         – histogram: P(correct token) per token type
  fig_margin_over_2nd.png         – box plot: margin over 2nd best per type
  fig_free_argmax_correct.png     – bar chart: free-argmax accuracy per type
  fig_correct_rank_cdf.png        – CDF of correct token rank per type
  fig_entropy_full.png            – histogram: full-vocab entropy per type
  fig_overview.png                – 2×3 overview panel
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root
from src.logit_stats import aggregate_structural_stats, print_structural_stats_summary


TOKEN_TYPES = ["NOTE_ON", "NOTE_OFF", "TIME_SHIFT"]
COLORS = {
    "NOTE_ON":    "#4C72B0",
    "NOTE_OFF":   "#C44E52",
    "TIME_SHIFT": "#55A868",
    "ALL":        "#8172B2",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


def _savefig(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _split_by_type(records):
    """Return dict of token_type -> numpy arrays for each scalar key."""
    result = {}
    for tt in TOKEN_TYPES + ["ALL"]:
        subset = records if tt == "ALL" else [r for r in records if r.get("token_type") == tt]
        if not subset:
            continue
        arrs = {}
        for k in ("prob_on_correct", "margin_over_2nd", "free_argmax_is_correct",
                  "correct_rank", "entropy_full"):
            vals = [r[k] for r in subset if k in r]
            if vals:
                arrs[k] = np.array(vals, dtype=float)
        if arrs:
            result[tt] = arrs
    return result


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_prob_on_correct(split, out_dir: Path):
    """Histogram of P(correct token) for each structural token type."""
    fig, axes = plt.subplots(1, len(split), figsize=(5 * len(split), 4), squeeze=False)
    for ax, (tt, arrs) in zip(axes[0], split.items()):
        vals = arrs.get("prob_on_correct")
        if vals is None:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=50, color=COLORS.get(tt, "#888888"), alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(float(vals.mean()), color="black", linestyle="-", linewidth=1.5,
                   label=f"Mean = {vals.mean():.4f}")
        ax.legend(fontsize=9)
        ax.set_xlabel("P(correct token)")
        ax.set_ylabel("Count")
        ax.set_title(f"{tt}\n(n={len(vals):,})")

    fig.suptitle("Probability on Correct Structural Token", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_prob_on_correct.png")


def plot_margin_over_2nd(split, out_dir: Path):
    """Box plot of margin over 2nd best, stratified by token type."""
    data = []
    labels = []
    for tt in TOKEN_TYPES:
        arrs = split.get(tt)
        if arrs is None:
            continue
        vals = arrs.get("margin_over_2nd")
        if vals is not None:
            data.append(vals)
            labels.append(tt)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 0.8},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    for patch, tt in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS.get(tt, "#888888"))
        patch.set_alpha(0.75)

    ax.axhline(0, color="#DD8452", linestyle="--", linewidth=1.2,
               label="0 (correct = top-1 boundary)")
    ax.legend(fontsize=9)
    ax.set_ylabel("Margin (correct logit − 2nd best logit)")
    ax.set_title("Decision Margin over 2nd Best\n(positive = correct is top-1, negative = wrong top-1)")
    _savefig(fig, out_dir / "fig_margin_over_2nd.png")


def plot_free_argmax_correct(split, out_dir: Path):
    """Bar chart: rate at which unconstrained argmax == correct token."""
    labels = []
    vals = []
    for tt in TOKEN_TYPES:
        arrs = split.get(tt)
        if arrs is None:
            continue
        v = arrs.get("free_argmax_is_correct")
        if v is not None:
            labels.append(tt)
            vals.append(float(v.mean()) * 100)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals,
                  color=[COLORS.get(l, "#888888") for l in labels],
                  alpha=0.85, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Free Argmax Is Correct (%)")
    ax.set_title("Structural Token: Free Argmax Accuracy\n(% of steps where model's top-1 == correct token)")
    _savefig(fig, out_dir / "fig_free_argmax_correct.png")


def plot_correct_rank_cdf(split, out_dir: Path):
    """CDF of correct token rank for each token type."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for tt in TOKEN_TYPES:
        arrs = split.get(tt)
        if arrs is None:
            continue
        ranks = arrs.get("correct_rank")
        if ranks is None:
            continue
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        ax.step(sorted_ranks, cdf, label=f"{tt} (n={len(ranks):,})",
                color=COLORS.get(tt), linewidth=1.8)

    ax.set_xlabel("Correct Token Rank (0 = top-1)")
    ax.set_ylabel("CDF")
    ax.set_xlim(-0.5, 20)
    ax.set_title("CDF of Correct Token Rank\n(steep at 0 = model consistently predicts correct token as top-1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / "fig_correct_rank_cdf.png")


def plot_entropy_full(split, out_dir: Path):
    """Histogram of full-vocabulary entropy per structural token type."""
    fig, axes = plt.subplots(1, len([tt for tt in TOKEN_TYPES if tt in split]),
                             figsize=(5 * sum(1 for tt in TOKEN_TYPES if tt in split), 4),
                             squeeze=False)
    ax_idx = 0
    for tt in TOKEN_TYPES:
        arrs = split.get(tt)
        if arrs is None:
            continue
        vals = arrs.get("entropy_full")
        if vals is None:
            continue
        ax = axes[0][ax_idx]
        ax_idx += 1
        ax.hist(vals, bins=50, color=COLORS.get(tt, "#888888"), alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(float(vals.mean()), color="black", linestyle="-", linewidth=1.5,
                   label=f"Mean = {vals.mean():.4f}")
        ax.legend(fontsize=9)
        ax.set_xlabel("Entropy [nats]")
        ax.set_ylabel("Count")
        ax.set_title(f"{tt}\n(n={len(vals):,})")

    fig.suptitle("Full-Vocabulary Entropy at Structural Steps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_entropy_full.png")


def plot_overview(split, summary, out_dir: Path):
    """2×3 overview panel for paper appendix."""
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    # (0,0) P(correct) per type bar
    ax00 = fig.add_subplot(gs[0, 0])
    _labels = [tt for tt in TOKEN_TYPES if tt in split and "prob_on_correct" in split[tt]]
    _vals = [float(split[tt]["prob_on_correct"].mean()) for tt in _labels]
    _errs = [float(split[tt]["prob_on_correct"].std()) for tt in _labels]
    if _labels:
        bars = ax00.bar(_labels, _vals, yerr=_errs, capsize=4,
                        color=[COLORS.get(l) for l in _labels], alpha=0.8, edgecolor="white")
        for bar, v in zip(bars, _vals):
            ax00.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                      f"{v:.3f}", ha="center", fontsize=9)
    ax00.set_title("(a) Mean P(correct token)")
    ax00.set_ylabel("Probability")
    ax00.set_ylim(0, min(1.1, max(_vals) * 1.25) if _vals else 1.1)

    # (0,1) Free argmax accuracy bar
    ax01 = fig.add_subplot(gs[0, 1])
    _labels2 = [tt for tt in TOKEN_TYPES if tt in split and "free_argmax_is_correct" in split[tt]]
    _vals2 = [float(split[tt]["free_argmax_is_correct"].mean()) * 100 for tt in _labels2]
    if _labels2:
        bars2 = ax01.bar(_labels2, _vals2,
                         color=[COLORS.get(l) for l in _labels2], alpha=0.8, edgecolor="white")
        for bar, v in zip(bars2, _vals2):
            ax01.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                      f"{v:.1f}%", ha="center", fontsize=9)
    ax01.set_title("(b) Free Argmax Correct (%)")
    ax01.set_ylabel("%")
    ax01.set_ylim(0, 115)

    # (0,2) Mean margin bar
    ax02 = fig.add_subplot(gs[0, 2])
    _labels3 = [tt for tt in TOKEN_TYPES if tt in split and "margin_over_2nd" in split[tt]]
    _vals3 = [float(split[tt]["margin_over_2nd"].mean()) for tt in _labels3]
    _errs3 = [float(split[tt]["margin_over_2nd"].std()) for tt in _labels3]
    if _labels3:
        ax02.bar(_labels3, _vals3, yerr=_errs3, capsize=4,
                 color=[COLORS.get(l) for l in _labels3], alpha=0.8, edgecolor="white")
    ax02.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax02.set_title("(c) Mean Margin over 2nd Best")
    ax02.set_ylabel("Logit gap")

    # (1,0) CDF of correct rank
    ax10 = fig.add_subplot(gs[1, 0])
    for tt in TOKEN_TYPES:
        if tt not in split or "correct_rank" not in split[tt]:
            continue
        ranks = split[tt]["correct_rank"]
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        ax10.step(sorted_ranks, cdf, label=tt, color=COLORS.get(tt), linewidth=1.5)
    ax10.set_xlim(-0.5, 15)
    ax10.set_xlabel("Rank (0 = top-1)")
    ax10.set_ylabel("CDF")
    ax10.legend(fontsize=8)
    ax10.set_title("(d) CDF of Correct Token Rank")
    ax10.grid(True, alpha=0.3)

    # (1,1) Entropy box plot
    ax11 = fig.add_subplot(gs[1, 1])
    box_data = []
    box_labels = []
    for tt in TOKEN_TYPES:
        if tt not in split or "entropy_full" not in split[tt]:
            continue
        box_data.append(split[tt]["entropy_full"])
        box_labels.append(tt)
    if box_data:
        bp = ax11.boxplot(box_data, labels=box_labels, patch_artist=True,
                          medianprops={"color": "black", "lw": 1.2},
                          flierprops={"marker": ".", "ms": 1.5, "alpha": 0.3},
                          whiskerprops={"lw": 0.7})
        for patch, lbl in zip(bp["boxes"], box_labels):
            patch.set_facecolor(COLORS.get(lbl, "#888888"))
            patch.set_alpha(0.75)
    ax11.set_ylabel("Entropy [nats]")
    ax11.set_title("(e) Full-Vocab Entropy Distribution")

    # (1,2) Summary table as text
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")
    overall = summary.get("overall", {})
    lines = [
        f"n total = {summary.get('total_structural_steps', 0):,}",
        "",
        f"P(correct):  {overall.get('prob_on_correct_mean', 0):.4f}",
        f"Margin:      {overall.get('margin_over_2nd_mean', 0):.4f}",
        f"Free Acc:    {overall.get('free_argmax_is_correct_mean', 0)*100:.1f}%",
        f"Rank(mean):  {overall.get('correct_rank_mean', 0):.2f}",
        f"Entropy:     {overall.get('entropy_full_mean', 0):.4f}",
    ]
    for i, line in enumerate(lines):
        ax12.text(0.05, 0.95 - i * 0.13, line, transform=ax12.transAxes,
                  fontsize=10, verticalalignment="top", family="monospace")
    ax12.set_title("(f) Overall Summary")

    fig.suptitle("Experiment A: Structural Token Logit Statistics", fontsize=14, fontweight="bold")
    _savefig(fig, out_dir / "fig_overview.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Exp A structural logit stats")
    parser.add_argument("--stats_path", type=str, required=True,
                        help="Path to structural_logit_stats.pt saved by inference.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save figures and summary (default: <stats_path parent>/structural_logit_analysis)")
    args = parser.parse_args()

    stats_path = Path(args.stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else stats_path.parent / "structural_logit_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading structural stats from: {stats_path}")
    records = torch.load(stats_path, map_location="cpu")
    print(f"Loaded {len(records):,} structural step records")

    _set_style()

    summary = aggregate_structural_stats(records)
    print_structural_stats_summary(summary)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {out_dir / 'summary.json'}")

    split = _split_by_type(records)

    print("\nGenerating figures...")
    plot_prob_on_correct(split, out_dir)
    plot_margin_over_2nd(split, out_dir)
    plot_free_argmax_correct(split, out_dir)
    plot_correct_rank_cdf(split, out_dir)
    plot_entropy_full(split, out_dir)
    plot_overview(split, summary, out_dir)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
