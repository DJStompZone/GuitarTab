#!/usr/bin/env python3
"""
Experiment 5 / Exp B: Model Confidence Analysis – Analysis Script

Loads per-TAB-step logit stats saved by inference.py (logit_stats.pt),
computes aggregate statistics, and generates publication-quality figures.

Usage (single-run mode)
-----------------------
python scripts/analyze_logit_stats.py \
    --stats_path outputs/<run>/logit_stats.pt \
    --output_dir outputs/<run>/logit_analysis \
    [--vocab_size 886]

Usage (compare mode – Exp B)
-----------------------------
python scripts/analyze_logit_stats.py \
    --compare \
    --constrained_stats outputs/<constrained_run>/logit_stats.pt \
    --unconstrained_stats outputs/<unconstrained_run>/logit_stats.pt \
    --output_dir outputs/expB_comparison \
    [--vocab_size 886]

Outputs (in --output_dir)
------------------------
  summary.json                 – aggregate stats dict
  fig_prob_valid_mass.png      – histogram: probability mass on valid TABs
  fig_entropy_within_valid.png – histogram: entropy within valid TABs
  fig_kl_from_uniform.png      – histogram: KL from uniform
  fig_margin.png               – histogram: top-1/top-2 logit gap
  fig_by_ambiguity.png         – bar chart stratified by ambiguity level
  fig_entropy_vs_ambiguity.png – scatter: entropy vs num_valid_tabs
  fig_free_choice.png          – bar chart: free argmax validity / match rate

Additional outputs in compare mode:
  fig_compare_overview.png     – side-by-side constrained vs unconstrained
  compare_summary.json         – both summaries
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
from src.logit_stats import aggregate_logit_stats, print_logit_stats_summary


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


def _records_to_arrays(records):
    keys = [
        "prob_valid_mass", "entropy_within_valid", "max_entropy_within_valid",
        "kl_from_uniform", "margin_within_valid", "num_valid_tabs",
        "free_choice_is_valid", "free_choice_matches_constrained",
    ]
    arr = {}
    for k in keys:
        vals = [r[k] for r in records if k in r]
        if vals:
            arr[k] = np.array(vals, dtype=float)
    return arr


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_prob_valid_mass(arr, vocab_size: int, out_dir: Path):
    """
    Histogram of total softmax probability mass on valid TABs.
    Shows chance level (|valid_tabs| / vocab_size) as reference.
    """
    vals = arr.get("prob_valid_mass")
    if vals is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=50, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.4)

    if vocab_size > 0 and "num_valid_tabs" in arr:
        chance_vals = arr["num_valid_tabs"] / vocab_size
        chance_mean = float(chance_vals.mean())
        ax.axvline(chance_mean, color="#DD8452", linestyle="--", linewidth=1.5,
                   label=f"Chance level (mean {chance_mean:.4f})")
        ax.legend()

    mean_val = float(vals.mean())
    ax.axvline(mean_val, color="#55A868", linestyle="-", linewidth=1.5,
               label=f"Mean = {mean_val:.4f}")
    ax.legend()

    ax.set_xlabel("Probability Mass on Valid TABs")
    ax.set_ylabel("Count")
    ax.set_title("Model's Probability Mass on Pitch-Valid TABs\n(higher = model learned pitch constraints)")
    _savefig(fig, out_dir / "fig_prob_valid_mass.png")


def plot_entropy_within_valid(arr, out_dir: Path):
    """
    Histogram of entropy within valid TABs, with max-entropy reference.
    """
    ent = arr.get("entropy_within_valid")
    max_ent = arr.get("max_entropy_within_valid")
    if ent is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: raw entropy
    ax = axes[0]
    ax.hist(ent, bins=50, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(float(ent.mean()), color="#55A868", linestyle="-", linewidth=1.5,
               label=f"Mean = {ent.mean():.4f}")
    if max_ent is not None:
        ax.axvline(float(max_ent.mean()), color="#DD8452", linestyle="--", linewidth=1.5,
                   label=f"Max entropy (mean) = {max_ent.mean():.4f}")
    ax.legend(fontsize=9)
    ax.set_xlabel("Entropy [nats]")
    ax.set_ylabel("Count")
    ax.set_title("Entropy within Valid TABs")

    # Right: normalized entropy (0=confident, 1=uniform)
    if max_ent is not None:
        mask = max_ent > 0
        if mask.any():
            norm_ent = ent[mask] / max_ent[mask]
            ax2 = axes[1]
            ax2.hist(norm_ent, bins=50, color="#C44E52", alpha=0.85, edgecolor="white", linewidth=0.4)
            ax2.axvline(float(norm_ent.mean()), color="#55A868", linestyle="-", linewidth=1.5,
                        label=f"Mean = {norm_ent.mean():.4f}")
            ax2.axvline(1.0, color="#DD8452", linestyle="--", linewidth=1.5, label="Uniform (1.0)")
            ax2.legend(fontsize=9)
            ax2.set_xlabel("Normalized Entropy (H / log|valid|)")
            ax2.set_ylabel("Count")
            ax2.set_title("Normalized Entropy\n(0 = fully confident, 1 = uniform)")

    fig.suptitle("Model Confidence Within Valid TABs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_entropy_within_valid.png")


def plot_kl_from_uniform(arr, out_dir: Path):
    """
    KL divergence from uniform distribution over valid TABs.
    KL > 0 shows the model has non-trivial preferences within valid choices.
    """
    vals = arr.get("kl_from_uniform")
    if vals is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=50, color="#8172B2", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(float(vals.mean()), color="#55A868", linestyle="-", linewidth=1.5,
               label=f"Mean = {vals.mean():.4f} nats")
    ax.axvline(0.0, color="#DD8452", linestyle="--", linewidth=1.5, label="Random baseline (0)")
    ax.legend()
    ax.set_xlabel("KL Divergence from Uniform [nats]")
    ax.set_ylabel("Count")
    ax.set_title("KL(P_model || Uniform) over Valid TABs\n(>0 = model uses context to discriminate)")
    _savefig(fig, out_dir / "fig_kl_from_uniform.png")


def plot_margin(arr, out_dir: Path):
    """
    Distribution of top-1 minus top-2 logit gap within valid TABs.
    """
    vals = arr.get("margin_within_valid")
    if vals is None:
        return

    # Restrict to ambiguity > 1 (margin is 0 by definition for ambi=1)
    ambi = arr.get("num_valid_tabs")
    if ambi is not None:
        mask = ambi > 1
        vals = vals[mask]
    if len(vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=50, color="#64B5CD", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(float(vals.mean()), color="#55A868", linestyle="-", linewidth=1.5,
               label=f"Mean = {vals.mean():.4f}")
    ax.legend()
    ax.set_xlabel("Logit Margin (Top-1 − Top-2 within valid TABs)")
    ax.set_ylabel("Count")
    ax.set_title("Model Decision Margin within Valid TABs\n(ambiguity ≥ 2 only; higher = more confident)")
    _savefig(fig, out_dir / "fig_margin.png")


def plot_by_ambiguity(summary, out_dir: Path):
    """
    Bar chart of key metrics stratified by ambiguity (num_valid_tabs).
    """
    by_ambi = summary.get("by_ambiguity", {})
    if not by_ambi:
        return

    labels = sorted(by_ambi.keys())
    metrics = [
        ("entropy_within_valid_mean", "Entropy within valid [nats]", "#4C72B0"),
        ("kl_from_uniform_mean",      "KL from uniform [nats]",     "#C44E52"),
        ("prob_valid_mass_mean",       "P(valid) mass",              "#55A868"),
        ("free_choice_is_valid_mean",  "Free argmax is valid (%)",   "#8172B2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric_key, ylabel, color) in zip(axes, metrics):
        vals = []
        errs = []
        counts = []
        for lbl in labels:
            bucket = by_ambi[lbl]
            vals.append(bucket.get(metric_key, float("nan")))
            err_key = metric_key.replace("_mean", "_std")
            errs.append(bucket.get(err_key, 0.0))
            counts.append(bucket.get("count", 0))

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=color, alpha=0.8, edgecolor="white")

        # Annotate with counts
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(errs) * 0.05 if max(errs) > 0 else bar.get_height() * 0.02,
                f"n={count:,}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("ambi_", "").replace("_", "-") + " valid" for l in labels])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Ambiguity (# valid TABs)")
        title_map = {
            "Entropy within valid [nats]": "Entropy ↑ with Ambiguity",
            "KL from uniform [nats]": "KL from Uniform",
            "P(valid) mass": "Prob Mass on Valid TABs",
            "Free argmax is valid (%)": "Freq. Free Argmax is Valid",
        }
        ax.set_title(title_map.get(ylabel, ylabel))
        if "(%)" in ylabel:
            ax.set_ylim(0, 1.1)

    fig.suptitle("Logit Statistics by Ambiguity Level", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_by_ambiguity.png")


def plot_entropy_vs_ambiguity_scatter(arr, out_dir: Path):
    """
    Scatter / box plot: entropy_within_valid vs num_valid_tabs.
    """
    ent = arr.get("entropy_within_valid")
    ambi = arr.get("num_valid_tabs")
    if ent is None or ambi is None:
        return

    unique_ambi = sorted(set(int(a) for a in ambi))
    # Cap at 8 for readability
    unique_ambi = [a for a in unique_ambi if a <= 8]

    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [ent[ambi == a] for a in unique_ambi]
    box_data = [d for d in box_data if len(d) > 0]
    plot_labels = [str(a) for a, d in zip(unique_ambi, box_data) if len(d) > 0]

    bp = ax.boxplot(box_data, labels=plot_labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 0.8},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})

    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(box_data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Overlay max-entropy line (log(ambi))
    max_ent_theory = [np.log(a) for a in unique_ambi if a > 0]
    ax.plot(range(1, len(unique_ambi) + 1), max_ent_theory, "r--",
            linewidth=1.5, label="log(|valid|) = max entropy")
    ax.legend()

    ax.set_xlabel("Number of Valid TABs (Ambiguity)")
    ax.set_ylabel("Entropy within Valid TABs [nats]")
    ax.set_title("Model Entropy vs. Ambiguity Level\n(below red dashed = model uses context beyond random)")
    _savefig(fig, out_dir / "fig_entropy_vs_ambiguity.png")


def plot_free_choice_analysis(arr, summary, out_dir: Path):
    """
    Bar chart: free-argmax validity and match-with-constrained rates.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    free_valid = summary.get("free_choice_is_valid_mean", 0.0)
    free_match = summary.get("free_choice_matches_constrained_mean", 0.0)

    bars = ax.bar(
        ["Free argmax\nis valid TAB", "Free argmax ==\nConstrained choice"],
        [free_valid * 100, free_match * 100],
        color=["#4C72B0", "#55A868"],
        alpha=0.85,
        edgecolor="white",
        width=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(
        "Free-Decoding vs Constrained-Decoding at TAB Steps\n"
        "(Left: constraint sometimes saves invalid; Right: constraint sometimes changes valid choice)"
    )
    _savefig(fig, out_dir / "fig_free_choice.png")


def plot_overview_panel(arr, summary, vocab_size: int, out_dir: Path):
    """
    Single 2×3 overview figure suitable for a paper appendix.
    """
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    # --- (0,0) prob_valid_mass histogram ---
    ax00 = fig.add_subplot(gs[0, 0])
    if "prob_valid_mass" in arr:
        vals = arr["prob_valid_mass"]
        ax00.hist(vals, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
        if vocab_size > 0 and "num_valid_tabs" in arr:
            chance = float((arr["num_valid_tabs"] / vocab_size).mean())
            ax00.axvline(chance, color="#DD8452", ls="--", lw=1.5, label=f"Chance {chance:.4f}")
        ax00.axvline(float(vals.mean()), color="#55A868", ls="-", lw=1.5,
                     label=f"Mean {vals.mean():.4f}")
        ax00.legend(fontsize=8)
        ax00.set_xlabel("P(valid mass)")
        ax00.set_title("(a) Prob Mass on Valid TABs")

    # --- (0,1) normalized entropy ---
    ax01 = fig.add_subplot(gs[0, 1])
    if "entropy_within_valid" in arr and "max_entropy_within_valid" in arr:
        mask = arr["max_entropy_within_valid"] > 0
        if mask.any():
            norm_ent = arr["entropy_within_valid"][mask] / arr["max_entropy_within_valid"][mask]
            ax01.hist(norm_ent, bins=40, color="#C44E52", alpha=0.85, edgecolor="white", linewidth=0.3)
            ax01.axvline(float(norm_ent.mean()), color="#55A868", ls="-", lw=1.5,
                         label=f"Mean {norm_ent.mean():.4f}")
            ax01.axvline(1.0, color="#DD8452", ls="--", lw=1.5, label="Uniform")
            ax01.set_xlim(0, 1.05)
            ax01.legend(fontsize=8)
            ax01.set_xlabel("Normalized Entropy")
            ax01.set_title("(b) Normalized Entropy\n(0=confident, 1=uniform)")

    # --- (0,2) KL from uniform ---
    ax02 = fig.add_subplot(gs[0, 2])
    if "kl_from_uniform" in arr:
        vals = arr["kl_from_uniform"]
        ax02.hist(vals, bins=40, color="#8172B2", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax02.axvline(float(vals.mean()), color="#55A868", ls="-", lw=1.5,
                     label=f"Mean {vals.mean():.4f}")
        ax02.axvline(0.0, color="#DD8452", ls="--", lw=1.5, label="Random (0)")
        ax02.legend(fontsize=8)
        ax02.set_xlabel("KL from Uniform [nats]")
        ax02.set_title("(c) KL(Model || Uniform)\nover Valid TABs")

    # --- (1,0) margin ---
    ax10 = fig.add_subplot(gs[1, 0])
    if "margin_within_valid" in arr and "num_valid_tabs" in arr:
        vals = arr["margin_within_valid"][arr["num_valid_tabs"] > 1]
        if len(vals) > 0:
            ax10.hist(vals, bins=40, color="#64B5CD", alpha=0.85, edgecolor="white", linewidth=0.3)
            ax10.axvline(float(vals.mean()), color="#55A868", ls="-", lw=1.5,
                         label=f"Mean {vals.mean():.4f}")
            ax10.legend(fontsize=8)
            ax10.set_xlabel("Logit Margin (T1−T2) [logit]")
            ax10.set_title("(d) Decision Margin\n(ambiguity ≥ 2)")

    # --- (1,1) entropy vs ambiguity box ---
    ax11 = fig.add_subplot(gs[1, 1])
    if "entropy_within_valid" in arr and "num_valid_tabs" in arr:
        ent = arr["entropy_within_valid"]
        ambi = arr["num_valid_tabs"]
        unique_ambi = sorted(set(int(a) for a in ambi if a <= 7))
        box_data = [ent[ambi == a] for a in unique_ambi]
        box_data_filtered = [(a, d) for a, d in zip(unique_ambi, box_data) if len(d) > 0]
        if box_data_filtered:
            ambi_f, data_f = zip(*box_data_filtered)
            bp = ax11.boxplot(data_f, labels=[str(a) for a in ambi_f], patch_artist=True,
                              medianprops={"color": "black", "lw": 1.2},
                              flierprops={"marker": ".", "ms": 1.5, "alpha": 0.3},
                              whiskerprops={"lw": 0.7})
            cols = plt.cm.Blues(np.linspace(0.3, 0.8, len(data_f)))
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c)
            max_ent_line = [np.log(a) for a in ambi_f if a > 0]
            ax11.plot(range(1, len(ambi_f) + 1), max_ent_line, "r--", lw=1.2, label="log|valid|")
            ax11.legend(fontsize=8)
            ax11.set_xlabel("# Valid TABs (ambiguity)")
            ax11.set_ylabel("Entropy [nats]")
            ax11.set_title("(e) Entropy vs Ambiguity")

    # --- (1,2) free choice bars ---
    ax12 = fig.add_subplot(gs[1, 2])
    fv = summary.get("free_choice_is_valid_mean", 0.0)
    fm = summary.get("free_choice_matches_constrained_mean", 0.0)
    bars = ax12.bar(
        ["Free\nargmax\nvalid", "Free ==\nConstrained"],
        [fv * 100, fm * 100],
        color=["#4C72B0", "#55A868"],
        alpha=0.85,
        edgecolor="white",
        width=0.5,
    )
    for bar in bars:
        ax12.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax12.set_ylim(0, 115)
    ax12.set_ylabel("%")
    ax12.set_title("(f) Free vs Constrained\nDecoding")

    fig.suptitle("Experiment 5: Model Confidence & Context Utilization",
                 fontsize=14, fontweight="bold")
    _savefig(fig, out_dir / "fig_overview.png")


# ---------------------------------------------------------------------------
# Exp B: Compare constrained vs unconstrained TAB logit stats
# ---------------------------------------------------------------------------

def plot_compare_overview(
    arr_c, summary_c,
    arr_u, summary_u,
    vocab_size: int,
    out_dir: Path,
):
    """
    Side-by-side comparison of constrained vs unconstrained TAB logit stats.

    Compares prob_valid_mass, entropy_within_valid, margin_within_valid,
    and free_choice_is_valid across the two conditions.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    conditions = [
        ("Constrained\n(correct context)", arr_c, summary_c, "#4C72B0"),
        ("Unconstrained\n(free generation)", arr_u, summary_u, "#C44E52"),
    ]

    metrics_cfg = [
        ("prob_valid_mass",         "Probability Mass on Valid TABs",        "P(valid mass)"),
        ("entropy_within_valid",    "Entropy within Valid TABs [nats]",      "Entropy [nats]"),
        ("margin_within_valid",     "Decision Margin (Top-1 − Top-2)",       "Logit margin"),
        ("free_choice_is_valid",    "Free Argmax Is Valid TAB (%)",          "%"),
    ]

    for ax_idx, (metric_key, title, xlabel) in enumerate(metrics_cfg):
        ax = fig.add_subplot(gs[ax_idx // 2, ax_idx % 2])

        vals_list = []
        labels = []
        for cond_name, arr, _, _ in conditions:
            v = arr.get(metric_key)
            if v is not None:
                # Filter ambiguity > 1 for margin
                if metric_key == "margin_within_valid" and "num_valid_tabs" in arr:
                    v = v[arr["num_valid_tabs"] > 1]
                vals_list.append(v)
                labels.append(cond_name)

        if not vals_list:
            ax.set_visible(False)
            continue

        for vals, (cond_name, _, _, color) in zip(vals_list, conditions):
            if "%" in xlabel:
                continue  # handled as bar chart below
            ax.hist(vals, bins=40, alpha=0.6, label=cond_name, edgecolor="white", linewidth=0.3)
            ax.axvline(float(vals.mean()), linestyle="-", linewidth=1.5,
                       label=f"{cond_name.split(chr(10))[0]} mean={vals.mean():.4f}")

        if "%" in xlabel:
            bar_vals = []
            bar_labels = []
            bar_colors = []
            for vals, (cond_name, _, _, color) in zip(vals_list, conditions):
                bar_vals.append(float(vals.mean()) * 100)
                bar_labels.append(cond_name.replace("\n", " "))
                bar_colors.append(color)
            bars = ax.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.8, edgecolor="white")
            for bar, v in zip(bars, bar_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 115)
            ax.set_ylabel("%")
        else:
            ax.legend(fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")

        ax.set_title(title)

    fig.suptitle(
        "Experiment B: TAB Token Confidence – Constrained vs Unconstrained\n"
        "(Constrained: correct pitch+time structural context; "
        "Unconstrained: free generation)",
        fontsize=12, fontweight="bold",
    )
    _savefig(fig, out_dir / "fig_compare_overview.png")


def plot_compare_summary_bars(summary_c, summary_u, out_dir: Path):
    """
    Bar chart directly comparing mean values of key metrics between conditions.
    """
    metric_keys = [
        ("prob_valid_mass_mean",                 "P(valid mass)",       False),
        ("normalized_entropy_mean",              "Normalized Entropy",  False),
        ("kl_from_uniform_mean",                 "KL from Uniform",     False),
        ("margin_within_valid_mean",             "Logit Margin",        False),
        ("free_choice_is_valid_mean",            "Free Argmax Valid",   True),
        ("free_choice_matches_constrained_mean", "Free == Constrained", True),
    ]

    labels = [m[0].replace("_mean", "").replace("_", " ") for m in metric_keys]
    labels = [m[2] if not m[2] else m[2] for m in metric_keys]  # use human label
    labels = [m[1] for m in metric_keys]
    vals_c = [summary_c.get(m[0], 0.0) * (100 if m[2] else 1) for m in metric_keys]
    vals_u = [summary_u.get(m[0], 0.0) * (100 if m[2] else 1) for m in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_c = ax.bar(x - width / 2, vals_c, width, label="Constrained", color="#4C72B0", alpha=0.85, edgecolor="white")
    bars_u = ax.bar(x + width / 2, vals_u, width, label="Unconstrained", color="#C44E52", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.set_title(
        "Exp B Summary: Constrained vs Unconstrained TAB Token Logit Stats\n"
        "(*% metrics: higher = model more aligned with pitch constraints)",
        fontsize=11,
    )

    _savefig(fig, out_dir / "fig_compare_summary_bars.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_single(stats_path: Path, out_dir: Path, vocab_size: int):
    print(f"Loading stats from: {stats_path}")
    records = torch.load(stats_path, map_location="cpu")
    print(f"Loaded {len(records):,} TAB step records")

    summary = aggregate_logit_stats(records, vocab_size=vocab_size)
    print_logit_stats_summary(summary)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {out_dir / 'summary.json'}")

    arr = _records_to_arrays(records)

    print("\nGenerating figures...")
    plot_prob_valid_mass(arr, vocab_size, out_dir)
    plot_entropy_within_valid(arr, out_dir)
    plot_kl_from_uniform(arr, out_dir)
    plot_margin(arr, out_dir)
    plot_by_ambiguity(summary, out_dir)
    plot_entropy_vs_ambiguity_scatter(arr, out_dir)
    plot_free_choice_analysis(arr, summary, out_dir)
    plot_overview_panel(arr, summary, vocab_size, out_dir)
    print(f"\nAll outputs in: {out_dir}")
    return records, summary, arr


def main():
    parser = argparse.ArgumentParser(description="Analyze Exp 5 / Exp B logit stats")
    # Single-run mode
    parser.add_argument("--stats_path", type=str, default=None,
                        help="Path to logit_stats.pt (single-run mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for outputs (default: sibling of stats_path)")
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="Output vocabulary size for chance-level computation")
    # Compare mode (Exp B)
    parser.add_argument("--compare", action="store_true",
                        help="Enable compare mode: constrained vs unconstrained")
    parser.add_argument("--constrained_stats", type=str, default=None,
                        help="[compare mode] Path to constrained logit_stats.pt")
    parser.add_argument("--unconstrained_stats", type=str, default=None,
                        help="[compare mode] Path to unconstrained logit_stats.pt")
    args = parser.parse_args()

    _set_style()

    if args.compare:
        if not args.constrained_stats or not args.unconstrained_stats:
            parser.error("--compare requires --constrained_stats and --unconstrained_stats")

        c_path = Path(args.constrained_stats)
        u_path = Path(args.unconstrained_stats)
        out_dir = Path(args.output_dir) if args.output_dir else c_path.parent.parent / "expB_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=== Constrained condition ===")
        c_subdir = out_dir / "constrained"
        c_subdir.mkdir(exist_ok=True)
        _, summary_c, arr_c = _run_single(c_path, c_subdir, args.vocab_size)

        print("\n=== Unconstrained condition ===")
        u_subdir = out_dir / "unconstrained"
        u_subdir.mkdir(exist_ok=True)
        _, summary_u, arr_u = _run_single(u_path, u_subdir, args.vocab_size)

        print("\nGenerating comparison figures...")
        plot_compare_overview(arr_c, summary_c, arr_u, summary_u, args.vocab_size, out_dir)
        plot_compare_summary_bars(summary_c, summary_u, out_dir)

        with open(out_dir / "compare_summary.json", "w") as f:
            json.dump({"constrained": summary_c, "unconstrained": summary_u}, f, indent=2)
        print(f"\nComparison outputs in: {out_dir}")

    else:
        if not args.stats_path:
            parser.error("--stats_path is required in single-run mode")

        stats_path = Path(args.stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        out_dir = Path(args.output_dir) if args.output_dir else stats_path.parent / "logit_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        _run_single(stats_path, out_dir, args.vocab_size)


if __name__ == "__main__":
    main()
