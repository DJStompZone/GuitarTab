#!/usr/bin/env python3
"""Generate figures for docs/experiment_report_v1_v2_grammar.md from embedded table data."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures" / "experiment_v1_v2_grammar"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )

    # --- Fig 1: Primary metrics (control R1/R2 vs treatment t10 R3/R4) ---
    metrics = [
        "f1",
        "tab_acc\n(aligned)",
        "pitch_acc\n(aligned)",
        "strict\ntab",
        "strict\npitch",
        "norm.\ntab",
        "norm.\npitch",
    ]
    v1_control = [0.885426, 0.761614, 0.918570, np.nan, np.nan, np.nan, np.nan]
    v2_control = [0.985960, 0.652677, 0.888020, np.nan, np.nan, np.nan, np.nan]
    v1_t10 = [0.885426, 0.761614, 0.918570, 0.748219, 0.902415, 0.761614, 0.918570]
    v2_t10 = [0.985960, 0.652677, 0.888020, 0.648108, 0.881803, 0.652677, 0.888020]

    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 1.5 * w, v1_control, w, label="v1 control (R1)", color="#4477AA")
    ax.bar(x - 0.5 * w, v2_control, w, label="v2 control (R2)", color="#CC6677")
    ax.bar(x + 0.5 * w, v1_t10, w, label="v1 treatment t10 (R3)", color="#66CCEE")
    ax.bar(x + 1.5 * w, v2_t10, w, label="v2 treatment t10 (R4)", color="#AA4499")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Primary metrics: control vs format-aware treatment (tol=10)")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_primary_metrics_control_vs_treatment.png")
    plt.close(fig)

    # --- Fig 2: Tolerance sensitivity (R5–R8 aligned metrics) ---
    tol = [5, 10, 20]
    v1_f1 = [0.885426, 0.885426, 0.886664]
    v1_tab = [0.761614, 0.761614, 0.760733]
    v1_pitch = [0.918570, 0.918570, 0.917525]
    v2_f1 = [0.985960, 0.985960, 0.985960]
    v2_tab = [0.652677, 0.652677, 0.652677]
    v2_pitch = [0.888020, 0.888020, 0.888020]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, y1, y2, title in zip(
        axes,
        [v1_f1, v1_tab, v1_pitch],
        [v2_f1, v2_tab, v2_pitch],
        ["F1", "tab_acc_aligned", "pitch_acc_aligned"],
    ):
        ax.plot(tol, y1, "o-", label="v1", color="#4477AA")
        ax.plot(tol, y2, "s--", label="v2", color="#CC6677")
        ax.set_title(title)
        ax.set_xticks(tol)
        ax.set_xlabel("timeline tolerance")
        ax.set_ylim(0.6, 1.02)
        ax.grid(alpha=0.35)
    axes[0].legend()
    fig.suptitle("Tolerance sensitivity (aligned metrics)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tolerance_sensitivity_aligned.png")
    plt.close(fig)

    # --- Fig 3: Orphan TAB before/after (v2) ---
    labels = ["target_orphan_tab", "pred_orphan_tab"]
    before = [681_228, 690_956]
    after = [0, 0]
    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(x - w / 2, before, w, label="v2 control (parser mismatch)", color="#CC6677")
    ax.bar(x + w / 2, after, w, label="v2 treatment (format-aware)", color="#228833")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Count (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e6)
    ax.legend()
    ax.set_title("Fairness fix: v2 orphan_tab inflation removed")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_orphan_tab_before_after_v2.png")
    plt.close(fig)

    # --- Fig 4: Guardrails baseline ---
    guard = ["token_accuracy", "tab_accuracy", "pitch_accuracy"]
    v1_g = [0.897840, 0.741549, 0.894097]
    v2_g = [0.786085, 0.644627, 0.878036]
    x = np.arange(len(guard))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - w / 2, v1_g, w, label="v1", color="#4477AA")
    ax.bar(x + w / 2, v2_g, w, label="v2", color="#CC6677")
    ax.set_xticks(x)
    ax.set_xticklabels(guard)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Guardrails: baseline token / tab / pitch accuracy")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_guardrails_baseline.png")
    plt.close(fig)

    print("Wrote:", *[str(OUT_DIR / f) for f in os.listdir(OUT_DIR)], sep="\n  ")


if __name__ == "__main__":
    main()
