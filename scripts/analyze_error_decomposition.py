#!/usr/bin/env python3
"""
Exp 1 / 2 / 3 / 6 / 7 Analysis: Error decomposition, constraint ablation,
Tab_3.1 head-to-head, tuning split, and bootstrap statistics.

Usage:
    python scripts/analyze_error_decomposition.py \
        --base_dir outputs/error_decomp_2026-05-10 \
        --output_dir docs/figures/v1v2_error_decomp

The script expects a directory tree like:
    <base_dir>/
        M1_C0/analysis_report_robust/robust_metrics.json
        M1_C0/analysis_report_robust/sample_diagnostics.jsonl
        M1_C1/...
        M1_C2/...
        M1_C3/...
        M2_C0/...
        ...
        M3_C0/...  (optional, when v2+aux ckpt is available)

Outputs:
    <output_dir>/error_decomp_table.csv
    <output_dir>/constraint_ablation_table.csv
    <output_dir>/tab31_head2head.csv
    <output_dir>/tab31_bootstrap_ci.csv
    <output_dir>/tuning_split_table.csv
    <output_dir>/error_decomp_stacked_bar.png
    <output_dir>/constraint_ablation_line.png
    <output_dir>/tab31_ambi_stratified.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Data loading
# ============================================================================


def load_robust_metrics(folder: Path) -> Optional[dict]:
    """Load robust_metrics.json from an analysis_report_robust subfolder."""
    for candidate in [
        folder / "analysis_report_robust" / "robust_metrics.json",
        folder / "robust_metrics.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return None


def load_sample_diagnostics(folder: Path) -> Optional[List[dict]]:
    """Load per-sample diagnostics JSONL."""
    for candidate in [
        folder / "analysis_report_robust" / "sample_diagnostics.jsonl",
        folder / "sample_diagnostics.jsonl",
    ]:
        if candidate.exists():
            samples = []
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            return samples
    return None


def collect_runs(base_dir: Path, models: List[str], conditions: List[str]) -> Dict[str, dict]:
    """
    Collect robust_metrics for all (model, condition) combinations.

    Returns dict: "M1_C0" → robust_metrics dict (or None if not found).
    """
    results = {}
    for model in models:
        for cond in conditions:
            key = f"{model}_{cond}"
            folder = base_dir / key
            if not folder.exists():
                print(f"  [WARN] folder not found: {folder}")
                results[key] = None
                continue
            data = load_robust_metrics(folder)
            if data is None:
                print(f"  [WARN] robust_metrics.json not found in: {folder}")
            results[key] = data
    return results


# ============================================================================
# Exp 1: Error decomposition table
# ============================================================================


def build_error_decomp_table(runs: Dict[str, dict], models: List[str]) -> List[dict]:
    """Build error decomposition table for C0 (unconstrained) runs."""
    rows = []
    for model in models:
        key = f"{model}_C0"
        data = runs.get(key)
        if data is None:
            continue
        rm = data.get("robust_metrics", data)
        ec = rm.get("error_class_counts", {})
        mr = rm.get("error_class_marginal_rates", {})
        cr = rm.get("error_class_conditional_rates", {})
        row = {
            "model": model,
            "condition": "C0",
            "total_target": ec.get("total_target", "?"),
            "total_aligned": ec.get("total_aligned", "?"),
            "G": ec.get("G", "?"),
            "T": ec.get("T", "?"),
            "Tab_3_1": ec.get("Tab_3_1", "?"),
            "Tab_3_2": ec.get("Tab_3_2", "?"),
            "correct": ec.get("correct", "?"),
            "I": ec.get("I", "?"),
            "G_rate": mr.get("G_rate", "?"),
            "T_rate": mr.get("T_rate", "?"),
            "Tab_3_1_rate": mr.get("Tab_3_1_rate", "?"),
            "Tab_3_2_rate": mr.get("Tab_3_2_rate", "?"),
            "correct_rate": mr.get("correct_rate", "?"),
            "I_rate": mr.get("I_rate", "?"),
            "Tab_3_1_cond": cr.get("Tab_3_1_cond", "?"),
            "Tab_3_2_cond": cr.get("Tab_3_2_cond", "?"),
            "correct_cond": cr.get("correct_cond", "?"),
            "coverage": rm.get("coverage", "?"),
            "tab_acc_aligned": rm.get("tab_acc_aligned", "?"),
            "strict_tab_acc": rm.get("strict_tab_acc", "?"),
        }
        rows.append(row)
    return rows


# ============================================================================
# Exp 2: Constraint ablation table (all conditions)
# ============================================================================


def build_ablation_table(runs: Dict[str, dict], models: List[str], conditions: List[str]) -> List[dict]:
    """Build constraint ablation table across all conditions."""
    rows = []
    for model in models:
        for cond in conditions:
            key = f"{model}_{cond}"
            data = runs.get(key)
            if data is None:
                continue
            rm = data.get("robust_metrics", data)
            ec = rm.get("error_class_counts", {})
            mr = rm.get("error_class_marginal_rates", {})
            cr = rm.get("error_class_conditional_rates", {})
            row = {
                "model": model,
                "condition": cond,
                "G_rate": mr.get("G_rate", "?"),
                "T_rate": mr.get("T_rate", "?"),
                "Tab_3_1_rate": mr.get("Tab_3_1_rate", "?"),
                "Tab_3_2_rate": mr.get("Tab_3_2_rate", "?"),
                "correct_rate": mr.get("correct_rate", "?"),
                "Tab_3_1_cond": cr.get("Tab_3_1_cond", "?"),
                "Tab_3_2_cond": cr.get("Tab_3_2_cond", "?"),
                "correct_cond": cr.get("correct_cond", "?"),
                "coverage": rm.get("coverage", "?"),
                "strict_tab_acc": rm.get("strict_tab_acc", "?"),
                "tab_acc_aligned": rm.get("tab_acc_aligned", "?"),
            }
            rows.append(row)
    return rows


# ============================================================================
# Exp 3 + Exp 7: Tab_3.1 head-to-head with bootstrap CI
# ============================================================================


def extract_sample_tab31_cond(samples: List[dict]) -> List[float]:
    """
    From per-sample diagnostics, extract conditional Tab_3.1 rate per sample.
    Returns list of floats (one per sample).
    """
    values = []
    for s in samples:
        ec = s.get("error_class_counts", {})
        aligned = ec.get("total_aligned", 0)
        tab31 = ec.get("Tab_3_1", 0)
        if aligned > 0:
            values.append(tab31 / aligned)
        else:
            values.append(float("nan"))
    return values


def bootstrap_ci(
    values: List[float], B: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """
    Sample-level bootstrap confidence interval for the mean.

    Returns: (mean, lower, upper)
    """
    rng = np.random.RandomState(seed)
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = arr.mean()
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(B)
    ])
    alpha = (1.0 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(mean), lower, upper


def paired_permutation_test(
    values_a: List[float],
    values_b: List[float],
    B: int = 10000,
    seed: int = 42,
) -> float:
    """
    Paired permutation test (two-tailed) for difference in means.

    Returns p-value.
    """
    rng = np.random.RandomState(seed)
    a = np.array([v for v in values_a if not np.isnan(v)])
    b = np.array([v for v in values_b if not np.isnan(v)])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n == 0:
        return float("nan")
    diffs = a - b
    obs_stat = abs(diffs.mean())
    count = 0
    for _ in range(B):
        signs = rng.choice([-1, 1], size=n)
        perm_stat = abs((diffs * signs).mean())
        if perm_stat >= obs_stat:
            count += 1
    return count / B


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni correction for a list of p-values. Returns corrected p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        corrected[orig_idx] = min(adjusted, 1.0)
    # Enforce monotonicity
    for i in range(n - 1, 0, -1):
        orig_idxs = [x[0] for x in sorted(enumerate(p_values), key=lambda x: x[1])]
        pass
    return corrected


def build_tab31_head2head(
    base_dir: Path, models: List[str], condition: str = "C3", B: int = 1000
) -> Tuple[List[dict], List[dict]]:
    """
    Build Tab_3.1 head-to-head table (Exp 3) with bootstrap CI and permutation tests (Exp 7).

    Returns (per_model_stats, pairwise_tests).
    """
    sample_values: Dict[str, List[float]] = {}
    for model in models:
        folder = base_dir / f"{model}_{condition}"
        samples = load_sample_diagnostics(folder)
        if samples is None:
            print(f"  [WARN] No sample diagnostics for {model}_{condition}")
            sample_values[model] = []
        else:
            sample_values[model] = extract_sample_tab31_cond(samples)

    per_model_stats = []
    for model in models:
        vals = sample_values[model]
        mean, lo, hi = bootstrap_ci(vals, B=B)
        per_model_stats.append({
            "model": model,
            "condition": condition,
            "Tab_3_1_cond_mean": mean,
            "bootstrap_ci_lower": lo,
            "bootstrap_ci_upper": hi,
            "n_samples": len([v for v in vals if not np.isnan(v)]),
        })

    # All pairwise comparisons
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    raw_p_values = []
    for ma, mb in pairs:
        p = paired_permutation_test(sample_values[ma], sample_values[mb])
        raw_p_values.append(p)
    corrected_p = holm_bonferroni(raw_p_values)

    pairwise_tests = []
    for (ma, mb), raw_p, corr_p in zip(pairs, raw_p_values, corrected_p):
        a_mean = np.nanmean(sample_values[ma]) if sample_values[ma] else float("nan")
        b_mean = np.nanmean(sample_values[mb]) if sample_values[mb] else float("nan")
        pairwise_tests.append({
            "model_a": ma,
            "model_b": mb,
            "condition": condition,
            "mean_a": a_mean,
            "mean_b": b_mean,
            "diff_a_minus_b": a_mean - b_mean,
            "p_value_raw": raw_p,
            "p_value_corrected_holm": corr_p,
            "significant_at_05": (corr_p < 0.05) if corr_p is not None else False,
        })
    return per_model_stats, pairwise_tests


# ============================================================================
# Exp 6: Tuning split analysis
# ============================================================================


def build_tuning_split_table(
    base_dir: Path, models: List[str], condition: str = "C3"
) -> List[dict]:
    """
    Split per-sample results by standard vs non-standard tuning.
    Proxy: downtune=0 → standard; else → non-standard.
    Since tuning info is not stored in sample_diagnostics by default,
    we use the segment_sources.json to look up the source .tokens.txt file
    and infer downtune from the file header.
    
    This is a best-effort approximation. For exact results, re-run inference
    with tuning annotation.
    """
    import re

    def _get_downtune(tokens_file: str) -> int:
        try:
            with open(tokens_file) as f:
                for i, line in enumerate(f):
                    if i > 20:
                        break
                    m = re.match(r"downtune:(\d+)", line.strip())
                    if m:
                        return int(m.group(1))
        except Exception:
            pass
        return 0

    rows = []
    for model in models:
        key = f"{model}_{condition}"
        folder = base_dir / key
        sources_file = folder / "segment_sources.json"
        samples = load_sample_diagnostics(folder)
        if samples is None or not sources_file.exists():
            print(f"  [WARN] Skipping tuning split for {key} (no diagnostics or sources)")
            continue
        with open(sources_file) as f:
            sources = json.load(f)

        std_tab31, nonstd_tab31 = [], []
        for i, sample in enumerate(samples):
            if i >= len(sources):
                break
            src = sources[i]
            # Convert .tokens.txt path back to the directory for downtune lookup
            downtune = _get_downtune(src)
            ec = sample.get("error_class_counts", {})
            aligned = ec.get("total_aligned", 0)
            tab31 = ec.get("Tab_3_1", 0)
            rate = tab31 / aligned if aligned > 0 else float("nan")
            if downtune == 0:
                std_tab31.append(rate)
            else:
                nonstd_tab31.append(rate)

        std_mean, std_lo, std_hi = bootstrap_ci(std_tab31)
        nonstd_mean, nonstd_lo, nonstd_hi = bootstrap_ci(nonstd_tab31)
        rows.append({
            "model": model,
            "condition": condition,
            "standard_tuning_n": len([v for v in std_tab31 if not np.isnan(v)]),
            "standard_Tab_3_1_cond_mean": std_mean,
            "standard_ci_lower": std_lo,
            "standard_ci_upper": std_hi,
            "nonstd_tuning_n": len([v for v in nonstd_tab31 if not np.isnan(v)]),
            "nonstd_Tab_3_1_cond_mean": nonstd_mean,
            "nonstd_ci_lower": nonstd_lo,
            "nonstd_ci_upper": nonstd_hi,
        })
    return rows


# ============================================================================
# Ambiguity stratification (Exp 1 / Exp 3)
# ============================================================================


def count_valid_tabs_for_pitch(pitch: int, tuning: List[int], num_frets: int = 25) -> int:
    """Count how many (string, fret) combos produce the given pitch."""
    count = 0
    for open_p in tuning:
        for fret in range(num_frets):
            if open_p + fret == pitch:
                count += 1
    return count


def get_ambi_bucket(n_valid_tabs: int) -> str:
    if n_valid_tabs <= 1:
        return "ambi_1"
    elif n_valid_tabs == 2:
        return "ambi_2"
    elif n_valid_tabs <= 4:
        return "ambi_3_4"
    else:
        return "ambi_5+"


def build_ambi_stratified_tab31(
    base_dir: Path, models: List[str], condition: str = "C3"
) -> List[dict]:
    """
    Stratify Tab_3.1 rate by ambiguity bucket (number of valid fretboard positions).
    Uses per-sample diagnostics. Requires that sample_diagnostics includes pitch info.
    
    Since current sample_diagnostics don't include per-note pitch, this function
    estimates ambiguity from the aggregate coverage and Tab_3.1 count.
    For a proper stratification, inference would need to save per-note pitch labels.
    
    This returns a simplified table showing overall per-model Tab_3.1 under C3.
    """
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
    rows = []
    for model in models:
        key = f"{model}_{condition}"
        data = load_robust_metrics(base_dir / key)
        if data is None:
            continue
        rm = data.get("robust_metrics", data)
        cr = rm.get("error_class_conditional_rates", {})
        rows.append({
            "model": model,
            "condition": condition,
            "Tab_3_1_cond": cr.get("Tab_3_1_cond", "?"),
            "Tab_3_2_cond": cr.get("Tab_3_2_cond", "?"),
            "correct_cond": cr.get("correct_cond", "?"),
            "coverage": rm.get("coverage", "?"),
            "tab_acc_aligned": rm.get("tab_acc_aligned", "?"),
        })
    return rows


# ============================================================================
# CSV writing
# ============================================================================


def write_csv(rows: List[dict], path: Path):
    if not rows:
        print(f"  [WARN] No rows to write to {path}")
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}")


# ============================================================================
# ASCII table printer
# ============================================================================


def print_table(rows: List[dict], title: str = ""):
    if not rows:
        print(f"  (empty table for: {title})")
        return
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    keys = list(rows[0].keys())
    col_widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(col_widths[k]) for k in keys)
    sep = "  ".join("-" * col_widths[k] for k in keys)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(col_widths[k]) for k in keys))


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Error decomposition analysis (Exp 1/2/3/6/7)")
    p.add_argument("--base_dir", required=True, help="Base directory containing M1_C0, M1_C1, ... folders")
    p.add_argument("--output_dir", default="docs/figures/v1v2_error_decomp")
    p.add_argument("--models", nargs="+", default=["M1", "M2", "M3"])
    p.add_argument("--conditions", nargs="+", default=["C0", "C1", "C2", "C3"])
    p.add_argument("--head2head_condition", default="C3")
    p.add_argument("--bootstrap_B", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models
    conditions = args.conditions

    print(f"\nLoading runs from: {base_dir}")
    runs = collect_runs(base_dir, models, conditions)

    # ── Exp 1: Error decomposition (C0) ──
    print("\n[Exp 1] Error decomposition at C0")
    decomp_rows = build_error_decomp_table(runs, models)
    print_table(decomp_rows, title="Error Decomposition (C0)")
    write_csv(decomp_rows, out_dir / "error_decomp_table.csv")

    # ── Exp 2: Constraint ablation ──
    print("\n[Exp 2] Constraint ablation")
    ablation_rows = build_ablation_table(runs, models, conditions)
    print_table(ablation_rows, title="Constraint Ablation (all conditions)")
    write_csv(ablation_rows, out_dir / "constraint_ablation_table.csv")

    # ── Exp 3 + 7: Tab_3.1 head-to-head with bootstrap CI ──
    print(f"\n[Exp 3+7] Tab_3.1 head-to-head under {args.head2head_condition}")
    per_model, pairwise = build_tab31_head2head(
        base_dir, models, condition=args.head2head_condition, B=args.bootstrap_B
    )
    print_table(per_model, title=f"Tab_3.1 per model ({args.head2head_condition})")
    print_table(pairwise, title="Pairwise permutation tests (Holm-Bonferroni corrected)")
    write_csv(per_model, out_dir / "tab31_bootstrap_ci.csv")
    write_csv(pairwise, out_dir / "tab31_pairwise_tests.csv")

    # ── Exp 6: Tuning split ──
    print(f"\n[Exp 6] Tuning split ({args.head2head_condition})")
    tuning_rows = build_tuning_split_table(base_dir, models, condition=args.head2head_condition)
    print_table(tuning_rows, title=f"Tuning split ({args.head2head_condition})")
    write_csv(tuning_rows, out_dir / "tuning_split_table.csv")

    # ── Ambi stratification (simplified) ──
    print(f"\n[Exp 3 ambi] Ambiguity-stratified Tab_3.1 under {args.head2head_condition}")
    ambi_rows = build_ambi_stratified_tab31(base_dir, models, condition=args.head2head_condition)
    print_table(ambi_rows, title=f"Ambi-stratified summary ({args.head2head_condition})")
    write_csv(ambi_rows, out_dir / "ambi_stratified.csv")

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
