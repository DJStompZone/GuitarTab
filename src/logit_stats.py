"""
Logit statistics collection for Experiment 5: Model Confidence Analysis.

Collects per-TAB-step metrics at each constrained decoding step to quantify
whether the model genuinely uses time/pitch context (rather than merely
benefiting from the mechanical elimination of invalid tokens).

Key metrics per TAB step
------------------------
prob_valid_mass          : total softmax probability assigned to valid TABs
                           (out of the full vocabulary). Compare with chance
                           = |valid_tabs| / vocab_size.
entropy_within_valid     : Shannon entropy of the renormalized distribution
                           over valid TABs [nats]. Lower = more confident.
max_entropy_within_valid : log(num_valid_tabs) – upper bound if model were
                           uniform within valid choices.
kl_from_uniform          : KL(P_renorm || Uniform_over_valid) [nats].
                           > 0 means model has preferences beyond random.
margin_within_valid      : top-1 minus top-2 logit gap within valid TABs.
free_choice_is_valid     : True if argmax(pre-mask logits) is a valid TAB.
free_choice_matches_constrained
                         : True if free argmax == constrained argmax.
                           When False, the constraint actively changes the
                           model's decision.
"""

import math
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core per-step computation
# ---------------------------------------------------------------------------

def compute_tab_step_stats(
    scores: torch.Tensor,      # [vocab_size], pre-mask logits (on device)
    valid_tab_ids: List[int],  # token IDs of valid (string, fret) for this pitch
    pitch: int,
) -> Dict[str, Any]:
    """
    Compute statistics for one TAB decoding step from pre-mask logits.

    Returns an empty dict if valid_tab_ids is empty.
    """
    num_valid = len(valid_tab_ids)
    if num_valid == 0:
        return {}

    device = scores.device
    valid_ids_t = torch.tensor(valid_tab_ids, dtype=torch.long, device=device)
    valid_logits = scores[valid_ids_t]  # [num_valid]

    # 1. Probability mass on valid TABs (out of full vocab softmax)
    full_probs = F.softmax(scores.float(), dim=0)
    prob_valid_mass = float(full_probs[valid_ids_t].sum().item())

    # 2. Entropy of renormalized distribution over valid TABs [nats]
    valid_probs = F.softmax(valid_logits.float(), dim=0)  # [num_valid]
    # Clamp to avoid log(0)
    entropy_within = float(
        -(valid_probs * (valid_probs.clamp(min=1e-12).log())).sum().item()
    )
    max_entropy = math.log(num_valid) if num_valid > 1 else 0.0

    # 3. KL(P_renorm || Uniform_over_valid) [nats]
    if num_valid > 1:
        log_uniform = -math.log(num_valid)
        kl = float(
            (valid_probs * (valid_probs.clamp(min=1e-12).log() - log_uniform))
            .sum()
            .item()
        )
        kl = max(kl, 0.0)  # numerical safety
    else:
        kl = 0.0

    # 4. Margin: top-1 minus top-2 logit within valid TABs
    if num_valid > 1:
        sorted_logits, _ = valid_logits.float().sort(descending=True)
        margin = float((sorted_logits[0] - sorted_logits[1]).item())
    else:
        margin = 0.0

    # 5. Free-decoding analysis
    free_argmax_id = int(scores.argmax().item())
    free_choice_is_valid = free_argmax_id in valid_tab_ids

    constrained_local_idx = int(valid_logits.argmax().item())
    constrained_choice_id = valid_tab_ids[constrained_local_idx]
    free_choice_matches_constrained = free_argmax_id == constrained_choice_id

    return {
        "pitch": pitch,
        "num_valid_tabs": num_valid,
        "prob_valid_mass": prob_valid_mass,
        "entropy_within_valid": entropy_within,
        "max_entropy_within_valid": max_entropy,
        "kl_from_uniform": kl,
        "margin_within_valid": margin,
        "free_choice_is_valid": free_choice_is_valid,
        "free_choice_matches_constrained": free_choice_matches_constrained,
        "constrained_choice_id": constrained_choice_id,
    }


# ---------------------------------------------------------------------------
# Accumulator used by BatchTablatureLogitsProcessor
# ---------------------------------------------------------------------------

class LogitStatsAccumulator:
    """Thread-unsafe but simple accumulator for per-step stats dicts."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def push(self, stats: Dict[str, Any], sample_idx: int, step_idx: int) -> None:
        if stats:
            rec = dict(stats)
            rec["sample_idx"] = sample_idx
            rec["step_idx"] = step_idx
            self.records.append(rec)

    def extend_with_offset(
        self,
        other_records: List[Dict[str, Any]],
        sample_offset: int,
    ) -> None:
        """Merge records from a per-batch processor, adding a global sample offset."""
        for rec in other_records:
            new_rec = dict(rec)
            new_rec["sample_idx"] = rec.get("sample_idx", 0) + sample_offset
            self.records.append(new_rec)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_logit_stats(
    records: List[Dict[str, Any]],
    vocab_size: int = 0,
) -> Dict[str, Any]:
    """
    Compute summary statistics over all collected TAB step records.

    Parameters
    ----------
    records   : list of dicts from LogitStatsAccumulator.records
    vocab_size: if > 0, computes chance-level prob_valid_mass

    Returns a nested dict with global means/stds and per-ambiguity breakdown.
    """
    if not records:
        return {}

    scalar_keys = [
        "prob_valid_mass",
        "entropy_within_valid",
        "max_entropy_within_valid",
        "kl_from_uniform",
        "margin_within_valid",
        "num_valid_tabs",
        "free_choice_is_valid",
        "free_choice_matches_constrained",
    ]

    arr: Dict[str, np.ndarray] = {}
    for k in scalar_keys:
        vals = [r[k] for r in records if k in r]
        if vals:
            arr[k] = np.array(vals, dtype=float)

    summary: Dict[str, Any] = {"total_tab_steps": len(records)}

    for k, vals in arr.items():
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std())
        summary[f"{k}_median"] = float(np.median(vals))

    # Normalized entropy: entropy / max_entropy  (1=uniform, 0=fully confident)
    if "entropy_within_valid" in arr and "max_entropy_within_valid" in arr:
        valid_mask = arr["max_entropy_within_valid"] > 0
        if valid_mask.any():
            norm = arr["entropy_within_valid"][valid_mask] / arr["max_entropy_within_valid"][valid_mask]
            summary["normalized_entropy_mean"] = float(norm.mean())
            summary["normalized_entropy_std"] = float(norm.std())
            summary["normalized_entropy_median"] = float(np.median(norm))

    # Chance-level probability mass: |valid_tabs| / vocab_size
    if vocab_size > 0 and "num_valid_tabs" in arr:
        chance = arr["num_valid_tabs"] / vocab_size
        summary["chance_prob_valid_mass_mean"] = float(chance.mean())

    # Ambiguity-stratified breakdown
    if "num_valid_tabs" in arr:
        ambi = arr["num_valid_tabs"]
        buckets = [
            (1, 1, "ambi_1"),
            (2, 2, "ambi_2"),
            (3, 4, "ambi_3_4"),
            (5, 999, "ambi_5plus"),
        ]
        strat: Dict[str, Any] = {}
        for lo, hi, label in buckets:
            mask = (ambi >= lo) & (ambi <= hi)
            if not mask.any():
                continue
            bucket: Dict[str, Any] = {"count": int(mask.sum())}
            for k in [
                "entropy_within_valid",
                "kl_from_uniform",
                "margin_within_valid",
                "prob_valid_mass",
                "free_choice_is_valid",
                "free_choice_matches_constrained",
            ]:
                if k in arr:
                    bucket[f"{k}_mean"] = float(arr[k][mask].mean())
                    bucket[f"{k}_std"] = float(arr[k][mask].std())
            strat[label] = bucket
        summary["by_ambiguity"] = strat

    return summary


def compute_structural_step_stats(
    scores: torch.Tensor,      # [vocab_size], pre-mask logits (on device)
    correct_token_id: int,
    token_type: str,           # "NOTE_ON" | "NOTE_OFF" | "TIME_SHIFT"
) -> Dict[str, Any]:
    """
    Compute statistics for one structural (FIXED_TOKEN) decoding step.

    Metrics
    -------
    prob_on_correct      : softmax probability assigned to the correct token
    margin_over_2nd      : correct logit minus the next-best logit (negative if correct is not top-1)
    free_argmax_is_correct: True if argmax(scores) == correct_token_id
    correct_rank         : 0-indexed rank of correct token in descending logit order
    entropy_full         : Shannon entropy over the full vocabulary [nats]
    """
    probs = F.softmax(scores.float(), dim=0)
    prob_on_correct = float(probs[correct_token_id].item())

    sorted_logits, sorted_ids = scores.float().sort(descending=True)

    matches = (sorted_ids == correct_token_id).nonzero(as_tuple=False)
    correct_rank = int(matches[0, 0].item()) if len(matches) > 0 else -1

    correct_logit = float(scores[correct_token_id].float().item())
    if len(sorted_logits) > 1:
        if correct_rank == 0:
            margin_over_2nd = float((sorted_logits[0] - sorted_logits[1]).item())
        else:
            margin_over_2nd = correct_logit - float(sorted_logits[0].item())
    else:
        margin_over_2nd = 0.0

    free_argmax_is_correct = correct_rank == 0

    entropy_full = float(
        -(probs * probs.clamp(min=1e-12).log()).sum().item()
    )

    return {
        "token_type": token_type,
        "correct_token_id": correct_token_id,
        "prob_on_correct": prob_on_correct,
        "margin_over_2nd": margin_over_2nd,
        "free_argmax_is_correct": free_argmax_is_correct,
        "correct_rank": correct_rank,
        "entropy_full": entropy_full,
    }


def aggregate_structural_stats(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute stratified summary statistics over structural step records.

    Returns a nested dict with overall and per-token-type breakdowns.
    """
    if not records:
        return {}

    scalar_keys = [
        "prob_on_correct",
        "margin_over_2nd",
        "free_argmax_is_correct",
        "correct_rank",
        "entropy_full",
    ]

    def _summarize(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        s: Dict[str, Any] = {"count": len(recs)}
        for k in scalar_keys:
            vals_raw = [r[k] for r in recs if k in r]
            if not vals_raw:
                continue
            arr = np.array(vals_raw, dtype=float)
            s[f"{k}_mean"] = float(arr.mean())
            s[f"{k}_std"] = float(arr.std())
            s[f"{k}_median"] = float(np.median(arr))
        return s

    summary: Dict[str, Any] = {"total_structural_steps": len(records)}
    summary["overall"] = _summarize(records)

    by_type: Dict[str, Any] = {}
    for tt in ("NOTE_ON", "NOTE_OFF", "TIME_SHIFT"):
        subset = [r for r in records if r.get("token_type") == tt]
        if subset:
            by_type[tt] = _summarize(subset)
    summary["by_token_type"] = by_type

    return summary


def print_structural_stats_summary(summary: Dict[str, Any]) -> None:
    """Print structural token logit stats in a readable format."""
    if not summary:
        print("[logit_stats] No structural records to summarize.")
        return

    n = summary.get("total_structural_steps", 0)
    print(f"\n{'='*70}")
    print(f"Structural Token Logit Stats Summary  (n = {n:,} steps)")
    print(f"{'='*70}")

    rows = [
        ("prob_on_correct",        "Prob on correct token",      "%"),
        ("margin_over_2nd",        "Margin over 2nd best",       "logit"),
        ("free_argmax_is_correct", "Free argmax is correct",     "%"),
        ("correct_rank",           "Rank of correct token",      "rank"),
        ("entropy_full",           "Full-vocab entropy",         "nats"),
    ]

    sections = [("Overall", summary.get("overall", {}))]
    for tt in ("NOTE_ON", "NOTE_OFF", "TIME_SHIFT"):
        sec = summary.get("by_token_type", {}).get(tt, {})
        if sec:
            sections.append((f"  {tt}", sec))

    for section_name, section_data in sections:
        if not section_data:
            continue
        count = section_data.get("count", "?")
        print(f"\n  [{section_name}]  n={count:,}")
        for key, label, unit in rows:
            mean_k = f"{key}_mean"
            std_k = f"{key}_std"
            if mean_k not in section_data:
                continue
            mean_val = section_data[mean_k]
            std_val = section_data.get(std_k, 0.0)
            if "%" in unit:
                print(f"    {label:<42} {mean_val*100:6.2f}% ± {std_val*100:.2f}%")
            else:
                print(f"    {label:<42} {mean_val:8.4f} ± {std_val:.4f}  [{unit}]")

    print(f"{'='*70}\n")


def print_logit_stats_summary(summary: Dict[str, Any]) -> None:
    """Print aggregate stats in a readable format."""
    if not summary:
        print("[logit_stats] No records to summarize.")
        return

    n = summary.get("total_tab_steps", 0)
    print(f"\n{'='*70}")
    print(f"Logit Statistics Summary  (n = {n:,} TAB steps)")
    print(f"{'='*70}")

    rows = [
        ("prob_valid_mass",             "Prob mass on valid TABs",      "%"),
        ("entropy_within_valid",        "Entropy within valid TABs",    "nats"),
        ("max_entropy_within_valid",    "Max entropy (log|valid|)",     "nats"),
        ("normalized_entropy",          "Normalized entropy",           "(0=conf,1=unif)"),
        ("kl_from_uniform",             "KL from Uniform",              "nats"),
        ("margin_within_valid",         "Top-1/Top-2 margin",           "logit"),
        ("free_choice_is_valid",        "Free argmax is valid TAB",     "%"),
        ("free_choice_matches_constrained", "Free == Constrained choice", "%"),
    ]

    for key, label, unit in rows:
        mean_k = f"{key}_mean"
        std_k = f"{key}_std"
        if mean_k not in summary:
            continue
        mean_val = summary[mean_k]
        std_val = summary.get(std_k, 0.0)
        if "%" in unit:
            print(f"  {label:<40} {mean_val*100:6.2f}% ± {std_val*100:.2f}%")
        else:
            print(f"  {label:<40} {mean_val:8.4f} ± {std_val:.4f}  [{unit}]")

    if "by_ambiguity" in summary:
        print(f"\n{'─'*70}")
        print("  Ambiguity-stratified breakdown:")
        print(f"  {'Bucket':<12} {'Count':>7} {'Entropy':>10} {'KL':>10} {'Margin':>10} {'P_valid':>10} {'FreeValid':>10}")
        for label, bucket in sorted(summary["by_ambiguity"].items()):
            count = bucket.get("count", 0)
            ent = bucket.get("entropy_within_valid_mean", float("nan"))
            kl = bucket.get("kl_from_uniform_mean", float("nan"))
            margin = bucket.get("margin_within_valid_mean", float("nan"))
            pv = bucket.get("prob_valid_mass_mean", float("nan"))
            fv = bucket.get("free_choice_is_valid_mean", float("nan"))
            print(
                f"  {label:<12} {count:>7,} {ent:>10.4f} {kl:>10.4f} "
                f"{margin:>10.4f} {pv:>10.4f} {fv*100:>9.1f}%"
            )

    print(f"{'='*70}\n")
