"""
Robust alignment-aware metrics for tablature token sequences.

This module is designed for cases where predicted token streams drift from target
index positions (e.g. unconstrained decoding), so strict index-wise accuracy is
too brittle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.metrics import (
    compute_pitch_from_tab_token,
    infer_tuning_from_input_output,
    classify_aligned_note_error,
    GUITAR_TUNING,
)


@dataclass
class NoteEvent:
    time: int
    pitch: Optional[int]  # NOTE_ON pitch (v1) or None (v2)
    string: Optional[int]
    fret: Optional[int]
    is_valid: bool = True


@dataclass
class ErrorClassCounts:
    """
    Format-invariant error decomposition per the taxonomy in the paper.

    G       — grammar/structural: pred notes that could not be parsed correctly
              (is_valid=False orphan tabs in v1; missing_tab / unclosed_note counts).
    T       — time: target notes with no matching pred note within tolerance.
    Tab_3_2 — aligned notes where pitch is wrong (= P in the plan).
    Tab_3_1 — aligned notes where pitch is correct but (string, fret) is wrong.
    correct — aligned notes where (string, fret) is exactly correct.
    I       — internal inconsistency (v1-only): pred NOTE_ON pitch ≠ TAB-implied pitch.

    All *_rate fields are marginal (denominator = total target notes).
    Conditional rates (over aligned notes only) are computed by the caller.
    """

    G: int = 0
    T: int = 0
    Tab_3_2: int = 0
    Tab_3_1: int = 0
    correct: int = 0
    I: int = 0
    total_target: int = 0
    total_aligned: int = 0

    @property
    def P(self) -> int:
        """Pitch error (= Tab_3.2 for aligned pairs)."""
        return self.Tab_3_2

    def marginal_rates(self) -> Dict[str, float]:
        denom = max(self.total_target, 1)
        return {
            "G_rate": self.G / denom,
            "T_rate": self.T / denom,
            "Tab_3_2_rate": self.Tab_3_2 / denom,
            "Tab_3_1_rate": self.Tab_3_1 / denom,
            "correct_rate": self.correct / denom,
            "I_rate": self.I / denom,
        }

    def conditional_rates(self) -> Dict[str, float]:
        denom = max(self.total_aligned, 1)
        return {
            "Tab_3_2_cond": self.Tab_3_2 / denom,
            "Tab_3_1_cond": self.Tab_3_1 / denom,
            "correct_cond": self.correct / denom,
        }


@dataclass
class ParsedSequence:
    notes: List[NoteEvent]
    issue_counts: Dict[str, int]
    token_counter_by_class: Dict[str, Counter]
    num_tokens: int
    num_valid_notes: int


@dataclass
class SampleRobustDiagnostics:
    sample_idx: int
    num_target_notes: int
    num_pred_notes: int
    num_matched_notes: int
    coverage: float
    precision: float
    f1: float
    tab_acc_aligned: float
    pitch_acc_aligned: float
    target_issues: Dict[str, int]
    pred_issues: Dict[str, int]
    error_class_counts: Optional[Dict[str, int]] = None


@dataclass
class RobustMetricsResult:
    num_samples: int
    total_target_tokens: int
    total_pred_tokens: int
    total_target_notes: int
    total_pred_notes: int
    valid_target_notes: int
    valid_pred_notes: int
    matched_notes: int
    coverage: float
    precision: float
    f1: float
    tab_acc_aligned: float
    pitch_acc_aligned: float
    strict_tab_acc: float
    strict_pitch_acc: float
    strict_tab_score: float
    strict_pitch_score: float
    normalized_tab_acc: float
    normalized_pitch_acc: float
    valid_event_ratio_target: float
    valid_event_ratio_pred: float
    syntax_penalty_pred: float
    syntax_issues_target: Dict[str, int]
    syntax_issues_pred: Dict[str, int]
    syntax_issues_per_1k_tokens_target: Dict[str, float]
    syntax_issues_per_1k_tokens_pred: Dict[str, float]
    syntax_issues_per_1k_events_target: Dict[str, float]
    syntax_issues_per_1k_events_pred: Dict[str, float]
    token_class_metrics: Dict[str, Dict[str, float]]
    per_sample: List[SampleRobustDiagnostics]
    # --- Format-invariant error taxonomy (§1 of plan) ---
    error_class_counts: Optional[Dict[str, int]] = None
    error_class_marginal_rates: Optional[Dict[str, float]] = None
    error_class_conditional_rates: Optional[Dict[str, float]] = None
    # --- Positional (index-based) error taxonomy: T+Tab_3_1+Tab_3_2+correct == 1.0 ---
    positional_error_class_counts: Optional[Dict[str, int]] = None
    positional_error_class_rates: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["per_sample"] = [asdict(s) for s in self.per_sample]
        return data


def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _token_class(token: str) -> str:
    if token.startswith("NOTE_ON_"):
        return "NOTE_ON"
    if token.startswith("NOTE_OFF_"):
        return "NOTE_OFF"
    if token.startswith("TAB_"):
        return "TAB"
    if token.startswith("TIME_SHIFT_"):
        return "TIME_SHIFT"
    return "OTHER"


def _parse_note_pitch(token: str) -> Optional[int]:
    if not token.startswith(("NOTE_ON_", "NOTE_OFF_")):
        return None
    parts = token.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _parse_tab(token: str) -> Tuple[Optional[int], Optional[int]]:
    if not token.startswith("TAB_"):
        return None, None
    parts = token.split("_")
    if len(parts) != 3:
        return None, None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None, None


def _parse_time_shift(token: str) -> Optional[int]:
    if not token.startswith("TIME_SHIFT_"):
        return None
    parts = token.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _decode_valid_tokens(token_ids: torch.Tensor, vocab) -> List[str]:
    tokens = []
    for token_id in token_ids.tolist():
        if token_id in (vocab.pad_id, vocab.bos_id, vocab.eos_id):
            continue
        tokens.append(vocab.id_to_token[token_id])
    return tokens


def parse_sequence_to_timeline(
    tokens: List[str],
    output_format: str = "v1",
    max_time_shift: Optional[int] = None,
) -> ParsedSequence:
    issue_counts = defaultdict(int)
    token_counter_by_class: Dict[str, Counter] = defaultdict(Counter)
    notes: List[NoteEvent] = []

    current_time = 0
    pending_pitch: Optional[int] = None
    active_notes = Counter()
    state = "START"

    for token in tokens:
        token_cls = _token_class(token)
        token_counter_by_class[token_cls][token] += 1

        if token_cls == "NOTE_ON":
            pitch = _parse_note_pitch(token)
            if pitch is None:
                issue_counts["invalid_note_on"] += 1
                continue
            pending_pitch = pitch
            active_notes[pitch] += 1
            if state not in {"START", "EVENT", "TIME_SHIFT"}:
                issue_counts["invalid_transition"] += 1
            state = "EVENT"
            continue

        if token_cls == "TAB":
            string, fret = _parse_tab(token)
            if string is None or fret is None:
                issue_counts["invalid_tab"] += 1
                continue
            if pending_pitch is None:
                if output_format == "v2":
                    notes.append(
                        NoteEvent(
                            time=current_time,
                            pitch=None,
                            string=string,
                            fret=fret,
                            is_valid=True,
                        )
                    )
                else:
                    issue_counts["orphan_tab"] += 1
                    notes.append(
                        NoteEvent(
                            time=current_time,
                            pitch=None,
                            string=string,
                            fret=fret,
                            is_valid=False,
                        )
                    )
            else:
                notes.append(
                    NoteEvent(
                        time=current_time,
                        pitch=pending_pitch,
                        string=string,
                        fret=fret,
                        is_valid=True,
                    )
                )
                pending_pitch = None
            if state not in {"EVENT", "TIME_SHIFT", "START"}:
                issue_counts["invalid_transition"] += 1
            state = "EVENT"
            continue

        if token_cls == "NOTE_OFF":
            pitch = _parse_note_pitch(token)
            if pitch is None:
                issue_counts["invalid_note_off"] += 1
                continue
            if active_notes[pitch] <= 0:
                issue_counts["orphan_note_off"] += 1
            else:
                active_notes[pitch] -= 1
                if active_notes[pitch] == 0:
                    del active_notes[pitch]
            if state not in {"EVENT", "TIME_SHIFT", "START"}:
                issue_counts["invalid_transition"] += 1
            state = "EVENT"
            continue

        if token_cls == "TIME_SHIFT":
            shift = _parse_time_shift(token)
            if shift is None or shift < 0:
                issue_counts["illegal_time_shift"] += 1
                continue
            if max_time_shift is not None and shift > max_time_shift:
                issue_counts["illegal_time_shift"] += 1
            current_time += shift
            state = "TIME_SHIFT"
            continue

        issue_counts["unknown_token"] += 1

    if pending_pitch is not None:
        issue_counts["missing_tab"] += 1
    if active_notes:
        issue_counts["unclosed_note"] += sum(active_notes.values())

    return ParsedSequence(
        notes=notes,
        issue_counts=dict(issue_counts),
        token_counter_by_class={k: Counter(v) for k, v in token_counter_by_class.items()},
        num_tokens=len(tokens),
        num_valid_notes=sum(1 for note in notes if note.is_valid),
    )


def _match_events_by_time(
    target_notes: List[NoteEvent],
    pred_notes: List[NoteEvent],
    tolerance: int,
) -> List[Tuple[int, int]]:
    if not target_notes or not pred_notes:
        return []

    pred_sorted = sorted(enumerate(pred_notes), key=lambda x: x[1].time)
    target_sorted = sorted(enumerate(target_notes), key=lambda x: x[1].time)
    used_pred = set()
    matches: List[Tuple[int, int]] = []

    for target_idx, target_note in target_sorted:
        best_pred_idx = None
        best_dt = None
        for pred_idx, pred_note in pred_sorted:
            if pred_idx in used_pred:
                continue
            dt = abs(pred_note.time - target_note.time)
            if dt > tolerance:
                continue
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            used_pred.add(best_pred_idx)
            matches.append((target_idx, best_pred_idx))

    return matches


def _counter_prf(target_counter: Counter, pred_counter: Counter) -> Dict[str, float]:
    intersect = sum((target_counter & pred_counter).values())
    pred_total = sum(pred_counter.values())
    target_total = sum(target_counter.values())
    precision = _safe_ratio(intersect, pred_total)
    recall = _safe_ratio(intersect, target_total)
    f1 = _safe_ratio(2 * precision * recall, precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_total": float(target_total),
        "pred_total": float(pred_total),
        "matched": float(intersect),
    }


def _issue_rates_per_1k(issue_counts: Counter, denominator: int) -> Dict[str, float]:
    if denominator <= 0:
        return {k: 0.0 for k in issue_counts}
    return {k: (v * 1000.0) / denominator for k, v in issue_counts.items()}


def _syntax_penalty_from_rates(issue_rates_per_1k: Dict[str, float]) -> float:
    # Fixed contract for strict track penalty.
    issue_weights = {
        "invalid_note_on": 1.2,
        "invalid_tab": 1.2,
        "orphan_tab": 1.0,
        "invalid_note_off": 1.0,
        "orphan_note_off": 0.8,
        "illegal_time_shift": 1.2,
        "unknown_token": 0.5,
        "missing_tab": 0.8,
        "unclosed_note": 0.8,
        "invalid_transition": 0.7,
    }
    weighted_rate = 0.0
    for issue_name, rate in issue_rates_per_1k.items():
        weighted_rate += issue_weights.get(issue_name, 1.0) * rate
    return min(1.0, weighted_rate / 100.0)


def _compute_positional_error_taxonomy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_vocab,
    input_ids: Optional[torch.Tensor] = None,
    input_vocab=None,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Error taxonomy using positional (index) alignment — same masking as tab_accuracy.

    For every target TAB position, pred at the identical sequence index is compared:
      T       — pred token is not a TAB token (note absent/replaced)
      Tab_3_2 — pred TAB has different pitch
      Tab_3_1 — pred TAB has same pitch but different (string, fret)
      correct — pred TAB is identical

    Invariant: T + Tab_3_1 + Tab_3_2 + correct == total_target == 100 %
    """
    T = 0
    Tab_3_1 = 0
    Tab_3_2 = 0
    correct = 0
    total_target = 0

    batch_size = targets.shape[0]

    for b in range(batch_size):
        sample_target_ids = targets[b]
        sample_pred_ids = predictions[b]

        # Apply target non-pad mask to both (mirrors compute_tablature_accuracy).
        sample_mask = sample_target_ids != output_vocab.pad_id
        sample_target_valid = sample_target_ids[sample_mask]
        sample_pred_valid = sample_pred_ids[sample_mask]

        sample_target_tokens = [output_vocab.id_to_token[idx.item()] for idx in sample_target_valid]
        sample_pred_tokens = [output_vocab.id_to_token[idx.item()] for idx in sample_pred_valid]

        tab_indices = [i for i, t in enumerate(sample_target_tokens) if t.startswith("TAB_")]
        if not tab_indices:
            continue

        tuning_list = None
        if input_ids is not None and input_vocab is not None and b < input_ids.shape[0]:
            sample_tuning = infer_tuning_from_input_output(
                input_ids[b], sample_target_ids, input_vocab, output_vocab
            )
            if sample_tuning is not None:
                tuning_list = [sample_tuning.get(s, GUITAR_TUNING[s - 1]) for s in range(1, 7)]

        for tab_pos in tab_indices:
            total_target += 1
            target_tab = sample_target_tokens[tab_pos]
            pred_tab = sample_pred_tokens[tab_pos]

            if not pred_tab.startswith("TAB_"):
                T += 1
                continue

            try:
                _, ts, tf = target_tab.split("_")
                target_string, target_fret = int(ts), int(tf)
                _, ps, pf = pred_tab.split("_")
                pred_string, pred_fret = int(ps), int(pf)
            except (ValueError, AttributeError):
                Tab_3_2 += 1
                continue

            ec_label = classify_aligned_note_error(
                target_string, target_fret,
                pred_string, pred_fret,
                tuning=tuning_list,
            )
            if ec_label == "correct":
                correct += 1
            elif ec_label == "Tab_3.1":
                Tab_3_1 += 1
            else:
                Tab_3_2 += 1

    counts: Dict[str, int] = {
        "T": T,
        "Tab_3_1": Tab_3_1,
        "Tab_3_2": Tab_3_2,
        "correct": correct,
        "total_target": total_target,
    }
    denom = max(total_target, 1)
    rates: Dict[str, float] = {
        "T_rate": T / denom,
        "Tab_3_2_rate": Tab_3_2 / denom,
        "Tab_3_1_rate": Tab_3_1 / denom,
        "correct_rate": correct / denom,
    }
    return counts, rates


def compute_robust_alignment_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_vocab,
    input_ids: Optional[torch.Tensor] = None,
    input_vocab=None,
    timeline_tolerance: int = 10,
    output_format: str = "v1",
    max_time_shift: Optional[int] = None,
) -> RobustMetricsResult:
    batch_size = targets.shape[0]

    total_target_notes = 0
    total_pred_notes = 0
    total_target_tokens = 0
    total_pred_tokens = 0
    valid_target_notes = 0
    valid_pred_notes = 0
    matched_notes = 0
    tab_match_count = 0
    pitch_match_count = 0
    normalized_tab_match_count = 0
    normalized_pitch_match_count = 0
    normalized_matched_notes = 0
    syntax_issues_target = Counter()
    syntax_issues_pred = Counter()
    per_sample: List[SampleRobustDiagnostics] = []

    target_class_counters: Dict[str, Counter] = defaultdict(Counter)
    pred_class_counters: Dict[str, Counter] = defaultdict(Counter)

    # Error taxonomy accumulators
    agg_ec = ErrorClassCounts()

    for b in range(batch_size):
        target_tokens = _decode_valid_tokens(targets[b], output_vocab)
        pred_tokens = _decode_valid_tokens(predictions[b], output_vocab)

        parsed_target = parse_sequence_to_timeline(
            target_tokens,
            output_format=output_format,
            max_time_shift=max_time_shift,
        )
        parsed_pred = parse_sequence_to_timeline(
            pred_tokens,
            output_format=output_format,
            max_time_shift=max_time_shift,
        )

        syntax_issues_target.update(parsed_target.issue_counts)
        syntax_issues_pred.update(parsed_pred.issue_counts)

        for cls_name, c in parsed_target.token_counter_by_class.items():
            target_class_counters[cls_name].update(c)
        for cls_name, c in parsed_pred.token_counter_by_class.items():
            pred_class_counters[cls_name].update(c)

        matches = _match_events_by_time(parsed_target.notes, parsed_pred.notes, timeline_tolerance)

        sample_total_target = len(parsed_target.notes)
        sample_total_pred = len(parsed_pred.notes)
        sample_valid_target = parsed_target.num_valid_notes
        sample_valid_pred = parsed_pred.num_valid_notes
        sample_matched = len(matches)

        sample_tab_correct = 0
        sample_pitch_correct = 0
        sample_tab_correct_valid = 0
        sample_pitch_correct_valid = 0
        sample_matched_valid = 0

        tuning_list = None
        if input_ids is not None and input_vocab is not None and b < input_ids.shape[0]:
            sample_tuning = infer_tuning_from_input_output(
                input_ids[b], targets[b], input_vocab, output_vocab
            )
            if sample_tuning is not None:
                tuning_list = [sample_tuning.get(s, -1) for s in range(1, 7)]

        # Per-sample error taxonomy counters
        sample_ec = ErrorClassCounts()
        matched_target_indices: set = {ti for ti, _ in matches}

        # G: pred notes that failed to parse (is_valid=False)
        sample_ec.G = sum(1 for n in parsed_pred.notes if not n.is_valid)
        # Also count missing_tab (NOTE_ON without TAB) as grammar errors
        sample_ec.G += parsed_pred.issue_counts.get("missing_tab", 0)

        # T: target notes not matched within tolerance
        sample_ec.T = sample_total_target - len(matches)

        sample_ec.total_target = sample_total_target
        sample_ec.total_aligned = len(matches)

        # I: v1-only internal inconsistency (NOTE_ON pitch ≠ TAB-implied pitch) in pred
        if output_format == "v1" and tuning_list is not None:
            for note in parsed_pred.notes:
                if note.pitch is not None and note.string is not None and note.fret is not None:
                    tab_implied = tuning_list[note.string - 1] + note.fret
                    if note.pitch != tab_implied:
                        sample_ec.I += 1

        for target_idx, pred_idx in matches:
            target_note = parsed_target.notes[target_idx]
            pred_note = parsed_pred.notes[pred_idx]

            if (
                target_note.string is not None
                and target_note.fret is not None
                and pred_note.string is not None
                and pred_note.fret is not None
                and (target_note.string, target_note.fret) == (pred_note.string, pred_note.fret)
            ):
                sample_tab_correct += 1

            target_tab_token = (
                f"TAB_{target_note.string}_{target_note.fret}"
                if target_note.string is not None and target_note.fret is not None
                else ""
            )
            pred_tab_token = (
                f"TAB_{pred_note.string}_{pred_note.fret}"
                if pred_note.string is not None and pred_note.fret is not None
                else ""
            )
            target_pitch = compute_pitch_from_tab_token(target_tab_token, tuning=tuning_list)
            pred_pitch = compute_pitch_from_tab_token(pred_tab_token, tuning=tuning_list)
            if target_pitch != -1 and pred_pitch != -1 and target_pitch == pred_pitch:
                sample_pitch_correct += 1

            # Error class classification for this aligned pair
            ec_label = classify_aligned_note_error(
                target_note.string, target_note.fret,
                pred_note.string, pred_note.fret,
                tuning=tuning_list,
            )
            if ec_label == "correct":
                sample_ec.correct += 1
            elif ec_label == "Tab_3.1":
                sample_ec.Tab_3_1 += 1
            else:
                sample_ec.Tab_3_2 += 1

            if target_note.is_valid and pred_note.is_valid:
                sample_matched_valid += 1
                if (
                    target_note.string is not None
                    and target_note.fret is not None
                    and pred_note.string is not None
                    and pred_note.fret is not None
                    and (target_note.string, target_note.fret) == (pred_note.string, pred_note.fret)
                ):
                    sample_tab_correct_valid += 1
                if target_pitch != -1 and pred_pitch != -1 and target_pitch == pred_pitch:
                    sample_pitch_correct_valid += 1

        total_target_tokens += parsed_target.num_tokens
        total_pred_tokens += parsed_pred.num_tokens
        total_target_notes += sample_total_target
        total_pred_notes += sample_total_pred
        valid_target_notes += sample_valid_target
        valid_pred_notes += sample_valid_pred
        matched_notes += sample_matched
        tab_match_count += sample_tab_correct
        pitch_match_count += sample_pitch_correct
        # Reuse counters for normalized aggregate.
        normalized_tab_match_count += sample_tab_correct_valid
        normalized_pitch_match_count += sample_pitch_correct_valid
        normalized_matched_notes += sample_matched_valid

        # Accumulate error taxonomy
        agg_ec.G += sample_ec.G
        agg_ec.T += sample_ec.T
        agg_ec.Tab_3_2 += sample_ec.Tab_3_2
        agg_ec.Tab_3_1 += sample_ec.Tab_3_1
        agg_ec.correct += sample_ec.correct
        agg_ec.I += sample_ec.I
        agg_ec.total_target += sample_ec.total_target
        agg_ec.total_aligned += sample_ec.total_aligned

        sample_cov = _safe_ratio(sample_matched, sample_total_target)
        sample_prec = _safe_ratio(sample_matched, sample_total_pred)
        sample_f1 = _safe_ratio(2 * sample_cov * sample_prec, sample_cov + sample_prec) if (sample_cov + sample_prec) > 0 else 0.0

        per_sample.append(
            SampleRobustDiagnostics(
                sample_idx=b,
                num_target_notes=sample_total_target,
                num_pred_notes=sample_total_pred,
                num_matched_notes=sample_matched,
                coverage=sample_cov,
                precision=sample_prec,
                f1=sample_f1,
                tab_acc_aligned=_safe_ratio(sample_tab_correct, sample_matched),
                pitch_acc_aligned=_safe_ratio(sample_pitch_correct, sample_matched),
                target_issues=parsed_target.issue_counts,
                pred_issues=parsed_pred.issue_counts,
                error_class_counts={
                    "G": sample_ec.G,
                    "T": sample_ec.T,
                    "Tab_3_2": sample_ec.Tab_3_2,
                    "Tab_3_1": sample_ec.Tab_3_1,
                    "correct": sample_ec.correct,
                    "I": sample_ec.I,
                    "total_target": sample_ec.total_target,
                    "total_aligned": sample_ec.total_aligned,
                },
            )
        )

    coverage = _safe_ratio(matched_notes, total_target_notes)
    precision = _safe_ratio(matched_notes, total_pred_notes)
    f1 = _safe_ratio(2 * coverage * precision, coverage + precision) if coverage + precision > 0 else 0.0
    strict_tab_acc = _safe_ratio(tab_match_count, total_target_notes)
    strict_pitch_acc = _safe_ratio(pitch_match_count, total_target_notes)
    normalized_tab_acc = _safe_ratio(normalized_tab_match_count, normalized_matched_notes)
    normalized_pitch_acc = _safe_ratio(normalized_pitch_match_count, normalized_matched_notes)

    all_classes = sorted(set(target_class_counters.keys()) | set(pred_class_counters.keys()))
    token_class_metrics = {}
    for cls_name in all_classes:
        token_class_metrics[cls_name] = _counter_prf(
            target_class_counters[cls_name],
            pred_class_counters[cls_name],
        )

    issue_rates_token_target = _issue_rates_per_1k(syntax_issues_target, total_target_tokens)
    issue_rates_token_pred = _issue_rates_per_1k(syntax_issues_pred, total_pred_tokens)
    issue_rates_event_target = _issue_rates_per_1k(syntax_issues_target, total_target_notes)
    issue_rates_event_pred = _issue_rates_per_1k(syntax_issues_pred, total_pred_notes)
    syntax_penalty_pred = _syntax_penalty_from_rates(issue_rates_token_pred)
    strict_tab_score = strict_tab_acc * (1.0 - syntax_penalty_pred)
    strict_pitch_score = strict_pitch_acc * (1.0 - syntax_penalty_pred)

    return RobustMetricsResult(
        num_samples=batch_size,
        total_target_tokens=total_target_tokens,
        total_pred_tokens=total_pred_tokens,
        total_target_notes=total_target_notes,
        total_pred_notes=total_pred_notes,
        valid_target_notes=valid_target_notes,
        valid_pred_notes=valid_pred_notes,
        matched_notes=matched_notes,
        coverage=coverage,
        precision=precision,
        f1=f1,
        tab_acc_aligned=_safe_ratio(tab_match_count, matched_notes),
        pitch_acc_aligned=_safe_ratio(pitch_match_count, matched_notes),
        strict_tab_acc=strict_tab_acc,
        strict_pitch_acc=strict_pitch_acc,
        strict_tab_score=strict_tab_score,
        strict_pitch_score=strict_pitch_score,
        normalized_tab_acc=normalized_tab_acc,
        normalized_pitch_acc=normalized_pitch_acc,
        valid_event_ratio_target=_safe_ratio(valid_target_notes, total_target_notes),
        valid_event_ratio_pred=_safe_ratio(valid_pred_notes, total_pred_notes),
        syntax_penalty_pred=syntax_penalty_pred,
        syntax_issues_target=dict(syntax_issues_target),
        syntax_issues_pred=dict(syntax_issues_pred),
        syntax_issues_per_1k_tokens_target=issue_rates_token_target,
        syntax_issues_per_1k_tokens_pred=issue_rates_token_pred,
        syntax_issues_per_1k_events_target=issue_rates_event_target,
        syntax_issues_per_1k_events_pred=issue_rates_event_pred,
        token_class_metrics=token_class_metrics,
        per_sample=per_sample,
        error_class_counts={
            "G": agg_ec.G,
            "T": agg_ec.T,
            "Tab_3_2": agg_ec.Tab_3_2,
            "Tab_3_1": agg_ec.Tab_3_1,
            "correct": agg_ec.correct,
            "I": agg_ec.I,
            "total_target": agg_ec.total_target,
            "total_aligned": agg_ec.total_aligned,
        },
        error_class_marginal_rates=agg_ec.marginal_rates(),
        error_class_conditional_rates=agg_ec.conditional_rates(),
        **dict(zip(
            ("positional_error_class_counts", "positional_error_class_rates"),
            _compute_positional_error_taxonomy(
                predictions=predictions,
                targets=targets,
                output_vocab=output_vocab,
                input_ids=input_ids,
                input_vocab=input_vocab,
            ),
        )),
    )
