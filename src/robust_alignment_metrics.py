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

from src.metrics import compute_pitch_from_tab_token, infer_tuning_from_input_output


@dataclass
class NoteEvent:
    time: int
    pitch: Optional[int]
    string: Optional[int]
    fret: Optional[int]


@dataclass
class ParsedSequence:
    notes: List[NoteEvent]
    issue_counts: Dict[str, int]
    token_counter_by_class: Dict[str, Counter]


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


@dataclass
class RobustMetricsResult:
    num_samples: int
    total_target_notes: int
    total_pred_notes: int
    matched_notes: int
    coverage: float
    precision: float
    f1: float
    tab_acc_aligned: float
    pitch_acc_aligned: float
    syntax_issues_target: Dict[str, int]
    syntax_issues_pred: Dict[str, int]
    token_class_metrics: Dict[str, Dict[str, float]]
    per_sample: List[SampleRobustDiagnostics]

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


def parse_sequence_to_timeline(tokens: List[str], max_time_shift: Optional[int] = None) -> ParsedSequence:
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
                issue_counts["orphan_tab"] += 1
                notes.append(NoteEvent(time=current_time, pitch=None, string=string, fret=fret))
            else:
                notes.append(NoteEvent(time=current_time, pitch=pending_pitch, string=string, fret=fret))
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


def compute_robust_alignment_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_vocab,
    input_ids: Optional[torch.Tensor] = None,
    input_vocab=None,
    timeline_tolerance: int = 10,
    max_time_shift: Optional[int] = None,
) -> RobustMetricsResult:
    batch_size = targets.shape[0]

    total_target_notes = 0
    total_pred_notes = 0
    matched_notes = 0
    tab_match_count = 0
    pitch_match_count = 0
    syntax_issues_target = Counter()
    syntax_issues_pred = Counter()
    per_sample: List[SampleRobustDiagnostics] = []

    target_class_counters: Dict[str, Counter] = defaultdict(Counter)
    pred_class_counters: Dict[str, Counter] = defaultdict(Counter)

    for b in range(batch_size):
        target_tokens = _decode_valid_tokens(targets[b], output_vocab)
        pred_tokens = _decode_valid_tokens(predictions[b], output_vocab)

        parsed_target = parse_sequence_to_timeline(target_tokens, max_time_shift=max_time_shift)
        parsed_pred = parse_sequence_to_timeline(pred_tokens, max_time_shift=max_time_shift)

        syntax_issues_target.update(parsed_target.issue_counts)
        syntax_issues_pred.update(parsed_pred.issue_counts)

        for cls_name, c in parsed_target.token_counter_by_class.items():
            target_class_counters[cls_name].update(c)
        for cls_name, c in parsed_pred.token_counter_by_class.items():
            pred_class_counters[cls_name].update(c)

        matches = _match_events_by_time(parsed_target.notes, parsed_pred.notes, timeline_tolerance)

        sample_total_target = len(parsed_target.notes)
        sample_total_pred = len(parsed_pred.notes)
        sample_matched = len(matches)

        sample_tab_correct = 0
        sample_pitch_correct = 0

        tuning_list = None
        if input_ids is not None and input_vocab is not None and b < input_ids.shape[0]:
            sample_tuning = infer_tuning_from_input_output(
                input_ids[b], targets[b], input_vocab, output_vocab
            )
            if sample_tuning is not None:
                tuning_list = [sample_tuning.get(s, -1) for s in range(1, 7)]

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

        total_target_notes += sample_total_target
        total_pred_notes += sample_total_pred
        matched_notes += sample_matched
        tab_match_count += sample_tab_correct
        pitch_match_count += sample_pitch_correct

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
            )
        )

    coverage = _safe_ratio(matched_notes, total_target_notes)
    precision = _safe_ratio(matched_notes, total_pred_notes)
    f1 = _safe_ratio(2 * coverage * precision, coverage + precision) if coverage + precision > 0 else 0.0

    all_classes = sorted(set(target_class_counters.keys()) | set(pred_class_counters.keys()))
    token_class_metrics = {}
    for cls_name in all_classes:
        token_class_metrics[cls_name] = _counter_prf(
            target_class_counters[cls_name],
            pred_class_counters[cls_name],
        )

    return RobustMetricsResult(
        num_samples=batch_size,
        total_target_notes=total_target_notes,
        total_pred_notes=total_pred_notes,
        matched_notes=matched_notes,
        coverage=coverage,
        precision=precision,
        f1=f1,
        tab_acc_aligned=_safe_ratio(tab_match_count, matched_notes),
        pitch_acc_aligned=_safe_ratio(pitch_match_count, matched_notes),
        syntax_issues_target=dict(syntax_issues_target),
        syntax_issues_pred=dict(syntax_issues_pred),
        token_class_metrics=token_class_metrics,
        per_sample=per_sample,
    )
