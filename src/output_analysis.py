"""
Output format analysis for Fretting-Transformer v1 predictions.

Analyzes model outputs to detect structural issues:
1. Note-On/Tab/Note-Off mismatch
2. Correct time, wrong note ratio
3. Extra notes
4. Absent notes
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

from src.tab_dataset import Vocabulary
from src.post_processing_reverse import get_token_info, GUITAR_TUNING
from src.metrics import compute_levenshtein_distance

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Note:
    """Single note with its associated tokens."""
    pitch: int
    note_on_idx: int
    tab: Optional[Tuple[int, int]] = None  # (string, fret)
    tab_idx: Optional[int] = None
    note_off_idx: Optional[int] = None
    computed_pitch: Optional[int] = None  # Pitch computed from TAB


@dataclass
class TimeSegment:
    """Group of notes before a TIME_SHIFT."""
    notes: List[Note]
    time_shift_delta: Optional[int]
    time_shift_idx: Optional[int]
    start_idx: int
    end_idx: int


@dataclass
class TimelineAlignmentAnalysis:
    """Timeline alignment between prediction and target."""
    # Target-based (Recall)
    target_timeline: List[int]  # Absolute positions in ticks
    covered_target_cuts: List[int]  # Positions with matching pred
    missing_target_cuts: List[int]  # Positions without matching pred

    # Prediction-based (Precision)
    pred_timeline: List[int]
    correct_pred_cuts: List[int]  # Positions with matching target
    hallucinated_pred_cuts: List[int]  # Positions without matching target

    # Matched pairs with timing errors
    matched_pairs: List[Dict]  # {"target_pos": int, "pred_pos": int, "offset": int}

    # Metrics
    target_coverage_rate: float  # Recall
    pred_precision_rate: float   # Precision
    f1_score: float

    # Statistics
    mean_timing_offset: float
    std_timing_offset: float
    max_timing_offset: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "target_timeline": self.target_timeline,
            "pred_timeline": self.pred_timeline,
            "covered_target_cuts": self.covered_target_cuts,
            "missing_target_cuts": self.missing_target_cuts,
            "correct_pred_cuts": self.correct_pred_cuts,
            "hallucinated_pred_cuts": self.hallucinated_pred_cuts,
            "matched_pairs": self.matched_pairs,
            "metrics": {
                "target_coverage_rate": self.target_coverage_rate,
                "pred_precision_rate": self.pred_precision_rate,
                "f1_score": self.f1_score,
                "mean_timing_offset": self.mean_timing_offset,
                "std_timing_offset": self.std_timing_offset,
                "max_timing_offset": self.max_timing_offset,
            }
        }


@dataclass
class StructuralIssue:
    """A single structural violation."""
    issue_type: str
    sample_idx: int
    token_idx: int
    details: Dict[str, Any] = field(default_factory=dict)
    segment_idx: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "issue_type": self.issue_type,
            "sample_idx": self.sample_idx,
            "token_idx": self.token_idx,
            "segment_idx": self.segment_idx,
            "details": self.details,
        }


@dataclass
class SampleAnalysis:
    """Analysis result for a single sample."""
    sample_idx: int
    num_segments: int
    structural_issues: List[StructuralIssue]
    extra_notes: List[Dict]
    absent_notes: List[Dict]
    time_shift_mismatches: List[Dict]

    # Timeline alignment analysis
    timeline_alignment: Optional[TimelineAlignmentAnalysis] = None

    # Levenshtein metrics
    levenshtein_distance: float = 0.0
    levenshtein_similarity: float = 0.0

    # Summary counts
    issue_counts: Dict[str, int] = field(default_factory=dict)
    issue_counts_timeline: Dict[str, int] = field(default_factory=dict)

    # Position distribution for extra/absent notes
    extra_notes_position_dist: Dict[str, int] = field(default_factory=dict)
    absent_notes_position_dist: Dict[str, int] = field(default_factory=dict)

    def compute_counts(self):
        """Compute issue counts from issues list."""
        counts = defaultdict(int)
        for issue in self.structural_issues:
            counts[issue.issue_type] += 1
        counts["extra_note"] = len(self.extra_notes)
        counts["absent_note"] = len(self.absent_notes)
        counts["time_shift_mismatch"] = len(self.time_shift_mismatches)
        self.issue_counts = dict(counts)

    def compute_position_distribution(self):
        """Compute position distribution for extra/absent notes."""
        # Count extra notes by position
        extra_dist = defaultdict(int)
        for note in self.extra_notes:
            pos = note.get("position", "unknown")
            extra_dist[pos] += 1
        self.extra_notes_position_dist = dict(extra_dist)

        # Count absent notes by position
        absent_dist = defaultdict(int)
        for note in self.absent_notes:
            pos = note.get("position", "unknown")
            absent_dist[pos] += 1
        self.absent_notes_position_dist = dict(absent_dist)

@dataclass
class OutputAnalysisResult:
    """Complete analysis result."""
    # Summary
    total_samples: int
    total_segments: int
    samples_with_issues: int

    # Aggregate counts
    issue_counts: Dict[str, int]
    issue_counts_timeline: Dict[str, int]

    # Per-sample details
    per_sample: List[SampleAnalysis]


    # All issues (flattened)
    all_issues: List[Dict]

    # Aggregate timeline alignment metrics
    avg_timeline_coverage: float = 0.0  # Average recall across samples
    avg_timeline_precision: float = 0.0  # Average precision across samples
    avg_timeline_f1: float = 0.0
    avg_timing_offset: float = 0.0
    std_timing_offset: float = 0.0

    # Levenshtein metrics
    avg_levenshtein_distance: float = 0.0
    avg_levenshtein_similarity: float = 0.0

    # Position distribution for extra/absent notes
    extra_notes_position_dist: Dict[str, int] = field(default_factory=dict)
    absent_notes_position_dist: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_samples": self.total_samples,
                "total_segments": self.total_segments,
                "samples_with_issues": self.samples_with_issues,
                "issue_counts": self.issue_counts,
                "timeline_alignment": {
                    "avg_coverage": self.avg_timeline_coverage,
                    "avg_precision": self.avg_timeline_precision,
                    "avg_f1": self.avg_timeline_f1,
                    "avg_timing_offset": self.avg_timing_offset,
                    "std_timing_offset": self.std_timing_offset,
                },
                "levenshtein": {
                    "avg_distance": self.avg_levenshtein_distance,
                    "avg_similarity": self.avg_levenshtein_similarity,
                },
                "position_distribution": {
                    "extra_notes": self.extra_notes_position_dist,
                    "absent_notes": self.absent_notes_position_dist,
                }
            },
            "per_sample": [
                {
                    "sample_idx": s.sample_idx,
                    "num_segments": s.num_segments,
                    "num_issues": sum(s.issue_counts.values()),
                    "issue_counts": s.issue_counts,
                    "timeline_alignment": s.timeline_alignment.to_dict() if s.timeline_alignment else None,
                    "position_distribution": {
                        "extra_notes": s.extra_notes_position_dist,
                        "absent_notes": s.absent_notes_position_dist,
                    }
                }
                for s in self.per_sample
            ],
            "issues": self.all_issues,
        }


# ============================================================================
# Helper Functions
# ============================================================================

def compute_pitch_from_tab(string: int, fret: int) -> int:
    """Compute MIDI pitch from guitar string and fret."""
    # string is 1-indexed (1-6), GUITAR_TUNING is 0-indexed
    if 1 <= string <= len(GUITAR_TUNING):
        open_pitch = GUITAR_TUNING[string - 1]
        return open_pitch + fret
    return -1


def parse_tokens(token_ids: List[int], vocab: Vocabulary) -> List[Dict]:
    """Parse token IDs into structured token info."""
    return [get_token_info(tid, vocab) for tid in token_ids]


def build_absolute_timeline(tokens: List[Dict]) -> List[int]:
    """
    Build absolute timeline positions from TIME_SHIFT tokens.

    Args:
        tokens: List of parsed token dicts (from parse_tokens)

    Returns:
        List of absolute positions (in ticks) where TIME_SHIFT occurs
        Includes position 0 as the implicit first cut point
    """
    # Start position is 0 (implicit first cut point)
    timeline = [0]
    current_pos = 0

    for token in tokens:
        if token.get("type") == "TIME_SHIFT":
            delta = token.get("delta", 0)
            current_pos += delta
            timeline.append(current_pos)

    return timeline


def find_closest_cut(position: int, timeline: List[int]) -> Tuple[int, int]:
    """
    Find closest cut point in timeline to given position.

    Args:
        position: Target position in ticks
        timeline: List of timeline positions

    Returns:
        Tuple of (closest_position, absolute_offset)
    """
    if not timeline:
        return -1, float('inf')

    closest = min(timeline, key=lambda x: abs(x - position))
    offset = abs(closest - position)
    return closest, offset


def get_timeline_cutoff_index(pred_tokens: List[Dict], target_tokens: List[Dict], tolerance: int = 10) -> int:
    """
    Find the index in prediction tokens where the accumulated time exceeds the target duration.
    
    Args:
        pred_tokens: List of parsed prediction tokens.
        target_tokens: List of parsed target tokens.
        tolerance: Tolerance in ticks.
        
    Returns:
        The index of the first token that is considered "post-end".
        Returns len(pred_tokens) if the whole sequence is within timeline.
    """
    # 1. Get target duration
    target_duration = sum(t['delta'] for t in target_tokens if t.get('type') == 'TIME_SHIFT')
    
    # 2. Find cutoff in prediction
    current_time = 0
    
    # We iterate through tokens.
    # The time of a token is determined by the SUM of preceding TIME_SHIFTS.
    # Actually, in our format (TimeSegment), notes occur BEFORE the TimeShift that advances time.
    # So:
    # Segment 0: starts at time 0. Notes/Tabs here are at time 0.
    # TIME_SHIFT 100.
    # Segment 1: starts at time 100. Notes/Tabs here are at time 100.
    
    cutoff_idx = len(pred_tokens)
    
    for i, token in enumerate(pred_tokens):
        # Check if the current time bucket is beyond target duration
        if current_time > target_duration + tolerance:
            return i
            
        if token.get('type') == 'TIME_SHIFT':
            current_time += token['delta']
            
    # Check one last time after loop if we ended exactly at boundary or something
    # But loop covers it.
    
    return cutoff_idx


# ============================================================================
# Structural Analysis (Category 1)
# ============================================================================

def validate_structure_v2(
    token_ids: List[int],
    tokens: List[Dict],
    vocab: Vocabulary,
    sample_idx: int,
) -> Tuple[List[StructuralIssue], List[TimeSegment]]:
    """
    Validate v2 format structure (TAB, TIME_SHIFT only).

    v2 format has minimal constraints:
    - Should only contain TAB and TIME_SHIFT tokens (plus special tokens)
    - TAB tokens can appear in any order before TIME_SHIFT

    Returns:
        Tuple of (issues list, segments list)
    """
    issues: List[StructuralIssue] = []
    segments: List[TimeSegment] = []

    # Build segments by TIME_SHIFT
    current_segment_tabs: List[Note] = []  # Store TAB info as Note objects (without pitch info)
    segment_start_idx = 0

    for i, token in enumerate(tokens):
        token_type = token.get("type", "UNKNOWN")

        if token_type == "TAB":
            # Create a Note object with TAB info (pitch will be computed)
            string = token["string"]
            fret = token["fret"]
            computed_pitch = compute_pitch_from_tab(string, fret)

            note = Note(
                pitch=computed_pitch,  # v2 doesn't have NOTE_ON, use computed pitch
                note_on_idx=-1,  # No NOTE_ON in v2
                tab=(string, fret),
                tab_idx=i,
                computed_pitch=computed_pitch,
            )
            current_segment_tabs.append(note)

        elif token_type == "TIME_SHIFT":
            # End current segment
            if current_segment_tabs or segment_start_idx < i:
                segments.append(TimeSegment(
                    notes=current_segment_tabs.copy(),
                    time_shift_delta=token.get("delta"),
                    time_shift_idx=i,
                    start_idx=segment_start_idx,
                    end_idx=i,
                ))
            current_segment_tabs = []
            segment_start_idx = i + 1

        elif token_type in ("NOTE_ON", "NOTE_OFF"):
            # These shouldn't exist in v2 format
            issues.append(StructuralIssue(
                issue_type="unexpected_token_in_v2",
                sample_idx=sample_idx,
                token_idx=i,
                details={
                    "token_type": token_type,
                    "expected": "TAB or TIME_SHIFT only",
                }
            ))

    # Handle remaining TABs (no trailing TIME_SHIFT)
    if current_segment_tabs:
        segments.append(TimeSegment(
            notes=current_segment_tabs,
            time_shift_delta=None,
            time_shift_idx=None,
            start_idx=segment_start_idx,
            end_idx=len(tokens) - 1,
        ))

    return issues, segments


def validate_structure(
    token_ids: List[int],
    vocab: Vocabulary,
    sample_idx: int,
    output_format: str = "v1",
) -> Tuple[List[StructuralIssue], List[TimeSegment]]:
    """
    Validate the structural integrity of a token sequence.

    For v1:
    - Each NOTE_ON is immediately followed by TAB
    - Each NOTE_ON has a corresponding NOTE_OFF (anywhere after)
    - TAB pitch matches NOTE_ON pitch (accounting for inferred tuning)
    - No orphan NOTE_OFF or TAB tokens

    For v2:
    - Only TAB and TIME_SHIFT tokens should exist
    - No structural validation needed (TAB can appear freely)

    Args:
        token_ids: Token IDs to validate
        vocab: Vocabulary
        sample_idx: Sample index for error reporting
        output_format: "v1", "v2", or "v3"

    Returns:
        Tuple of (issues list, segments list)
    """
    issues: List[StructuralIssue] = []
    tokens = parse_tokens(token_ids, vocab)

    # For v2 format: only validate token types, no structural constraints
    if output_format == "v2":
        return validate_structure_v2(token_ids, tokens, vocab, sample_idx)

    # v1/v3 validation below
    # Track NOTE_ON tokens that need NOTE_OFF
    # Key: pitch, Value: list of (note_on_idx, tab_idx, tab_info)
    open_notes: Dict[int, List[Tuple[int, Optional[int], Optional[Dict]]]] = defaultdict(list)

    # Track which TAB tokens have been associated with NOTE_ON
    used_tab_indices = set()

    # Build segments
    segments: List[TimeSegment] = []
    current_segment_notes: List[Note] = []
    segment_start_idx = 0
    
    # Track potential pitch mismatches for tuning inference
    potential_pitch_mismatches = []
    valid_pitch_pairs = [] # List of (note_on_pitch, computed_pitch_standard)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_type = token.get("type", "UNKNOWN")

        if token_type == "NOTE_ON":
            pitch = token["pitch"]
            note = Note(pitch=pitch, note_on_idx=i)

            # Check if next token is TAB
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.get("type") == "TAB":
                    note.tab = (next_token["string"], next_token["fret"])
                    note.tab_idx = i + 1
                    note.computed_pitch = compute_pitch_from_tab(
                        next_token["string"], next_token["fret"]
                    )
                    used_tab_indices.add(i + 1)
                    
                    # Store for tuning analysis
                    if note.computed_pitch != -1:
                        valid_pitch_pairs.append((pitch, note.computed_pitch))

                    # Check pitch mismatch (deferred)
                    if note.computed_pitch != pitch:
                         potential_pitch_mismatches.append({
                            "token_idx": i,
                            "details": {
                                "note_on_pitch": pitch,
                                "tab_string": next_token["string"],
                                "tab_fret": next_token["fret"],
                                "computed_pitch": note.computed_pitch,
                            }
                        })
                else:
                    # Missing TAB
                    issues.append(StructuralIssue(
                        issue_type="missing_tab",
                        sample_idx=sample_idx,
                        token_idx=i,
                        details={
                            "note_on_pitch": pitch,
                            "next_token_type": next_token.get("type"),
                            "next_token_str": vocab.id_to_token.get(token_ids[i + 1], "UNKNOWN"),
                        }
                    ))
            else:
                # NOTE_ON at end of sequence, no TAB follows
                issues.append(StructuralIssue(
                    issue_type="missing_tab",
                    sample_idx=sample_idx,
                    token_idx=i,
                    details={
                        "note_on_pitch": pitch,
                        "reason": "end_of_sequence",
                    }
                ))

            # Track this note as open (needs NOTE_OFF)
            open_notes[pitch].append((i, note.tab_idx, note.tab))
            current_segment_notes.append(note)

        elif token_type == "TAB":
            # Check if this TAB was already used
            if i not in used_tab_indices:
                issues.append(StructuralIssue(
                    issue_type="orphan_tab",
                    sample_idx=sample_idx,
                    token_idx=i,
                    details={
                        "tab_string": token["string"],
                        "tab_fret": token["fret"],
                        "prev_token_type": tokens[i - 1].get("type") if i > 0 else None,
                    }
                ))

        elif token_type == "NOTE_OFF":
            pitch = token["pitch"]

            if pitch in open_notes and len(open_notes[pitch]) > 0:
                # Close the oldest open note with this pitch
                note_on_idx, tab_idx, tab_info = open_notes[pitch].pop(0)

                # Find the Note in current_segment_notes and update it
                for note in current_segment_notes:
                    if note.note_on_idx == note_on_idx and note.note_off_idx is None:
                        note.note_off_idx = i
                        break
            else:
                # Orphan NOTE_OFF
                issues.append(StructuralIssue(
                    issue_type="orphan_note_off",
                    sample_idx=sample_idx,
                    token_idx=i,
                    details={
                        "pitch": pitch,
                        "open_pitches": list(open_notes.keys()),
                    }
                ))

        elif token_type == "TIME_SHIFT":
            # End current segment
            if current_segment_notes or segment_start_idx < i:
                segments.append(TimeSegment(
                    notes=current_segment_notes.copy(),
                    time_shift_delta=token.get("delta"),
                    time_shift_idx=i,
                    start_idx=segment_start_idx,
                    end_idx=i,
                ))
            current_segment_notes = []
            segment_start_idx = i + 1

        i += 1

    # Handle any remaining notes (no trailing TIME_SHIFT)
    if current_segment_notes:
        segments.append(TimeSegment(
            notes=current_segment_notes,
            time_shift_delta=None,
            time_shift_idx=None,
            start_idx=segment_start_idx,
            end_idx=len(tokens) - 1,
        ))

    # Check for unclosed NOTE_ONs (missing NOTE_OFF)
    for pitch, open_list in open_notes.items():
        for note_on_idx, tab_idx, tab_info in open_list:
            issues.append(StructuralIssue(
                issue_type="missing_note_off",
                sample_idx=sample_idx,
                token_idx=note_on_idx,
                details={
                    "pitch": pitch,
                    "note_on_idx": note_on_idx,
                }
            ))

    # Analyze tuning and process pitch mismatches
    tuning_offset = 0
    if valid_pitch_pairs:
        # Calculate offsets: computed (standard) - actual
        offsets = [comp - actual for actual, comp in valid_pitch_pairs]
        counts = Counter(offsets)
        if counts:
            most_common_offset, count = counts.most_common(1)[0]
            # If the most common offset accounts for a significant portion (e.g. > 30%), use it
            if count > len(valid_pitch_pairs) * 0.3:
                tuning_offset = most_common_offset

    for pm in potential_pitch_mismatches:
        diff = pm["details"]["computed_pitch"] - pm["details"]["note_on_pitch"]
        if diff != tuning_offset:
            # Real mismatch
            pm["details"]["inferred_tuning_offset"] = tuning_offset
            issues.append(StructuralIssue(
                issue_type="pitch_mismatch",
                sample_idx=sample_idx,
                token_idx=pm["token_idx"],
                details=pm["details"]
            ))

    return issues, segments


# ============================================================================
# Comparative Analysis (Categories 2, 3, 4)
# ============================================================================

def compare_sequences(
    pred_ids: List[int],
    target_ids: List[int],
    vocab: Vocabulary,
    sample_idx: int,
    output_format: str = "v1",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Compare prediction vs target sequences.

    Args:
        pred_ids: Prediction token IDs
        target_ids: Target token IDs
        vocab: Vocabulary
        sample_idx: Sample index
        output_format: "v1", "v2", or "v3"

    Returns:
        Tuple of (extra_notes, absent_notes, time_shift_mismatches)
    """
    pred_tokens = parse_tokens(pred_ids, vocab)
    target_tokens = parse_tokens(target_ids, vocab)

    extra_notes: List[Dict] = []
    absent_notes: List[Dict] = []
    time_shift_mismatches: List[Dict] = []

    # Extract TIME_SHIFT boundaries to align segments
    pred_time_shifts = []
    target_time_shifts = []

    for i, t in enumerate(pred_tokens):
        if t.get("type") == "TIME_SHIFT":
            pred_time_shifts.append((i, t.get("delta", 0)))

    for i, t in enumerate(target_tokens):
        if t.get("type") == "TIME_SHIFT":
            target_time_shifts.append((i, t.get("delta", 0)))

    # Compare number of time segments
    if len(pred_time_shifts) != len(target_time_shifts):
        time_shift_mismatches.append({
            "type": "segment_count_mismatch",
            "pred_segments": len(pred_time_shifts),
            "target_segments": len(target_time_shifts),
        })

    # Compare TIME_SHIFT deltas
    min_segments = min(len(pred_time_shifts), len(target_time_shifts))
    for seg_idx in range(min_segments):
        pred_delta = pred_time_shifts[seg_idx][1]
        target_delta = target_time_shifts[seg_idx][1]
        if pred_delta != target_delta:
            time_shift_mismatches.append({
                "type": "delta_mismatch",
                "segment_idx": seg_idx,
                "pred_delta": pred_delta,
                "target_delta": target_delta,
            })

    # Compare notes per time segment
    def get_time_segment_pitches(tokens: List[Dict], start: int, end: int, format: str = "v1") -> List[int]:
        """Get pitches in a time segment (from NOTE_ON for v1/v3, from TAB for v2)."""
        pitches = []
        for t in tokens[start:end]:
            if format == "v2":
                # For v2, extract pitch from TAB tokens
                if t.get("type") == "TAB":
                    pitch = compute_pitch_from_tab(t["string"], t["fret"])
                    pitches.append(pitch)
            else:
                # For v1/v3, use NOTE_ON
                if t.get("type") == "NOTE_ON":
                    pitches.append(t["pitch"])
        return pitches

    # Build time segments for both pred and target
    pred_time_segments = []
    target_time_segments = []

    # Pred time segments
    prev_end = 0
    for idx, delta in pred_time_shifts:
        pred_time_segments.append((prev_end, idx))
        prev_end = idx + 1
    if prev_end < len(pred_tokens):
        pred_time_segments.append((prev_end, len(pred_tokens)))

    # Target time segments
    prev_end = 0
    for idx, delta in target_time_shifts:
        target_time_segments.append((prev_end, idx))
        prev_end = idx + 1
    if prev_end < len(target_tokens):
        target_time_segments.append((prev_end, len(target_tokens)))

    # Calculate total number of segments for position classification
    total_segments = max(len(pred_time_segments), len(target_time_segments))

    # Helper function to determine position in sequence
    def get_position_category(seg_idx: int, total_segs: int) -> str:
        """Classify segment position into 10 bins (0-9) representing position in sequence."""
        if total_segs <= 1:
            return "single_segment"

        # Divide into 10 equal bins
        # seg_idx ranges from 0 to total_segs-1
        # bin ranges from 0 to 9
        bin_size = total_segs / 10.0
        bin_idx = int(seg_idx / bin_size)

        # Ensure bin_idx is in valid range [0, 9]
        bin_idx = min(bin_idx, 9)

        return f"bin_{bin_idx}"

    # Compare notes in each time segment
    for seg_idx in range(min(len(pred_time_segments), len(target_time_segments))):
        pred_start, pred_end = pred_time_segments[seg_idx]
        target_start, target_end = target_time_segments[seg_idx]

        pred_pitches = get_time_segment_pitches(pred_tokens, pred_start, pred_end, output_format)
        target_pitches = get_time_segment_pitches(target_tokens, target_start, target_end, output_format)

        pred_pitch_set = set(pred_pitches)
        target_pitch_set = set(target_pitches)

        # Determine position for this segment
        position = get_position_category(seg_idx, total_segments)

        # Extra pitches in prediction
        for pitch in pred_pitch_set - target_pitch_set:
            extra_notes.append({
                "segment_idx": seg_idx,
                "pitch": pitch,
                "type": "hallucinated_pitch",
                "position": position,
            })

        # Missing pitches in prediction
        for pitch in target_pitch_set - pred_pitch_set:
            absent_notes.append({
                "segment_idx": seg_idx,
                "pitch": pitch,
                "type": "missing_pitch",
                "position": position,
            })

        # Check for duplicated notes
        from collections import Counter
        pred_counter = Counter(pred_pitches)
        target_counter = Counter(target_pitches)

        for pitch, pred_count in pred_counter.items():
            target_count = target_counter.get(pitch, 0)
            if pred_count > target_count:
                extra_notes.append({
                    "segment_idx": seg_idx,
                    "pitch": pitch,
                    "type": "duplicated_note",
                    "pred_count": pred_count,
                    "target_count": target_count,
                    "position": position,
                })
            elif pred_count < target_count:
                absent_notes.append({
                    "segment_idx": seg_idx,
                    "pitch": pitch,
                    "type": "missing_occurrence",
                    "pred_count": pred_count,
                    "target_count": target_count,
                    "position": position,
                })

    # Calculate total duration of target
    target_total_duration = 0
    if target_time_shifts:
        # Sum of all deltas
        target_total_duration = sum(t['delta'] for t in target_tokens if t.get('type') == 'TIME_SHIFT')
    
    # Calculate start times for pred segments
    current_pred_time = 0
    pred_segment_start_times = []
    
    # Iterate through segments (which end at a time shift)
    for idx, (start, end) in enumerate(pred_time_segments):
        pred_segment_start_times.append(current_pred_time)
        
        # Add delta of this segment's time shift (which is at 'end' index if it exists)
        # Note: 'end' is exclusive range [start:end], so the time shift is at end-1?
        # Actually compare_sequences construction:
        # pred_time_shifts contains (idx, delta). 
        # pred_segments constructed as: (prev_end, idx). 
        # Wait, let's look at construction:
        # for idx, delta in pred_time_shifts:
        #    pred_segments.append((prev_end, idx))
        #    prev_end = idx + 1
        # So segment goes up to 'idx' (exclusive). 
        # The token at 'idx' is the TIME_SHIFT.
        # So pred_tokens[idx] is the TIME_SHIFT.
        
        # Careful: pred_time_segments[i] is (start, end).
        # The TIME_SHIFT token is at index `end` in the original token list?
        # No, look at construction:
        #   pred_time_shifts.append((i, delta)) -> i is index of TIME_SHIFT
        #   pred_segments.append((prev_end, idx)) -> idx is index of TIME_SHIFT
        # So pred_tokens[idx] IS the time shift.
        
        if idx < len(pred_time_shifts):
             # This segment ends with a TIME_SHIFT
             shift_idx = pred_time_shifts[idx][0]
             delta = pred_time_shifts[idx][1]
             current_pred_time += delta
        else:
             # Last segment (trailing notes, no time shift at end)
             pass

    # Check for extra/missing time segments at the end
    if len(pred_time_segments) > len(target_time_segments):
        for seg_idx in range(len(target_time_segments), len(pred_time_segments)):
            pred_start, pred_end = pred_time_segments[seg_idx]

            # Determine position for this segment
            position = get_position_category(seg_idx, total_segments)

            # Check if this segment is temporally beyond target duration
            # Use a small tolerance (e.g., 10 ticks)
            tolerance = 10
            is_post_end = False

            if seg_idx < len(pred_segment_start_times):
                seg_start_time = pred_segment_start_times[seg_idx]
                if seg_start_time > target_total_duration + tolerance:
                    is_post_end = True

            pitches = get_time_segment_pitches(pred_tokens, pred_start, pred_end, output_format)
            for pitch in pitches:
                if is_post_end:
                    extra_notes.append({
                        "segment_idx": seg_idx,
                        "pitch": pitch,
                        "type": "post_end_hallucination",
                        "time": seg_start_time if seg_idx < len(pred_segment_start_times) else -1,
                        "position": position,
                    })
                else:
                    # It's extra in terms of index, but within time.
                    # This is likely a structural mismatch (fragmentation).
                    # We can mark it as 'structural_mismatch' or 'extra_time_segment_within_bounds'
                    extra_notes.append({
                        "segment_idx": seg_idx,
                        "pitch": pitch,
                        "type": "segmentation_mismatch_extra",
                        "time": seg_start_time if seg_idx < len(pred_segment_start_times) else -1,
                        "position": position,
                    })

    if len(target_time_segments) > len(pred_time_segments):
        for seg_idx in range(len(pred_time_segments), len(target_time_segments)):
            target_start, target_end = target_time_segments[seg_idx]

            # Determine position for this segment
            position = get_position_category(seg_idx, total_segments)

            pitches = get_time_segment_pitches(target_tokens, target_start, target_end, output_format)
            for pitch in pitches:
                absent_notes.append({
                    "segment_idx": seg_idx,
                    "pitch": pitch,
                    "type": "truncated_sequence",
                    "position": position,
                })

    return extra_notes, absent_notes, time_shift_mismatches


def analyze_timeline_alignment(
    pred_tokens: List[Dict],
    target_tokens: List[Dict],
    tolerance: int = 10,
) -> TimelineAlignmentAnalysis:
    """
    Analyze alignment between predicted and target timelines.

    Args:
        pred_tokens: Parsed prediction tokens (from parse_tokens)
        target_tokens: Parsed target tokens (from parse_tokens)
        tolerance: Maximum offset (in ticks) to consider a match (default: 10)

    Returns:
        TimelineAlignmentAnalysis with detailed alignment metrics
    """
    # Build absolute timelines
    pred_timeline = build_absolute_timeline(pred_tokens)
    target_timeline = build_absolute_timeline(target_tokens)

    # Analysis 1: Target-based coverage (Recall)
    # For each target cut point, find if there's a matching prediction cut
    covered_target_cuts = []
    missing_target_cuts = []
    matched_pairs = []

    for target_pos in target_timeline:
        # Find closest prediction cut point
        closest_pred, min_offset = find_closest_cut(target_pos, pred_timeline)

        if min_offset <= tolerance:
            # This target cut is covered by a prediction
            covered_target_cuts.append(target_pos)
            matched_pairs.append({
                'target_pos': target_pos,
                'pred_pos': closest_pred,
                'offset': closest_pred - target_pos
            })
        else:
            # This target cut is missing in prediction
            missing_target_cuts.append(target_pos)

    # Analysis 2: Prediction-based precision
    # For each prediction cut point, find if there's a matching target cut
    correct_pred_cuts = []
    hallucinated_pred_cuts = []

    for pred_pos in pred_timeline:
        closest_target, min_offset = find_closest_cut(pred_pos, target_timeline)

        if min_offset <= tolerance:
            # This prediction cut is correct (matches a target)
            correct_pred_cuts.append(pred_pos)
        else:
            # This prediction cut is hallucinated (no matching target)
            hallucinated_pred_cuts.append(pred_pos)

    # Calculate metrics
    target_coverage_rate = len(covered_target_cuts) / len(target_timeline) if target_timeline else 0.0
    pred_precision_rate = len(correct_pred_cuts) / len(pred_timeline) if pred_timeline else 0.0

    # Calculate F1 score
    f1_score = 0.0
    if target_coverage_rate + pred_precision_rate > 0:
        f1_score = 2 * (target_coverage_rate * pred_precision_rate) / (target_coverage_rate + pred_precision_rate)

    # Timing offset statistics (only for matched pairs)
    offsets = [pair['offset'] for pair in matched_pairs]
    mean_offset = float(np.mean(offsets)) if offsets else 0.0
    std_offset = float(np.std(offsets)) if offsets else 0.0
    max_offset = int(max([abs(o) for o in offsets])) if offsets else 0

    return TimelineAlignmentAnalysis(
        target_timeline=target_timeline,
        covered_target_cuts=covered_target_cuts,
        missing_target_cuts=missing_target_cuts,
        pred_timeline=pred_timeline,
        correct_pred_cuts=correct_pred_cuts,
        hallucinated_pred_cuts=hallucinated_pred_cuts,
        matched_pairs=matched_pairs,
        target_coverage_rate=target_coverage_rate,
        pred_precision_rate=pred_precision_rate,
        f1_score=f1_score,
        mean_timing_offset=mean_offset,
        std_timing_offset=std_offset,
        max_timing_offset=max_offset,
    )


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_single_sample(
    pred_ids: List[int],
    target_ids: List[int],
    vocab: Vocabulary,
    sample_idx: int,
    output_format: str = "v1",
    timeline_tolerance: int = 10,
    calc_levenshtein: bool = False,
) -> SampleAnalysis:
    """Analyze a single sample."""
    # Remove padding
    pred_ids = [t for t in pred_ids if t != vocab.pad_id]
    target_ids = [t for t in target_ids if t != vocab.pad_id]

    # Structural analysis
    structural_issues, segments = validate_structure(pred_ids, vocab, sample_idx, output_format)

    # Comparative analysis
    extra_notes, absent_notes, time_mismatches = compare_sequences(
        pred_ids, target_ids, vocab, sample_idx, output_format
    )

    # Parse tokens for timeline analysis
    pred_tokens = parse_tokens(pred_ids, vocab)
    target_tokens = parse_tokens(target_ids, vocab)

    # Timeline alignment analysis
    timeline_alignment = analyze_timeline_alignment(
        pred_tokens, target_tokens, tolerance=timeline_tolerance
    )

    # Levenshtein metrics
    lev_dist = 0.0
    lev_sim = 0.0
    if calc_levenshtein:
        lev_dist = compute_levenshtein_distance(pred_ids, target_ids)
        max_len = max(len(pred_ids), len(target_ids))
        lev_sim = 1.0 - (lev_dist / max_len) if max_len > 0 else 1.0

    analysis = SampleAnalysis(
        sample_idx=sample_idx,
        num_segments=len(segments),
        structural_issues=structural_issues,
        extra_notes=extra_notes,
        absent_notes=absent_notes,
        time_shift_mismatches=time_mismatches,
        timeline_alignment=timeline_alignment,
        levenshtein_distance=lev_dist,
        levenshtein_similarity=lev_sim,
    )
    analysis.compute_counts()
    analysis.compute_position_distribution()

    # Compute timeline-scoped counts
    cutoff_idx = get_timeline_cutoff_index(pred_tokens, target_tokens, tolerance=timeline_tolerance)
    timeline_counts = defaultdict(int)

    # 1. Structural issues within timeline
    for issue in structural_issues:
        if issue.token_idx < cutoff_idx:
            timeline_counts[issue.issue_type] += 1

    # 2. Extra notes (exclude post-end hallucinations)
    # post_end_hallucination is explicitly marked in compare_sequences
    extra_count = 0
    for note in extra_notes:
        if note.get("type") != "post_end_hallucination":
            extra_count += 1
    timeline_counts["extra_note"] = extra_count

    # 3. Absent notes (always relevant to target timeline)
    timeline_counts["absent_note"] = len(absent_notes)

    # 4. Time shift mismatches (mostly relevant to the aligned part)
    timeline_counts["time_shift_mismatch"] = len(time_mismatches)

    analysis.issue_counts_timeline = dict(timeline_counts)

    return analysis


def analyze_all_predictions(
    predictions: List[List[int]],
    targets: List[List[int]],
    vocab: Vocabulary,
    output_format: str = "v1",
    timeline_tolerance: int = 10,
    verbose: bool = True,
    calc_levenshtein: bool = False,
) -> OutputAnalysisResult:
    """
    Analyze all predictions.

    Args:
        predictions: List of prediction token ID sequences
        targets: List of target token ID sequences
        vocab: Output vocabulary
        output_format: "v1", "v2", or "v3"
        timeline_tolerance: Tolerance in ticks for timeline alignment (default: 10)
        verbose: Print progress
        calc_levenshtein: Whether to calculate Levenshtein distance (default: False)

    Returns:
        OutputAnalysisResult with complete analysis
    """
    per_sample: List[SampleAnalysis] = []
    all_issues: List[Dict] = []
    total_segments = 0
    samples_with_issues = 0

    # Aggregate issue counts
    aggregate_counts: Dict[str, int] = defaultdict(int)
    aggregate_counts_timeline: Dict[str, int] = defaultdict(int)

    # Aggregate position distributions
    aggregate_extra_pos_dist: Dict[str, int] = defaultdict(int)
    aggregate_absent_pos_dist: Dict[str, int] = defaultdict(int)

    # Aggregate timeline metrics
    timeline_coverages = []
    timeline_precisions = []
    timeline_f1s = []
    all_offsets = []

    lev_dists = []
    lev_sims = []

    if verbose:
        from tqdm import tqdm
        iterator = tqdm(enumerate(zip(predictions, targets)),
                       total=len(predictions),
                       desc="Analyzing samples")
    else:
        iterator = enumerate(zip(predictions, targets))

    for sample_idx, (pred, target) in iterator:
        analysis = analyze_single_sample(
            pred, target, vocab, sample_idx, output_format, timeline_tolerance, calc_levenshtein
        )
        per_sample.append(analysis)

        total_segments += analysis.num_segments

        # Aggregate counts
        for issue_type, count in analysis.issue_counts.items():
            aggregate_counts[issue_type] += count

        for issue_type, count in analysis.issue_counts_timeline.items():
            aggregate_counts_timeline[issue_type] += count

        # Aggregate position distributions
        for position, count in analysis.extra_notes_position_dist.items():
            aggregate_extra_pos_dist[position] += count

        for position, count in analysis.absent_notes_position_dist.items():
            aggregate_absent_pos_dist[position] += count

        # Check if sample has issues
        if sum(analysis.issue_counts.values()) > 0:
            samples_with_issues += 1

        # Collect timeline metrics
        if analysis.timeline_alignment:
            timeline_coverages.append(analysis.timeline_alignment.target_coverage_rate)
            timeline_precisions.append(analysis.timeline_alignment.pred_precision_rate)
            timeline_f1s.append(analysis.timeline_alignment.f1_score)
            all_offsets.extend([p['offset'] for p in analysis.timeline_alignment.matched_pairs])

        if calc_levenshtein:
            lev_dists.append(analysis.levenshtein_distance)
            lev_sims.append(analysis.levenshtein_similarity)

        # Flatten issues

        for issue in analysis.structural_issues:
            all_issues.append(issue.to_dict())

        for extra in analysis.extra_notes:
            all_issues.append({
                "issue_type": "extra_note",
                "sample_idx": sample_idx,
                "token_idx": -1,
                "details": extra,
            })

        for absent in analysis.absent_notes:
            all_issues.append({
                "issue_type": "absent_note",
                "sample_idx": sample_idx,
                "token_idx": -1,
                "details": absent,
            })

        for mismatch in analysis.time_shift_mismatches:
            all_issues.append({
                "issue_type": "time_shift_mismatch",
                "sample_idx": sample_idx,
                "token_idx": -1,
                "details": mismatch,
            })

    # Compute aggregate timeline metrics
    avg_coverage = float(np.mean(timeline_coverages)) if timeline_coverages else 0.0
    avg_precision = float(np.mean(timeline_precisions)) if timeline_precisions else 0.0
    avg_f1 = float(np.mean(timeline_f1s)) if timeline_f1s else 0.0
    avg_offset = float(np.mean(all_offsets)) if all_offsets else 0.0
    std_offset = float(np.std(all_offsets)) if all_offsets else 0.0

    avg_lev_dist = float(np.mean(lev_dists)) if lev_dists else 0.0
    avg_lev_sim = float(np.mean(lev_sims)) if lev_sims else 0.0

    return OutputAnalysisResult(
        total_samples=len(predictions),
        total_segments=total_segments,
        samples_with_issues=samples_with_issues,
        issue_counts=dict(aggregate_counts),
        issue_counts_timeline=dict(aggregate_counts_timeline),
        per_sample=per_sample,
        all_issues=all_issues,
        avg_timeline_coverage=avg_coverage,
        avg_timeline_precision=avg_precision,
        avg_timeline_f1=avg_f1,
        avg_timing_offset=avg_offset,
        std_timing_offset=std_offset,
        avg_levenshtein_distance=avg_lev_dist,
        avg_levenshtein_similarity=avg_lev_sim,
        extra_notes_position_dist=dict(aggregate_extra_pos_dist),
        absent_notes_position_dist=dict(aggregate_absent_pos_dist),
    )


# ============================================================================
# Output Functions
# ============================================================================

def save_results_json(result: OutputAnalysisResult, path: str):
    """Save analysis results to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


def print_timeline_summary(result: OutputAnalysisResult):
    """Print timeline alignment summary."""
    print("\n" + "=" * 80)
    print("Timeline Alignment Analysis")
    print("=" * 80)

    print(f"\nOverall Timeline Metrics:")
    print(f"  Target Coverage (Recall):    {result.avg_timeline_coverage * 100:6.2f}%")
    print(f"  Prediction Precision:        {result.avg_timeline_precision * 100:6.2f}%")
    print(f"  F1 Score:                    {result.avg_timeline_f1 * 100:6.2f}%")

    print(f"\nTiming Offset Statistics (for matched cut points):")
    print(f"  Mean offset:                 {result.avg_timing_offset:6.2f} ticks")
    print(f"  Std deviation:               {result.std_timing_offset:6.2f} ticks")
    print(f"  (Reference: 480 ticks = 1 quarter note)")

    print("=" * 80)


def generate_timeline_report(
    analysis_result: OutputAnalysisResult,
    top_k: int = 10,
) -> str:
    """
    Generate a detailed report for the top-k worst timeline alignment samples.
    
    Args:
        analysis_result: The complete analysis result.
        top_k: Number of samples to include.
        
    Returns:
        String containing the report.
    """
    # Sort samples by F1 score (ascending) -> worst first
    # Filter out samples with no timeline analysis
    valid_samples = [s for s in analysis_result.per_sample if s.timeline_alignment]
    sorted_samples = sorted(
        valid_samples,
        key=lambda s: s.timeline_alignment.f1_score
    )
    
    worst_samples = sorted_samples[:top_k]
    
    report = []
    report.append("=" * 80)
    report.append(f"TIMELINE ALIGNMENT ANALYSIS - TOP {top_k} WORST SAMPLES")
    report.append("=" * 80)
    report.append("\nRanking metric: F1 Score (Lower is worse)\n")
    
    for rank, sample in enumerate(worst_samples, 1):
        align = sample.timeline_alignment
        report.append("-" * 80)
        report.append(f"Rank {rank}: Sample {sample.sample_idx}")
        report.append(f"F1: {align.f1_score:.4f} | Coverage: {align.target_coverage_rate:.4f} | Precision: {align.pred_precision_rate:.4f}")
        report.append("-" * 80)
        
        # Combine all events for a chronological view
        events = []
        
        # 1. Matched pairs
        for pair in align.matched_pairs:
            events.append({
                'pos': pair['target_pos'], 
                'type': 'MATCH', 
                'target': pair['target_pos'], 
                'pred': pair['pred_pos'],
                'diff': pair['offset']
            })
            
        # 2. Missing targets (False Negatives)
        for pos in align.missing_target_cuts:
            events.append({
                'pos': pos,
                'type': 'MISSING (FN)',
                'target': pos,
                'pred': '---',
                'diff': '---'
            })
            
        # 3. Hallucinated preds (False Positives)
        for pos in align.hallucinated_pred_cuts:
            events.append({
                'pos': pos,
                'type': 'EXTRA (FP)',
                'target': '---',
                'pred': pos,
                'diff': '---'
            })
            
        # Sort by position (target pos for match/missing, pred pos for extra)
        events.sort(key=lambda x: x['pos'])
        
        report.append(f"{'Type':<15s} | {'Target (Ticks)':<15s} | {'Pred (Ticks)':<15s} | {'Diff':<10s}")
        report.append("-" * 65)
        
        for e in events:
            report.append(f"{e['type']:<15s} | {str(e['target']):<15s} | {str(e['pred']):<15s} | {str(e['diff']):<10s}")
            
        report.append("\n")
        
    return "\n".join(report)


def print_issue_summary(result: OutputAnalysisResult, output_format: str = "v1"):
    """Print structural issue summary."""
    print("=" * 80)
    print(f"Output Format Analysis Report (Format: {output_format})")
    print("=" * 80)

    print(f"\nOverall Statistics:")
    print(f"  Total samples analyzed: {result.total_samples}")
    print(f"  Total segments: {result.total_segments}")
    pct = result.samples_with_issues / result.total_samples * 100 if result.total_samples > 0 else 0
    print(f"  Samples with issues: {result.samples_with_issues} ({pct:.1f}%)")

    # Structural issues depend on format
    if output_format == "v2":
        print(f"\nStructural Issues (v2 format - TAB, TIME_SHIFT only):")
        structural_types = ["unexpected_token_in_v2"]
    else:
        print(f"\nStructural Issues (Note-On/Tab/Note-Off Mismatch):")
        structural_types = ["missing_tab", "missing_note_off", "pitch_mismatch",
                           "orphan_note_off", "orphan_tab"]

    print(f"{'Issue Type':<30s} {'Total':>10s} {'Within Timeline':>18s}")
    print("-" * 60)
    
    for issue_type in structural_types:
        count = result.issue_counts.get(issue_type, 0)
        count_timeline = result.issue_counts_timeline.get(issue_type, 0)
        print(f"{issue_type.replace('_', ' ').title():<30s} {count:>10d} {count_timeline:>18d}")

    print(f"\nNote Count Issues:")
    
    extra_count = result.issue_counts.get('extra_note', 0)
    extra_count_timeline = result.issue_counts_timeline.get('extra_note', 0)
    print(f"{'Extra notes':<30s} {extra_count:>10d} {extra_count_timeline:>18d}")
    
    absent_count = result.issue_counts.get('absent_note', 0)
    absent_count_timeline = result.issue_counts_timeline.get('absent_note', 0)
    print(f"{'Absent notes':<30s} {absent_count:>10d} {absent_count_timeline:>18d}")

    print(f"\nTiming Issues:")
    time_mismatch_count = result.issue_counts.get('time_shift_mismatch', 0)
    time_mismatch_count_timeline = result.issue_counts_timeline.get('time_shift_mismatch', 0)
    print(f"{'TIME_SHIFT mismatch':<30s} {time_mismatch_count:>10d} {time_mismatch_count_timeline:>18d}")

    print("\n" + "=" * 80)
    
    print(f"\nSequence Similarity (Levenshtein):")
    print(f"  Average Distance:            {result.avg_levenshtein_distance:6.2f}")
    print(f"  Average Similarity:          {result.avg_levenshtein_similarity * 100:6.2f}%")
    print("=" * 80)


def print_position_distribution(result: OutputAnalysisResult):
    """Print position distribution analysis for extra and absent notes."""
    print("\n" + "=" * 80)
    print("Position Distribution Analysis (10-bin granularity)")
    print("=" * 80)
    print("Note: Sequences are divided into 10 equal bins from start (0) to end (9)")
    print()

    # Define position order for display
    positions = ["single_segment"] + [f"bin_{i}" for i in range(10)]

    def get_position_label(pos: str) -> str:
        """Convert position key to display label."""
        if pos == "single_segment":
            return "Single Segment"
        elif pos.startswith("bin_"):
            bin_num = int(pos.split("_")[1])
            pct_start = bin_num * 10
            pct_end = (bin_num + 1) * 10
            return f"Bin {bin_num} ({pct_start}-{pct_end}%)"
        return pos

    # Extra notes distribution
    print("\nExtra Notes Position Distribution:")
    print("-" * 80)

    total_extra = sum(result.extra_notes_position_dist.values())
    if total_extra > 0:
        print(f"Total extra notes: {total_extra}")
        print("\nDistribution across sequence:")

        print(f"{'Position':<25s} {'Count':>10s} {'Percentage':>12s} {'Bar Chart':<35s}")
        print("-" * 85)

        for pos in positions:
            count = result.extra_notes_position_dist.get(pos, 0)
            if count > 0:
                pct = count / total_extra * 100
                bar_length = int(pct / 100 * 40)
                bar = "█" * bar_length
                label = get_position_label(pos)
                print(f"{label:<25s} {count:>10d} {pct:>11.1f}% {bar:<35s}")

        print("-" * 85)
    else:
        print("No extra notes found.")

    # Absent notes distribution
    print("\nAbsent Notes Position Distribution:")
    print("-" * 80)

    total_absent = sum(result.absent_notes_position_dist.values())
    if total_absent > 0:
        print(f"Total absent notes: {total_absent}")
        print("\nDistribution across sequence:")

        print(f"{'Position':<25s} {'Count':>10s} {'Percentage':>12s} {'Bar Chart':<35s}")
        print("-" * 85)

        for pos in positions:
            count = result.absent_notes_position_dist.get(pos, 0)
            if count > 0:
                pct = count / total_absent * 100
                bar_length = int(pct / 100 * 40)
                bar = "█" * bar_length
                label = get_position_label(pos)
                print(f"{label:<25s} {count:>10d} {pct:>11.1f}% {bar:<35s}")

        print("-" * 85)
    else:
        print("No absent notes found.")

    # Summary insights with grouped analysis
    print("\nKey Insights:")
    print("-" * 80)

    if total_extra > 0:
        max_extra_pos = max(result.extra_notes_position_dist.items(), key=lambda x: x[1])
        max_extra_pct = max_extra_pos[1] / total_extra * 100
        max_extra_label = get_position_label(max_extra_pos[0])
        print(f"• Extra notes are most common in {max_extra_label} ({max_extra_pct:.1f}%)")

        # Group into beginning/middle/end for high-level summary
        beginning_count = sum(result.extra_notes_position_dist.get(f"bin_{i}", 0) for i in range(0, 3))
        middle_count = sum(result.extra_notes_position_dist.get(f"bin_{i}", 0) for i in range(3, 7))
        end_count = sum(result.extra_notes_position_dist.get(f"bin_{i}", 0) for i in range(7, 10))

        print(f"  - Beginning (0-30%): {beginning_count} ({beginning_count/total_extra*100:.1f}%)")
        print(f"  - Middle (30-70%):   {middle_count} ({middle_count/total_extra*100:.1f}%)")
        print(f"  - End (70-100%):     {end_count} ({end_count/total_extra*100:.1f}%)")

    if total_absent > 0:
        max_absent_pos = max(result.absent_notes_position_dist.items(), key=lambda x: x[1])
        max_absent_pct = max_absent_pos[1] / total_absent * 100
        max_absent_label = get_position_label(max_absent_pos[0])
        print(f"• Absent notes are most common in {max_absent_label} ({max_absent_pct:.1f}%)")

        # Group into beginning/middle/end for high-level summary
        beginning_count = sum(result.absent_notes_position_dist.get(f"bin_{i}", 0) for i in range(0, 3))
        middle_count = sum(result.absent_notes_position_dist.get(f"bin_{i}", 0) for i in range(3, 7))
        end_count = sum(result.absent_notes_position_dist.get(f"bin_{i}", 0) for i in range(7, 10))

        print(f"  - Beginning (0-30%): {beginning_count} ({beginning_count/total_absent*100:.1f}%)")
        print(f"  - Middle (30-70%):   {middle_count} ({middle_count/total_absent*100:.1f}%)")
        print(f"  - End (70-100%):     {end_count} ({end_count/total_absent*100:.1f}%)")

    print("=" * 80)


def print_summary(result: OutputAnalysisResult, output_format: str = "v1"):
    """Print full analysis summary to console (wrapper for backward compatibility)."""
    print_issue_summary(result, output_format)
    # Timeline alignment summary appended
    print_timeline_summary(result)
    # Position distribution summary
    print_position_distribution(result)

