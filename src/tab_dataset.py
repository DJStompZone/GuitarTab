"""
Dataset for guitar tablature transcription using DadaGP tokens.

Loads .tokens.txt files, converts to our event format, tokenizes, and segments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypedDict
import numpy as np
from tqdm import tqdm
import torch

from src.dadagp_parser import (
    parse_dadagp_file,
    dadagp_to_events,
    DadaGPToken,
    StructuralToken,
    Event,
    NoteOnEvent,
    NoteOffEvent,
    TimeShiftEvent,
    TabEvent,
)


# ============================================================================
# Time shift schema
# ============================================================================

TICKS_PER_BEAT = 480

# Allowed beat-fraction set for the new representation.
# NOTE: This is dormant by default (time_shift_mode="ticks") so it won't
# affect existing training until explicitly enabled.
BEAT_FRACTION_PAIRS: list[tuple[int, int]] = [
    # Long values (multi-beat)
    (4, 1),
    (2, 1),
    (1, 1),
    # Triplets / special groupings
    (2, 3),
    (1, 3),
    (1, 2),
    (1, 4),
    (1, 6),
    (1, 8),
    (1, 12),
    (1, 16),
    (1, 24),
    (1, 32),
]


def get_allowed_time_shift_ticks(
    max_time_shift: int,
    time_shift_mode: str = "ticks",
) -> list[int]:
    """
    Returns the list of allowed TIME_SHIFT deltas (in ticks).

    - "ticks": legacy behaviour, 1..max_time_shift (dense integer ticks)
    - "beat_fraction": restricted to musically meaningful beat fractions,
      still expressed as integer ticks, using TICKS_PER_BEAT=480.
    """
    if time_shift_mode == "ticks":
        return list(range(1, max_time_shift + 1))
    elif time_shift_mode == "beat_fraction":
        allowed: set[int] = set()
        for num, den in BEAT_FRACTION_PAIRS:
            # Only keep exact integer multiples in ticks
            ticks_num = TICKS_PER_BEAT * num
            if ticks_num % den != 0:
                continue
            ticks = ticks_num // den
            if 1 <= ticks <= max_time_shift:
                allowed.add(ticks)
        return sorted(allowed)
    else:
        raise ValueError(f"Unknown time_shift_mode: {time_shift_mode}")


# ============================================================================
# Event to Token ID Conversion
# ============================================================================


@dataclass
class Vocabulary:
    """Vocabulary for tokenization."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    vocab_size: int

    # Special token IDs
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    # Optional: document which TIME_SHIFT deltas this vocab supports (in ticks)
    time_shift_values: Optional[list[int]] = None


def build_vocabulary(
    max_pitch: int = 127,
    max_time_shift: int = 500,
    num_strings: int = 6,
    num_frets: int = 21,
    output_format: str = "v1",
    time_shift_mode: str = "ticks",
) -> tuple[Vocabulary, Vocabulary]:
    """
    Build vocabularies for input and output sequences.

    Args:
        max_pitch: Maximum MIDI pitch
        max_time_shift: Maximum time shift value
        num_strings: Number of guitar strings
        num_frets: Number of frets
        output_format: "v1", "v2", or "v3"
            - v1: input = NOTE_ON, NOTE_OFF, TIME_SHIFT; output = NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT
            - v2: input = NOTE_ON, NOTE_OFF, TIME_SHIFT; output = TAB, TIME_SHIFT
            - v3: input = NOTE_ON, TIME_SHIFT; output = NOTE_ON, TAB, TIME_SHIFT (no NOTE_OFF)

    Returns:
        Tuple of (input_vocab, output_vocab)
    """
    # Decide TIME_SHIFT deltas used by both input/output vocab
    time_shift_values = get_allowed_time_shift_ticks(
        max_time_shift=max_time_shift,
        time_shift_mode=time_shift_mode,
    )

    # Build input vocabulary (NOTE_ON, NOTE_OFF, TIME_SHIFT)
    input_token_to_id: dict[str, int] = {}
    input_id_to_token: dict[int, str] = {}
    idx = 0

    # Special tokens
    for token in ["PAD", "BOS", "EOS", "UNK"]:
        input_token_to_id[token] = idx
        input_id_to_token[idx] = token
        idx += 1

    # NOTE_ON tokens
    for pitch in range(max_pitch + 1):
        token = f"NOTE_ON_{pitch}"
        input_token_to_id[token] = idx
        input_id_to_token[idx] = token
        idx += 1
    
    # NOTE_OFF tokens (not for v3)
    if output_format != "v3":
        for pitch in range(max_pitch + 1):
            token = f"NOTE_OFF_{pitch}"
            input_token_to_id[token] = idx
            input_id_to_token[idx] = token
            idx += 1

    # TIME_SHIFT tokens
    for shift in time_shift_values:
        token = f"TIME_SHIFT_{shift}"
        input_token_to_id[token] = idx
        input_id_to_token[idx] = token
        idx += 1

    input_vocab = Vocabulary(
        token_to_id=input_token_to_id,
        id_to_token=input_id_to_token,
        vocab_size=idx,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        time_shift_values=time_shift_values,
    )

    # Build output vocabulary (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB)
    # v1: NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB
    # v2: TAB, TIME_SHIFT only (no NOTE_ON/OFF)
    # v3: NOTE_ON, TIME_SHIFT, TAB (no NOTE_OFF)
    output_token_to_id: dict[str, int] = {}
    output_id_to_token: dict[int, str] = {}
    idx = 0

    # Special tokens (always present)
    for token in ["PAD", "BOS", "EOS", "UNK"]:
        output_token_to_id[token] = idx
        output_id_to_token[idx] = token
        idx += 1

    if output_format in ("v1", "v3"):
        # NOTE_ON tokens (for v1 and v3)
        for pitch in range(max_pitch + 1):
            token = f"NOTE_ON_{pitch}"
            output_token_to_id[token] = idx
            output_id_to_token[idx] = token
            idx += 1

    if output_format == "v1":
        # NOTE_OFF tokens (only for v1)
        for pitch in range(max_pitch + 1):
            token = f"NOTE_OFF_{pitch}"
            output_token_to_id[token] = idx
            output_id_to_token[idx] = token
            idx += 1

    # TIME_SHIFT tokens (always present)
    for shift in time_shift_values:
        token = f"TIME_SHIFT_{shift}"
        output_token_to_id[token] = idx
        output_id_to_token[idx] = token
        idx += 1

    # TAB tokens (always present)
    for string in range(1, num_strings + 1):
        for fret in range(num_frets):
            token = f"TAB_{string}_{fret}"
            output_token_to_id[token] = idx
            output_id_to_token[idx] = token
            idx += 1

    output_vocab = Vocabulary(
        token_to_id=output_token_to_id,
        id_to_token=output_id_to_token,
        vocab_size=idx,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        time_shift_values=time_shift_values,
    )

    return input_vocab, output_vocab


def event_to_token_string(event: Event) -> str:
    """
    Convert event to token string.

    Args:
        event: Event object

    Returns:
        Token string (e.g., "NOTE_ON_60")
    """
    if isinstance(event, NoteOnEvent):
        return f"NOTE_ON_{event.pitch}"
    elif isinstance(event, NoteOffEvent):
        return f"NOTE_OFF_{event.pitch}"
    elif isinstance(event, TimeShiftEvent):
        return f"TIME_SHIFT_{event.delta}"
    elif isinstance(event, TabEvent):
        return f"TAB_{event.string}_{event.fret}"
    else:
        return "UNK"


def events_to_ids(events: list[Event], vocab: Vocabulary) -> list[int]:
    """
    Convert events to token IDs.

    Args:
        events: List of events
        vocab: Vocabulary

    Returns:
        List of token IDs
    """
    ids: list[int] = []
    for event in events:
        token_str = event_to_token_string(event)
        token_id = vocab.token_to_id.get(token_str, vocab.unk_id)
        ids.append(token_id)
    return ids


# ============================================================================
# Bar-Aligned Splitting
# ============================================================================


def _remap_bar_positions_after_output_filter(
    bar_positions: list[tuple[int, int]],
    full_output_events: list[Event],
    keep_event: Callable[[Event], bool],
) -> list[tuple[int, int]]:
    """
    dadagp_to_events() records bar boundaries as indices into the *full* output
    stream (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB). If we filter output events,
    those indices must be remapped to positions in the filtered list; otherwise
    _split_into_segments slices the wrong range and input/output desync.
    """
    if not bar_positions:
        return bar_positions

    def out_prefix_len(out_idx: int) -> int:
        return sum(1 for ev in full_output_events[:out_idx] if keep_event(ev))

    return [(inp_i, out_prefix_len(out_i)) for inp_i, out_i in bar_positions]


def find_split_bar_near_length(
    bar_positions: list[tuple[int, int]], target_length: int, current_bar_idx: int = 0
) -> Optional[int]:
    """
    Find the bar index closest to target_length from current bar.

    Args:
        bar_positions: List of (input_idx, output_idx) pairs for each bar
        target_length: Target sequence length (used for input)
        current_bar_idx: Current bar index

    Returns:
        Bar index to split at, or None if at end
    """
    if current_bar_idx >= len(bar_positions):
        return None

    # Calculate ideal split position in input sequence
    current_input_pos = bar_positions[current_bar_idx][0]
    ideal_input_split = current_input_pos + target_length

    # Find bars within acceptable range
    candidate_bars = []
    for bar_idx in range(current_bar_idx + 1, len(bar_positions)):
        input_pos = bar_positions[bar_idx][0]
        # Accept bars within ±50% of target length
        if current_input_pos < input_pos <= ideal_input_split + target_length // 2:
            candidate_bars.append(bar_idx)

    if not candidate_bars:
        # No bars found in range, return last bar
        return len(bar_positions)

    # Find closest to ideal split
    closest_bar_idx = min(
        candidate_bars, key=lambda idx: abs(bar_positions[idx][0] - ideal_input_split)
    )
    return closest_bar_idx


# ============================================================================
# Dataset
# ============================================================================


class TabDataset:
    """
    Dataset for guitar tablature transcription.

    Loads DadaGP .tokens.txt files and converts to segmented input/output pairs.
    """

    def __init__(
        self,
        token_files: list[str],
        max_sequence_length: int = 512,
        max_pitch: int = 127,
        max_time_shift: int = 500,
        num_strings: int = 6,
        num_frets: int = 21,
        output_format: str = "v1",
        time_shift_mode: str = "ticks",
    ):
        """
        Initialize dataset.

        Args:
            token_files: List of paths to .tokens.txt files
            max_sequence_length: Maximum sequence length for splitting
            max_pitch: Maximum MIDI pitch
            max_time_shift: Maximum time shift value
            num_strings: Number of guitar strings
            num_frets: Number of frets
            output_format: "v1", "v2", or "v3"
                - v1: input = NOTE_ON, NOTE_OFF, TIME_SHIFT; output = NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT
                - v2: input = NOTE_ON, NOTE_OFF, TIME_SHIFT; output = TAB, TIME_SHIFT
                - v3: input = NOTE_ON, TIME_SHIFT; output = NOTE_ON, TAB, TIME_SHIFT (no NOTE_OFF)
        """
        self.token_files = token_files
        self.max_sequence_length = max_sequence_length
        self.output_format = output_format
        self.time_shift_mode = time_shift_mode

        # Build vocabularies
        print("Building vocabularies...")

        # Format descriptions for clarity
        format_descriptions = {
            "v1": "NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT",
            "v2": "TAB, TIME_SHIFT",
            "v3": "NOTE_ON, TAB, TIME_SHIFT",
        }
        format_desc = format_descriptions.get(output_format, "unknown")
        print(f"Output format: {output_format} ({format_desc})")

        self.max_pitch = max_pitch
        self.max_time_shift = max_time_shift
        self.num_strings = num_strings
        self.input_vocab, self.output_vocab = build_vocabulary(
            max_pitch=max_pitch,
            max_time_shift=max_time_shift,
            num_strings=num_strings,
            num_frets=num_frets,
            output_format=output_format,
            time_shift_mode=time_shift_mode,
        )

        print(f"Input vocab size: {self.input_vocab.vocab_size}")
        print(f"Output vocab size: {self.output_vocab.vocab_size}")

        # Process files and create segments
        print("Processing files and creating segments...")
        self.segments = self._prepare_segments()

        print(f"Created {len(self.segments)} segments from {len(token_files)} files")

    def _prepare_segments(self) -> list[tuple[list[int], list[int]]]:
        """
        Process all files and create segments.

        Returns:
            List of (input_ids, output_ids) tuples
        """
        all_segments: list[tuple[list[int], list[int]]] = []
        self.segment_sources: list[str] = []  # maps segment index → source file path

        for token_file in tqdm(self.token_files, desc="Processing files"):
            try:
                # Parse DadaGP tokens
                dadagp_tokens = parse_dadagp_file(token_file)

                # Convert to events with bar positions
                input_events, output_events, bar_positions = dadagp_to_events(
                    dadagp_tokens, max_time_shift=self.max_time_shift
                )

                # Adjust events according to output_format
                if self.output_format == "v2":
                    # v2: input = NOTE_ON, NOTE_OFF, TIME_SHIFT
                    #     output = TAB, TIME_SHIFT (drop NOTE_ON/NOTE_OFF from output)
                    full_output_events = output_events
                    filtered_output_events: list[Event] = []
                    for ev in output_events:
                        if isinstance(ev, (TabEvent, TimeShiftEvent)):
                            filtered_output_events.append(ev)
                    output_events = filtered_output_events
                    bar_positions = _remap_bar_positions_after_output_filter(
                        bar_positions,
                        full_output_events,
                        lambda ev: isinstance(ev, (TabEvent, TimeShiftEvent)),
                    )
                elif self.output_format == "v3":
                    # v3: input = NOTE_ON, TIME_SHIFT (no NOTE_OFF in input)
                    #     output = NOTE_ON, TAB, TIME_SHIFT (drop NOTE_OFF from output)
                    full_input_events = input_events
                    full_output_events = output_events
                    input_events = [ev for ev in input_events if not isinstance(ev, NoteOffEvent)]
                    filtered_output_events: list[Event] = []
                    for ev in output_events:
                        if isinstance(ev, (NoteOnEvent, TabEvent, TimeShiftEvent)):
                            filtered_output_events.append(ev)
                    output_events = filtered_output_events
                    bar_positions = [
                        (
                            sum(
                                1
                                for ev in full_input_events[:inp_i]
                                if not isinstance(ev, NoteOffEvent)
                            ),
                            sum(
                                1
                                for ev in full_output_events[:out_i]
                                if isinstance(ev, (NoteOnEvent, TabEvent, TimeShiftEvent))
                            ),
                        )
                        for inp_i, out_i in bar_positions
                    ]

                # Convert to IDs
                input_ids = events_to_ids(input_events, self.input_vocab)
                output_ids = events_to_ids(output_events, self.output_vocab)

                # Split into segments at bar boundaries
                segments = self._split_into_segments(
                    input_ids, output_ids, bar_positions
                )

                for inp, out in segments:
                    out = [self.output_vocab.bos_id] + out + [self.output_vocab.eos_id]
                    if len(out) > self.max_sequence_length:
                        out = out[: self.max_sequence_length - 1] + [
                            self.output_vocab.eos_id
                        ]
                    all_segments.append((inp, out))
                    self.segment_sources.append(token_file)

                # all_segments.extend(segments)

            except Exception as e:
                print(f"Error processing {token_file}: {e}")
                continue

        return all_segments

    def _split_into_segments(
        self,
        input_ids: list[int],
        output_ids: list[int],
        bar_positions: list[tuple[int, int]],
    ) -> list[tuple[list[int], list[int]]]:
        """
        Split sequences into segments at bar boundaries.

        Args:
            input_ids: Input token IDs
            output_ids: Output token IDs
            bar_positions: List of (input_idx, output_idx) for each bar

        Returns:
            List of (input_segment, output_segment) tuples
        """
        segments: list[tuple[list[int], list[int]]] = []

        if not bar_positions:
            raise ValueError("No bar positions found")

        # Split at bar boundaries
        current_bar_idx = 0

        while current_bar_idx < len(bar_positions):
            # Find next split point
            split_bar_idx = find_split_bar_near_length(
                bar_positions, self.max_sequence_length, current_bar_idx
            )

            if split_bar_idx is None:
                break

            # Get positions for current segment
            start_input_pos = bar_positions[current_bar_idx][0]
            start_output_pos = bar_positions[current_bar_idx][1]

            # End positions
            if split_bar_idx < len(bar_positions):
                end_input_pos = bar_positions[split_bar_idx][0]
                end_output_pos = bar_positions[split_bar_idx][1]
            else:
                # Last segment: go to end
                end_input_pos = len(input_ids)
                end_output_pos = len(output_ids)

            # Extract segments
            input_seg = input_ids[start_input_pos:end_input_pos]
            output_seg = output_ids[start_output_pos:end_output_pos]

            # Only keep substantial segments and filter out those with UNK tokens
            if len(input_seg) > self.max_sequence_length // 10:
                if self.input_vocab.unk_id not in input_seg and self.output_vocab.unk_id not in output_seg:
                    segments.append((input_seg, output_seg))

            # Move to next bar (50% overlap for context)
            overlap_bars = (split_bar_idx - current_bar_idx) // 2
            current_bar_idx += max(1, overlap_bars)

        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a segment.

        Args:
            idx: Index

        Returns:
            Tuple of (input_ids, output_ids) as numpy arrays
            - input_ids: [input_seq_len] - Input sequence token IDs
            - output_ids: [output_seq_len] - Output sequence token IDs
            Note: input_seq_len != output_seq_len (output has extra TAB tokens)
        """
        input_ids, output_ids = self.segments[idx]
        return np.array(input_ids, dtype=np.int64), np.array(output_ids, dtype=np.int64)

    def get_vocab_sizes(self) -> tuple[int, int]:
        """Get vocabulary sizes."""
        return self.input_vocab.vocab_size, self.output_vocab.vocab_size

    def get_pad_ids(self) -> tuple[int, int]:
        """Get PAD token IDs."""
        return self.input_vocab.pad_id, self.output_vocab.pad_id


# ============================================================================
# Collate Function for DataLoader
# ============================================================================


class TabDatasetBatchInput(TypedDict):
    input_ids: torch.Tensor  # [B, L_enc]
    output_ids: torch.Tensor  # [B, L_dec]
    attention_mask: torch.Tensor  # [B, L_enc]
    decoder_attention_mask: torch.Tensor  # [B, L_dec]


def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray]],
    input_pad_id: int = 0,
    output_pad_id: int = 0,
) -> TabDatasetBatchInput:
    """
    Collate function for batching with padding.

    Args:
        batch: List of (input_ids, output_ids) tuples from __getitem__
               Each item: (input_ids[L_in], output_ids[L_out])

    Returns:
        Dictionary with padded tensors and attention masks:
        - input_ids: [batch_size, max_input_len] - Padded input sequences
        - output_ids: [batch_size, max_output_len] - Padded output sequences
        - attention_mask: [batch_size, max_input_len] - Input padding mask (1=real, 0=pad)
        - decoder_attention_mask: [batch_size, max_output_len] - Output padding mask (1=real, 0=pad)
    """
    input_ids_list = [item[0] for item in batch]
    output_ids_list = [item[1] for item in batch]

    # Find max lengths
    max_input_len = max(len(ids) for ids in input_ids_list)
    max_output_len = max(len(ids) for ids in output_ids_list)

    # Pad sequences
    batch_size = len(batch)
    padded_inputs = np.full((batch_size, max_input_len), input_pad_id, dtype=np.int64)
    padded_outputs = np.full(
        (batch_size, max_output_len), output_pad_id, dtype=np.int64
    )

    input_masks = np.zeros((batch_size, max_input_len), dtype=np.int64)
    output_masks = np.zeros((batch_size, max_output_len), dtype=np.int64)

    for i, (input_ids, output_ids) in enumerate(batch):
        input_len = len(input_ids)
        output_len = len(output_ids)

        padded_inputs[i, :input_len] = input_ids
        padded_outputs[i, :output_len] = output_ids

        input_masks[i, :input_len] = 1
        output_masks[i, :output_len] = 1

    return {
        "input_ids": torch.from_numpy(padded_inputs),
        "output_ids": torch.from_numpy(padded_outputs),
        "attention_mask": torch.from_numpy(input_masks),
        "decoder_attention_mask": torch.from_numpy(output_masks),
    }
