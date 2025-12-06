"""
Dataset for guitar tablature transcription using DadaGP tokens.

Loads .tokens.txt files, converts to our event format, tokenizes, and segments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict
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


def build_vocabulary(
    max_pitch: int = 127,
    max_time_shift: int = 500,
    num_strings: int = 6,
    num_frets: int = 21,
) -> tuple[Vocabulary, Vocabulary]:
    """
    Build vocabularies for input and output sequences.

    Args:
        max_pitch: Maximum MIDI pitch
        max_time_shift: Maximum time shift value
        num_strings: Number of guitar strings
        num_frets: Number of frets

    Returns:
        Tuple of (input_vocab, output_vocab)
    """
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

    # NOTE_OFF tokens
    for pitch in range(max_pitch + 1):
        token = f"NOTE_OFF_{pitch}"
        input_token_to_id[token] = idx
        input_id_to_token[idx] = token
        idx += 1

    # TIME_SHIFT tokens
    for shift in range(1, max_time_shift + 1):
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
    )

    # Build output vocabulary (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB)
    output_token_to_id: dict[str, int] = {}
    output_id_to_token: dict[int, str] = {}
    idx = 0

    # Special tokens
    for token in ["PAD", "BOS", "EOS", "UNK"]:
        output_token_to_id[token] = idx
        output_id_to_token[idx] = token
        idx += 1

    # NOTE_ON tokens
    for pitch in range(max_pitch + 1):
        token = f"NOTE_ON_{pitch}"
        output_token_to_id[token] = idx
        output_id_to_token[idx] = token
        idx += 1

    # NOTE_OFF tokens
    for pitch in range(max_pitch + 1):
        token = f"NOTE_OFF_{pitch}"
        output_token_to_id[token] = idx
        output_id_to_token[idx] = token
        idx += 1

    # TIME_SHIFT tokens
    for shift in range(1, max_time_shift + 1):
        token = f"TIME_SHIFT_{shift}"
        output_token_to_id[token] = idx
        output_id_to_token[idx] = token
        idx += 1

    # TAB tokens
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
        """
        self.token_files = token_files
        self.max_sequence_length = max_sequence_length

        # Build vocabularies
        print("Building vocabularies...")
        self.input_vocab, self.output_vocab = build_vocabulary(
            max_pitch=max_pitch,
            max_time_shift=max_time_shift,
            num_strings=num_strings,
            num_frets=num_frets,
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

        for token_file in tqdm(self.token_files, desc="Processing files"):
            try:
                # Parse DadaGP tokens
                dadagp_tokens = parse_dadagp_file(token_file)

                # Convert to events with bar positions
                input_events, output_events, bar_positions = dadagp_to_events(
                    dadagp_tokens
                )

                # Convert to IDs
                input_ids = events_to_ids(input_events, self.input_vocab)
                output_ids = events_to_ids(output_events, self.output_vocab)

                # Split into segments at bar boundaries
                segments = self._split_into_segments(
                    input_ids, output_ids, bar_positions
                )
                
                # insert BOS and EOS
                for inp, out in segments:
                    # insert BOS and EOS
                    out = [self.output_vocab.bos_id] + out + [self.output_vocab.eos_id]
                    
                    # only EOS
                    # out = out + [self.output_vocab.eos_id]
                    
                    # ensure max length
                    if len(out) > self.max_sequence_length:
                        out = out[:self.max_sequence_length - 1] + [self.output_vocab.eos_id]
                                    
                    all_segments.append((inp, out))
                
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

            # No bars found, return whole sequence if small enough
            if len(input_ids) <= self.max_sequence_length:
                return [(input_ids, output_ids)]
            else:
                return []

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

            # Only keep substantial segments
            if len(input_seg) > self.max_sequence_length // 10:
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

def collate_fn(batch, input_pad_id=0, output_pad_id=0):

    input_ids_list = [b[0] for b in batch]
    output_ids_list = [b[1] for b in batch]

    max_input_len = max(len(x) for x in input_ids_list)
    max_output_len = max(len(x) for x in output_ids_list)

    B = len(batch)

    padded_inputs = np.full((B, max_input_len), input_pad_id)
    padded_outputs = np.full((B, max_output_len), output_pad_id)

    input_masks = np.zeros((B, max_input_len))
    output_masks = np.zeros((B, max_output_len))

    for i, (inp, out) in enumerate(batch):
        padded_inputs[i, :len(inp)] = inp
        padded_outputs[i, :len(out)] = out
        input_masks[i, :len(inp)] = 1
        output_masks[i, :len(out)] = 1

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "attention_mask": torch.tensor(input_masks, dtype=torch.long),
        "output_ids": torch.tensor(padded_outputs, dtype=torch.long),
        "decoder_attention_mask": torch.tensor(output_masks, dtype=torch.long),
    }
