#!/usr/bin/env python3
"""
Prepare François Leduc dataset for training.

Converts MIDI files → DadaGP-format .tokens.txt and generates
train/val/test JSON split files.

# Modify 2026-03-16
MIDI format (François Leduc dataset):
  - Standard monophonic MIDI: ALL note_on events are on channel 0 only
    (NOT per-string channels as originally assumed)
  - Pitch-to-TAB heuristic: for each MIDI pitch, try strings 6→1 (high E
    first), pick the first string that yields a valid fret [0, MAX_FRET]
  - 220 ticks per beat

# Original (WRONG — Leduc MIDI uses channel 0 only, not per-string channels)
# MIDI format (François Leduc dataset):
#   - Per-string channels: channel C = string C+1 (channels 0-5)
#   - pitch = fret + STANDARD_TUNING[channel]
#   - 220 ticks per beat

Output .tokens.txt: DadaGP token format, ticks converted to 960 TPB equivalent.
JSON split files: full relative paths (no .tokens.txt suffix), used by src/dataloader.py.

Usage:
    python scripts/prepare_leduc_dataset.py
    python scripts/prepare_leduc_dataset.py --dataset-dir path/to/leduc
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import mido

# Standard guitar tuning: E2, A2, D3, G3, B3, E4
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

TGT_TPB = 960   # Target ticks per beat (DadaGP standard)
MAX_FRET = 20   # Max fret (matches TabDataset default num_frets=21, frets 0-20)
MAX_WAIT = 480  # Max single wait token value

GUITAR_TYPE_TO_INSTRUMENT = {
    "electric": "clean",
    "electric-band": "clean",
    "acoustic": "acoustic",
    "nylon": "acoustic",
}


def convert_ticks(ticks: int, src_tpb: int) -> int:
    """Convert ticks from source TPB to 960 TPB."""
    return round(ticks * TGT_TPB / src_tpb)


# Add 2026-03-16: pitch-to-TAB heuristic for monophonic MIDI
def pitch_to_tab(pitch: int) -> tuple[int, int] | None:
    """
    Map a MIDI pitch to (string, fret) on a standard-tuned 6-string guitar.

    Strategy: try strings 6→1 (high E first, low E last).
    Pick the first string that yields a valid fret in [0, MAX_FRET].
    High pitches land on high strings; low pitches fall back to low strings.

    Returns None if the pitch is outside the guitar's range entirely.
    """
    for s in range(6, 0, -1):  # string 6 (high E) down to string 1 (low E)
        fret = pitch - STANDARD_TUNING[s - 1]
        if 0 <= fret <= MAX_FRET:
            return (s, fret)
    return None


# Modify 2026-03-16: add transpose parameter for pitch transposition augmentation
# Original: def midi_to_tokens(midi_path: Path, artist: str, guitar_type: str) -> list[str]:
def midi_to_tokens(midi_path: Path, artist: str, guitar_type: str, transpose: int = 0) -> list[str]:
    """
    Convert a François Leduc MIDI file to DadaGP token format.

    Token order follows DadaGP convention: notes come BEFORE the wait
    that follows them (not before them).

    # Add 2026-03-16: transpose (int) shifts all pitches by N semitones before
    pitch-to-TAB mapping. Used for data augmentation. Default 0 = no change.

    Returns empty list if no valid notes are found.
    """
    mid = mido.MidiFile(str(midi_path))
    src_tpb = mid.ticks_per_beat  # typically 220 for this dataset

    # Extract tempo and time signature from track 0
    tempo_bpm = 120
    ts_num, ts_den = 4, 4
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo_bpm = max(30, min(300, round(mido.tempo2bpm(msg.tempo))))
        elif msg.type == "time_signature":
            ts_num, ts_den = msg.numerator, msg.denominator

    # Bar length in source ticks
    beats_per_bar = ts_num * (4.0 / ts_den)
    bar_src = int(beats_per_bar * src_tpb)

    # Modify 2026-03-16: parse note_on events using pitch-to-TAB heuristic
    # (Leduc MIDI uses channel 0 for ALL notes; channel-to-string mapping was wrong)
    note_track = mid.tracks[-1]
    abs_tick = 0
    # Original: tick_notes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    # tick_notes now stores (string, fret) instead of (channel, fret)
    tick_notes: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for msg in note_track:
        abs_tick += msg.time
        if not hasattr(msg, "channel"):
            continue
        # Original: ch = msg.channel
        # Original: if ch > 5 or ch == 10:  # skip drums (ch 10) and >6-string
        # Original:     continue
        # Original: if msg.type == "note_on" and msg.velocity > 0:
        # Original:     fret = msg.note - STANDARD_TUNING[ch]
        # Original:     if 0 <= fret <= MAX_FRET:
        # Original:         tick_notes[abs_tick].append((ch, fret))
        # Add 2026-03-16: use pitch-to-TAB heuristic instead of channel mapping
        if msg.type == "note_on" and msg.velocity > 0:
            # Modify 2026-03-16: apply transposition offset before mapping to TAB
            # Original: tab = pitch_to_tab(msg.note)
            tab = pitch_to_tab(msg.note + transpose)
            if tab is not None:
                tick_notes[abs_tick].append(tab)  # (string, fret)

    if not tick_notes:
        return []

    instr = GUITAR_TYPE_TO_INSTRUMENT.get(guitar_type, "clean")
    artist_tok = artist.replace(" ", "_").replace(",", "").replace(".", "")[:30]

    tokens = [
        f"artist:{artist_tok}",
        f"tempo:{tempo_bpm}",
        "downtune:0",
        "start",
    ]

    sorted_ticks = sorted(tick_notes.keys())
    next_bar_src = bar_src  # first bar boundary in source ticks

    # Modify 2026-03-16: split emit_gap into two parts so new_measure is emitted
    # BEFORE the note (after all wait tokens), not in the middle of the gap.
    # This matches DadaGP structure: [wait...] → new_measure → NOTE_ON
    # so segments starting at new_measure always begin with NOTE_ON (not TIME_SHIFT).
    # Original: emit_gap emitted wait-up-to-bar → new_measure → wait-remainder
    # which put wait tokens AFTER new_measure → segments started with TIME_SHIFT.

    def emit_wait_only(from_src: int, to_src: int) -> None:
        """Emit only wait tokens for the gap [from_src, to_src). No new_measure."""
        # Add 2026-03-16
        current_tgt = convert_ticks(from_src, src_tpb)
        to_tgt = convert_ticks(to_src, src_tpb)
        remaining = to_tgt - current_tgt
        while remaining > 0:
            chunk = min(remaining, MAX_WAIT)
            tokens.append(f"wait:{chunk}")
            remaining -= chunk

    def flush_bar_boundaries(up_to_src: int) -> None:
        """Emit new_measure tokens for all bar boundaries <= up_to_src."""
        # Add 2026-03-16
        nonlocal next_bar_src
        while next_bar_src <= up_to_src:
            tokens.append("new_measure")
            next_bar_src += bar_src

    # Handle initial silence before the first note:
    # emit all wait tokens first, then flush any bar boundaries, then note follows.
    first_tick = sorted_ticks[0]
    if first_tick > 0:
        emit_wait_only(0, first_tick)
    flush_bar_boundaries(first_tick)

    # Emit: notes at each tick, then (wait → new_measures) before the next tick
    for i, src_tick in enumerate(sorted_ticks):
        # Emit notes (sorted by string for deterministic output)
        # Modify 2026-03-16: tick_notes now stores (string, fret) directly
        # Original: for ch, fret in sorted(tick_notes[src_tick]):
        # Original:     tokens.append(f"{instr}:note:s{ch + 1}:f{fret}")
        for string, fret in sorted(tick_notes[src_tick]):
            tokens.append(f"{instr}:note:s{string}:f{fret}")

        # Emit wait then new_measure(s) before the next note
        # Modify 2026-03-16: was emit_gap(src_tick, next_tick) which interleaved
        # new_measure in the middle of the wait. Now: wait first, then new_measure.
        # Original: if i + 1 < len(sorted_ticks): emit_gap(src_tick, sorted_ticks[i + 1])
        if i + 1 < len(sorted_ticks):
            next_tick = sorted_ticks[i + 1]
            emit_wait_only(src_tick, next_tick)
            flush_bar_boundaries(next_tick)

    tokens.append("end")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare François Leduc dataset: MIDI → DadaGP tokens + JSON splits"
    )
    parser.add_argument(
        "--dataset-dir",
        default="10984521/FrancoisLeducDatasetPublication/FrancoisLeducDatasetPublication",
        help="Path to the dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project root directory (for computing relative paths in JSON files)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    project_dir = Path(args.project_dir).resolve()

    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")

    # Load metadata
    # Modify 2026-03-17: re-split all songs at 7:2:1 instead of using the
    # metadata.csv split column (which had only 62/8/9 → val too small).
    # Original:
    # split_data: dict[str, list[dict]] = {"train": [], "validate": [], "test": []}
    # with open(metadata_path, newline="", encoding="utf-8") as f:
    #     for row in csv.DictReader(f):
    #         s = row["split"]
    #         if s in split_data:
    #             split_data[s].append(row)

    # Add 2026-03-17: load ALL rows and re-split at 7:2:1
    import random
    all_rows: list[dict] = []
    with open(metadata_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rng = random.Random(42)  # fixed seed for reproducibility
    rng.shuffle(all_rows)

    n = len(all_rows)
    n_test = round(n * 0.1)
    n_val  = round(n * 0.2)
    n_train = n - n_val - n_test

    split_data: dict[str, list[dict]] = {
        "train":    all_rows[:n_train],
        "validate": all_rows[n_train : n_train + n_val],
        "test":     all_rows[n_train + n_val :],
    }

    total = sum(len(v) for v in split_data.values())
    print(
        f"Loaded {total} entries (7:2:1 re-split): "
        f"train={len(split_data['train'])}, "
        f"val={len(split_data['validate'])}, "
        f"test={len(split_data['test'])}"
    )
    print(f"Dataset dir:  {dataset_dir}")
    print(f"Project dir:  {project_dir}")
    print()

    # Add 2026-03-16: pitch transposition offsets for train augmentation
    # Only augment train split; val/test use only the original (offset=0)
    TRAIN_AUGMENT_OFFSETS = list(range(-5, 6))  # [-5, -4, ..., 0, ..., +5] = 11 versions

    # Convert MIDI → .tokens.txt
    errors: list[str] = []
    success = 0

    # Modify 2026-03-16: generate multiple transpositions for train split
    # Original: for split, rows in split_data.items():
    # Original:     for row in rows:
    # Original:         ...
    # Original:         tok = midi_to_tokens(midi_path, row["artist"], row["guitar_type"])
    # Original:         out_path.write_text(...)
    for split, rows in split_data.items():
        # Add 2026-03-16: train split gets all transpositions; val/test get only offset=0
        offsets = TRAIN_AUGMENT_OFFSETS if split == "train" else [0]
        for row in rows:
            midi_path = dataset_dir / row["midi_filename"]
            slice_id = row["slice_id"]

            for offset in offsets:
                # Add 2026-03-16: suffix for augmented files (empty string for original)
                suffix = f"_t{offset:+d}" if offset != 0 else ""
                out_path = dataset_dir / f"{slice_id}{suffix}.tokens.txt"

                try:
                    tok = midi_to_tokens(midi_path, row["artist"], row["guitar_type"], transpose=offset)
                    if tok:
                        out_path.write_text("\n".join(tok) + "\n", encoding="utf-8")
                        if offset == 0:
                            print(f"  [{split:8}] {slice_id}: {len(tok)} tokens")
                        success += 1
                    else:
                        if offset == 0:
                            msg = f"No valid notes: {midi_path.name}"
                            errors.append(msg)
                            print(f"  [EMPTY   ] {slice_id}: no valid notes found")
                except Exception as e:
                    msg = f"{midi_path.name} (t={offset:+d}): {e}"
                    errors.append(msg)
                    if offset == 0:
                        print(f"  [ERROR   ] {slice_id}: {e}")

    aug_count = len(TRAIN_AUGMENT_OFFSETS)
    print(f"\nConverted {success} files successfully (train x{aug_count} augmentations).")

    # Create JSON split files
    # Each entry is the full relative path to the .tokens.txt file WITHOUT the
    # .tokens.txt suffix — this matches the filtering logic in src/dataloader.py:
    #   gp_file = token_file[:-len('.tokens.txt')]
    #   if gp_file in selected_files: ...
    split_map = {
        "train": "train_files.json",
        "validate": "val_files.json",
        "test": "test_files.json",
    }

    # Modify 2026-03-16: include augmented files in train_files.json
    # Original: for split, json_name in split_map.items():
    # Original:     paths = []
    # Original:     for row in split_data[split]:
    # Original:         tokens_abs = dataset_dir / f"{row['slice_id']}.tokens.txt"
    # Original:         path_key = rel_str[: -len(".tokens.txt")]
    # Original:         paths.append(path_key)

    def collect_paths(rows: list[dict], offsets: list[int]) -> list[str]:
        """Collect relative path keys for all (row, offset) combinations that exist."""
        # Add 2026-03-17
        paths = []
        for row in rows:
            for offset in offsets:
                suffix = f"_t{offset:+d}" if offset != 0 else ""
                tokens_abs = dataset_dir / f"{row['slice_id']}{suffix}.tokens.txt"
                if not tokens_abs.exists():
                    continue
                rel_str = str(tokens_abs.relative_to(project_dir))
                paths.append(rel_str[: -len(".tokens.txt")])
        return sorted(paths)

    def write_json(json_name: str, paths: list[str]) -> None:
        """Write a sorted list of path keys to a JSON file."""
        # Add 2026-03-17
        json_path = dataset_dir / json_name
        json_path.write_text(json.dumps(paths, indent=2), encoding="utf-8")
        print(f"Wrote {json_name}: {len(paths)} files → {json_path}")

    # Modify 2026-03-17: produce both augmented and non-augmented train JSON files
    # so the user can choose which to use for training.
    # Original: single loop writing train_files.json with aug, val/test without
    write_json("train_files.json",        collect_paths(split_data["train"],    TRAIN_AUGMENT_OFFSETS))
    write_json("train_files_no_aug.json", collect_paths(split_data["train"],    [0]))
    write_json("val_files.json",          collect_paths(split_data["validate"], [0]))
    write_json("test_files.json",         collect_paths(split_data["test"],     [0]))

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")

    print(
        "\nDone! To train with this dataset:\n"
        "  python train.py data=leduc"
    )


if __name__ == "__main__":
    main()
