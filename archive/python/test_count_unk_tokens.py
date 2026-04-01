#!/usr/bin/env python3
"""
Count UNK tokens after running the normal DadaGP -> Event -> ID pipeline.

Usage examples:

  # 單一檔案
  python scripts/count_unk_tokens.py path/to/song.tokens.txt

  # 掃描整個資料夾底下的所有 .tokens.txt
  python scripts/count_unk_tokens.py path/to/Dataset_dir

  # 多個路徑一起掃描
  python scripts/count_unk_tokens.py Dataset/train Dataset/valid/some_song.tokens.txt
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from src.tab_dataset import build_vocabulary, event_to_token_string
from src.dadagp_parser import parse_dadagp_file, dadagp_to_events
from omegaconf import OmegaConf


def iter_token_files_from_paths(paths: Iterable[str]) -> Iterable[Path]:
    """Yield all .tokens.txt files under given files/directories."""
    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix == ".txt" and path.name.endswith(".tokens.txt"):
                yield path
        elif path.is_dir():
            for sub in path.rglob("*.tokens.txt"):
                yield sub


def iter_token_files_from_split(
    data_dir: Path,
    token_pattern: str,
    selected_files_json: Optional[Path] = None,
    max_files: Optional[int] = None,
) -> list[Path]:
    """
    Reproduce the same file selection logic as training (see src/dataloader.create_dataset):
      - Glob all files under data_dir matching token_pattern
      - If selected_files_json is provided: filter by that split (list of original .gp paths, without .tokens.txt)
      - Otherwise: use all matching token files.
    """
    from glob import glob
    import json

    # Find all token files
    token_files = sorted(glob(str(data_dir / token_pattern), recursive=True))

    # If no split JSON: just use all token files
    if selected_files_json is None:
        paths = [Path(p) for p in token_files]
        if max_files is not None:
            paths = paths[:max_files]
        return paths

    # Load selected gp paths
    with selected_files_json.open("r") as f:
        selected_files = set(json.load(f))

    filtered: list[Path] = []
    for token_file in token_files:
        if token_file.endswith(".tokens.txt"):
            gp_file = token_file[: -len(".tokens.txt")]
            if gp_file in selected_files:
                filtered.append(Path(token_file))

    if max_files is not None:
        filtered = filtered[:max_files]

    return filtered


def count_unk_for_events(events, vocab):
    """
    Run the same event -> token_str -> id mapping as in training,
    and count how many times we hit vocab.unk_id.

    Returns:
        total_tokens, unk_tokens, Counter of unk by token_str
    """
    total = 0
    unk = 0
    unk_by_token = Counter()

    for ev in events:
        total += 1
        token_str = event_to_token_string(ev)
        token_id = vocab.token_to_id.get(token_str, vocab.unk_id)
        if token_id == vocab.unk_id:
            unk += 1
            unk_by_token[token_str] += 1

    return total, unk, unk_by_token

def count_unk_for_segments(segments, vocab):
    """
    Counts UNK statistics over the generated integer token-level segments

    Args:
        segments (list of sequence of ints): the segments

    Returns:
        total_segments, segments_with_unk, total_tokens, unk_tokens, unk_by_tokenID
    """
    total_segments = len(segments)
    segments_with_unk = 0

    total_tokens = 0
    unk_tokens = 0
    unk_by_tokenID = Counter()

    for segment in segments:
        has_unk = False
        for token_id in segment:
            total_tokens += 1
            if token_id == vocab.unk_id:
                unk_tokens += 1
                unk_by_tokenID[token_id] += 1
                has_unk = True
        
        if has_unk:
            segments_with_unk += 1

    return total_segments, segments_with_unk, total_tokens, unk_tokens, unk_by_tokenID


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count UNK tokens produced by the DadaGP -> events -> IDs pipeline.\n\n"
            "Typical usage (match training split):\n"
            "  python test_count_unk_tokens.py configs/data/train_split.yaml\n"
        )
    )
    parser.add_argument(
        "config",
        help="YAML data config (e.g., configs/data/train_split.yaml) containing data_dir, token_pattern, selected_files_json, etc.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Show top-K most frequent UNK-causing token strings (default: 20).",
    )

    args = parser.parse_args()

    # Load YAML config (same style as training data configs)
    cfg = OmegaConf.load(args.config)

    # Dataset / vocab hyperparameters (fall back to sensible defaults if missing)
    max_pitch = int(cfg.get("max_pitch", 127))
    max_time_shift = int(cfg.get("max_time_shift", 500))
    num_strings = int(cfg.get("num_strings", 6))
    num_frets = int(cfg.get("num_frets", 21))
    output_format = str(cfg.get("output_format", "v1"))
    max_files_cfg = cfg.get("max_files", None)
    max_files = int(max_files_cfg) if max_files_cfg is not None else None

    # Load Segment hyperparameters
    max_sequence_length = int(cfg.get("max_sequence_length", 512))

    data_dir = Path(cfg.get("data_dir", "."))
    token_pattern = str(cfg.get("token_pattern", "**/*.tokens.txt"))
    selected_files_json_str = cfg.get("selected_files_json", None)
    selected_files_json = (
        Path(selected_files_json_str) if selected_files_json_str is not None else None
    )
    print("selected_files_json =", selected_files_json)

    # Build vocabularies exactly as in training/demo
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=max_pitch,
        max_time_shift=max_time_shift,
        num_strings=num_strings,
        num_frets=num_frets,
        output_format=output_format,
    )

    # Determine which token files to use based on YAML config
    token_files = iter_token_files_from_split(
        data_dir=data_dir,
        token_pattern=token_pattern,
        selected_files_json=selected_files_json,
        max_files=max_files,
    )
    if not token_files:
        print("No .tokens.txt files found matching YAML data config.")
        return

    print(f"Found {len(token_files)} .tokens.txt files.")
    print(
        f"Vocab: max_pitch={max_pitch}, "
        f"max_time_shift={max_time_shift}, "
        f"num_strings={num_strings}, num_frets={num_frets}, "
        f"output_format={output_format}"
    )
    print("Using max sequence length (segments):", max_sequence_length)
    print()

    # Re-use TabDataset logic for segments mapping
    from src.tab_dataset import events_to_ids

    # Temporary dataset instance specifically just to re-use _split_into_segments
    from src.tab_dataset import TabDataset
    dummy_dataset = TabDataset(
        token_files=[], 
        max_sequence_length=max_sequence_length,
        max_pitch=max_pitch,
        max_time_shift=max_time_shift,
        num_strings=num_strings,
        num_frets=num_frets,
        output_format=output_format
    )


    # Global stats
    global_stats = {
        "input_total": 0,
        "input_unk": 0,
        "output_total": 0,
        "output_unk": 0,
        "input_seg_total": 0,
        "input_seg_unk": 0,
        "output_seg_total": 0,
        "output_seg_unk": 0,
    }
    
    global_unk_input = Counter()
    global_unk_output = Counter()

    per_file_stats = []

    for path in token_files:
        try:
            dadagp_tokens = parse_dadagp_file(str(path))
            input_events, output_events, bar_positions = dadagp_to_events(dadagp_tokens)
            
            # --- Overall statistics matching (pre-segmentation string counts)
            in_total_raw, in_unk_raw, in_unk_by_tok = count_unk_for_events(
                input_events, input_vocab
            )
            out_total_raw, out_unk_raw, out_unk_by_tok = count_unk_for_events(
                output_events, output_vocab
            )

            global_unk_input.update(in_unk_by_tok)
            global_unk_output.update(out_unk_by_tok)

            # --- Segment-level statistics matching
            input_ids = events_to_ids(input_events, input_vocab)
            output_ids = events_to_ids(output_events, output_vocab)

            segments = dummy_dataset._split_into_segments(
                input_ids, output_ids, bar_positions
            )
            
            in_segments = [s[0] for s in segments]
            out_segments = [s[1] for s in segments]

            in_seg_tot, in_seg_unk, in_tot, in_unk, _ = count_unk_for_segments(in_segments, input_vocab)
            out_seg_tot, out_seg_unk, out_tot, out_unk, _ = count_unk_for_segments(out_segments, output_vocab)

            global_stats["input_total"] += in_tot
            global_stats["input_unk"] += in_unk
            global_stats["output_total"] += out_tot
            global_stats["output_unk"] += out_unk

            global_stats["input_seg_total"] += in_seg_tot
            global_stats["input_seg_unk"] += in_seg_unk
            global_stats["output_seg_total"] += out_seg_tot
            global_stats["output_seg_unk"] += out_seg_unk

            per_file_stats.append(
                (path, in_seg_tot, in_seg_unk, out_seg_tot, out_seg_unk, in_tot, in_unk, out_tot, out_unk)
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")

    # Print per-file summary (short)
    print("Per-file Segment UNK summary (input -> output segments):")
    for path, iseg_tot, iseg_unk, oseg_tot, oseg_unk, in_total, in_unk, out_total, out_unk in per_file_stats[:50]:
        iseg_ratio = (iseg_unk / iseg_tot * 100.0) if iseg_tot > 0 else 0.0
        oseg_ratio = (oseg_unk / oseg_tot * 100.0) if oseg_tot > 0 else 0.0
        
        in_ratio = (in_unk / in_total * 100.0) if in_total > 0 else 0.0
        out_ratio = (out_unk / out_total * 100.0) if out_total > 0 else 0.0
        print(
            f"- {path}: \n"
            f"  [Seg] input {iseg_unk}/{iseg_tot} ({iseg_ratio:.3f}%), output {oseg_unk}/{oseg_tot} ({oseg_ratio:.3f}%)\n"
            f"  [Tok] input {in_unk}/{in_total} ({in_ratio:.3f}%), output {out_unk}/{out_total} ({out_ratio:.3f}%)"
        )
    if len(per_file_stats) > 50:
        print(f"... ({len(per_file_stats) - 50} more files omitted)")

    print()
    # Global summary
    in_seg_total = global_stats["input_seg_total"]
    in_seg_unk = global_stats["input_seg_unk"]
    out_seg_total = global_stats["output_seg_total"]
    out_seg_unk = global_stats["output_seg_unk"]

    in_total = global_stats["input_total"]
    in_unk = global_stats["input_unk"]
    out_total = global_stats["output_total"]
    out_unk = global_stats["output_unk"]


    in_seg_ratio = (in_seg_unk / in_seg_total * 100.0) if in_seg_total > 0 else 0.0
    out_seg_ratio = (out_seg_unk / out_seg_total * 100.0) if out_seg_total > 0 else 0.0

    in_ratio = (in_unk / in_total * 100.0) if in_total > 0 else 0.0
    out_ratio = (out_unk / out_total * 100.0) if out_total > 0 else 0.0

    print("Global UNK summary:")
    print(" === Segment Level Stats ===")
    print(
        f"- Input  segments: total={in_seg_total}, containing UNK={in_seg_unk} ({in_seg_ratio:.4f}%)"
    )
    print(
        f"- Output segments: total={out_seg_total}, containing UNK={out_seg_unk} ({out_seg_ratio:.4f}%)"
    )

    print("\n === Token Level Stats ===")
    print(
        f"- Input  tokens: total={in_total}, UNK={in_unk} ({in_ratio:.4f}%)"
    )
    print(
        f"- Output tokens: total={out_total}, UNK={out_unk} ({out_ratio:.4f}%)"
    )
    print()

    # Files with most UNK segments
    if per_file_stats:
        # sort by output_seg_unk/out_seg_tot asc then desc
        def sort_key(x):
            if x[3] == 0:
                return -1
            return x[4] / x[3]

        per_file_sorted = sorted(
            per_file_stats, key=lambda x: (sort_key(x), x[4]), reverse=True
        )
        top_n = min(10, len(per_file_sorted))
        print(f"Top {top_n} files by Target (Output) UNK Segments %:")
        for i in range(top_n):
            path, iseg_tot, iseg_unk, oseg_tot, oseg_unk, in_total, in_unk, out_total, out_unk = per_file_sorted[i]
            iseg_ratio = (iseg_unk / iseg_tot * 100.0) if iseg_tot > 0 else 0.0
            oseg_ratio = (oseg_unk / oseg_tot * 100.0) if oseg_tot > 0 else 0.0
            print(
                f"{i+1:2d}. {path}:\n"
                f"   Output  segments UNK {oseg_unk}/{oseg_tot} ({oseg_ratio:.3f}%)\n"
                f"   Input   segments UNK {iseg_unk}/{iseg_tot} ({iseg_ratio:.3f}%)"
            )
        print()

    # Top-K UNK-causing token strings
    def print_top(counter: Counter, title: str):
        if not counter:
            print(f"{title}: no UNK tokens.")
            return
        print(title)
        for token_str, cnt in counter.most_common(args.top_k):
            print(f"  {token_str}: {cnt}")
        print()

    print_top(global_unk_input, "Top UNK-causing token strings (INPUT sequence)")
    print_top(global_unk_output, "Top UNK-causing token strings (OUTPUT sequence)")


if __name__ == "__main__":
    main()


