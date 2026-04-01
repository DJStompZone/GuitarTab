#!/usr/bin/env python3
"""
Analyze how well DadaGP wait durations can be represented
by a beat-fraction TIME_SHIFT vocabulary using greedy decomposition.

用法（跟 test_count_unk_tokens.py 類似）:

  python test_timeshift_beat_fraction_coverage.py configs/data/train_split.yaml
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from omegaconf import OmegaConf

from src.dadagp_parser import parse_dadagp_file, WaitToken
from src.tab_dataset import TICKS_PER_BEAT, get_allowed_time_shift_ticks


def iter_token_files_from_split(
    data_dir: Path,
    token_pattern: str,
    selected_files_json: Optional[Path] = None,
    max_files: Optional[int] = None,
) -> list[Path]:
    """
    Copy of the selection logic used in test_count_unk_tokens.py:
      - Glob all files under data_dir matching token_pattern
      - If selected_files_json is provided: filter by that split
      - Otherwise: use all matching token files.
    """
    from glob import glob
    import json

    token_files = sorted(glob(str(data_dir / token_pattern), recursive=True))

    if selected_files_json is None:
        paths = [Path(p) for p in token_files]
        if max_files is not None:
            paths = paths[:max_files]
        return paths

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


def greedy_decompose_duration(
    total_ticks: int,
    allowed_ticks: list[int],
) -> tuple[list[int], int]:
    """
    Greedy decomposition of a duration (in ticks) into allowed TIME_SHIFT deltas.

    Returns:
        (list_of_deltas, error_ticks)
        where error_ticks = sum(list_of_deltas) - total_ticks
    """
    remaining = total_ticks
    used: list[int] = []

    # Greedy: big to small
    for t in sorted(allowed_ticks, reverse=True):
        if t <= 0:
            continue
        count, remaining = divmod(remaining, t)
        if count > 0:
            used.extend([t] * count)

    if remaining > 0:
        # Round the tail to the nearest single unit
        best_t = min(allowed_ticks, key=lambda x: abs(x - remaining))
        used.append(best_t)

    approx = sum(used)
    error = approx - total_ticks
    return used, error


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check how well DadaGP wait durations can be represented\n"
            "by a beat-fraction TIME_SHIFT vocabulary (greedy decomposition).\n\n"
            "Typical usage (match training split):\n"
            "  python test_timeshift_beat_fraction_coverage.py configs/data/train_split.yaml\n"
        )
    )
    parser.add_argument(
        "config",
        help="YAML data config (e.g., configs/data/train_split.yaml) containing data_dir, token_pattern, selected_files_json, etc.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files to scan (for quick experiments).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Hyper-params just to know the intended max_time_shift; representation
    # itself only depends on allowed beat fractions & TICKS_PER_BEAT.
    max_time_shift = int(cfg.get("max_time_shift", 500))

    data_dir = Path(cfg.get("data_dir", "."))
    token_pattern = str(cfg.get("token_pattern", "**/*.tokens.txt"))
    selected_files_json_str = cfg.get("selected_files_json", None)
    selected_files_json = (
        Path(selected_files_json_str) if selected_files_json_str is not None else None
    )

    max_files_cfg = cfg.get("max_files", None)
    max_files_cfg_int = int(max_files_cfg) if max_files_cfg is not None else None

    # If user passes --max-files on CLI, it overrides YAML max_files
    effective_max_files = args.max_files if args.max_files is not None else max_files_cfg_int

    token_files = iter_token_files_from_split(
        data_dir=data_dir,
        token_pattern=token_pattern,
        selected_files_json=selected_files_json,
        max_files=effective_max_files,
    )
    if not token_files:
        print("No .tokens.txt files found matching YAML data config.")
        return

    # Beat-fraction TIME_SHIFT schema (still expressed in ticks)
    allowed_ticks = get_allowed_time_shift_ticks(
        max_time_shift=max_time_shift,
        time_shift_mode="beat_fraction",
    )

    print(f"Scanning {len(token_files)} .tokens.txt files.")
    print(f"TICKS_PER_BEAT = {TICKS_PER_BEAT}")
    print(f"Configured max_time_shift = {max_time_shift}")
    print(f"Beat-fraction TIME_SHIFT deltas (ticks): {allowed_ticks}")
    print()

    # Global stats (count "coalesced" wait blocks, not raw wait tokens)
    total_wait_blocks = 0
    total_raw_waits = 0
    exact_representable = 0
    non_exact = 0

    error_counter = Counter()  # error_ticks -> count
    abs_error_counter = Counter()  # abs(error_ticks) -> count

    per_file_max_abs_error: list[tuple[Path, int]] = []

    for path in token_files:
        tokens = parse_dadagp_file(str(path))

        file_raw_waits = 0
        file_max_abs_error = 0

        # Coalesce consecutive waits (same as dadagp_parser.dadagp_to_events)
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not isinstance(tok, WaitToken):
                i += 1
                continue
            coalesced = tok.ticks
            num_raw_in_block = 1
            i += 1
            while i < len(tokens) and isinstance(tokens[i], WaitToken):
                coalesced += tokens[i].ticks
                num_raw_in_block += 1
                i += 1

            file_raw_waits += num_raw_in_block
            total_wait_blocks += 1
            total_raw_waits += num_raw_in_block

            deltas, err = greedy_decompose_duration(coalesced, allowed_ticks)
            if err == 0:
                exact_representable += 1
            else:
                non_exact += 1
                error_counter[err] += 1
                abs_err = abs(err)
                abs_error_counter[abs_err] += 1
                if abs_err > file_max_abs_error:
                    file_max_abs_error = abs_err

        if file_raw_waits > 0:
            per_file_max_abs_error.append((path, file_max_abs_error))

    print("=== Global Coverage Statistics (consecutive waits coalesced) ===")
    print(f"Total wait blocks (after coalescing): {total_wait_blocks}")
    print(f"Total raw wait tokens: {total_raw_waits}")
    print(f"Exactly representable (error = 0): {exact_representable}")
    print(f"Non-exact (need rounding): {non_exact}")
    if total_wait_blocks > 0:
        exact_ratio = exact_representable / total_wait_blocks * 100.0
        print(f"Exact coverage: {exact_ratio:.4f}%")
    print()

    if non_exact > 0:
        # Error distribution in ticks
        print("Top absolute timing errors (in ticks):")
        for abs_err, cnt in abs_error_counter.most_common(10):
            beats = abs_err / TICKS_PER_BEAT
            print(f"  |error| = {abs_err:4d} ticks (~{beats:.5f} beats): count = {cnt}")
        print()

        # Per-file worst case
        print("Top files by max absolute timing error (ticks):")
        per_file_sorted = sorted(
            per_file_max_abs_error,
            key=lambda x: x[1],
            reverse=True,
        )
        top_n = min(10, len(per_file_sorted))
        for i in range(top_n):
            path, max_abs_err = per_file_sorted[i]
            beats = max_abs_err / TICKS_PER_BEAT
            print(f"{i+1:2d}. {path}: max |error| = {max_abs_err} ticks (~{beats:.5f} beats)")


if __name__ == "__main__":
    main()

