#!/usr/bin/env python3
"""
Visualize model-predicted guitar tablature as a rendered image.

Usage:
    python visualize_song.py --output-dir outputs/inference_folder --segment 0 --show both
    python visualize_song.py --output-dir outputs/inference_folder --segment 0 --show pred --out my_tab.png
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─────────────────────────────────────────────────────────────────────────────
# Reuse FTEvent and decode_tokens_to_events from inference.py
# ─────────────────────────────────────────────────────────────────────────────

class FTEvent:
    def __init__(self, type, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)


def decode_tokens_to_events(tokens):
    """Convert Fretting-Transformer token strings into FTEvent list."""
    events = []
    for t in tokens:
        if t.startswith("NOTE_ON_"):
            pitch = int(t.split("_")[2])
            events.append(FTEvent("NOTE_ON", pitch=pitch))
        elif t.startswith("NOTE_OFF_"):
            pitch = int(t.split("_")[2])
            events.append(FTEvent("NOTE_OFF", pitch=pitch))
        elif t.startswith("TIME_SHIFT_"):
            delta = int(t.split("_")[2])
            events.append(FTEvent("TIME_SHIFT", delta=delta))
        elif t.startswith("TAB_"):
            parts = t.split("_")
            events.append(FTEvent("TAB", string=int(parts[1]), fret=int(parts[2])))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Core rendering logic
# ─────────────────────────────────────────────────────────────────────────────

STRING_NAMES = ["e", "B", "G", "D", "A", "E"]   # string 1→6 (top to bottom)
# Y index: string 1 = top line (y=5), string 6 = bottom line (y=0)
STRING_Y = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}


def events_to_timeline(events, ticks_per_beat=480):
    """
    Parse FTEvent list into per-string note timeline.

    Returns:
        timeline: dict {string_num: [(tick, fret), ...]}
        total_ticks: int
    """
    timeline = defaultdict(list)
    current_tick = 0
    for ev in events:
        if ev.type == "TIME_SHIFT":
            current_tick += ev.delta
        elif ev.type == "TAB":
            timeline[ev.string].append((current_tick, ev.fret))
    return timeline, current_tick


def render_tab_axes(ax, timeline, total_ticks,
                    ticks_per_beat=480, bars_per_row=4,
                    row_index=0, row_bar_start=0, num_bars_in_row=None):
    """
    Draw one row of TAB on a given matplotlib Axes.

    Args:
        ax: matplotlib Axes
        timeline: {string: [(tick, fret)]}
        total_ticks: total duration in ticks
        ticks_per_beat: MIDI ticks per beat
        bars_per_row: bars displayed per row
        row_index: which row (for tick offset calculation)
        row_bar_start: first bar number (0-indexed) for this row
        num_bars_in_row: how many bars to draw
    """
    ticks_per_bar = ticks_per_beat * 4          # assume 4/4
    bar_width = 1.0                              # normalized: 1 unit = 1 bar
    total_width = bar_width * num_bars_in_row

    # Draw 6 horizontal string lines
    for string_num in range(1, 7):
        y = STRING_Y[string_num]
        ax.hlines(y, 0, total_width, colors="black", linewidths=0.8)

    # Draw bar separator lines (vertical)
    for b in range(num_bars_in_row + 1):
        ax.vlines(b * bar_width, -0.4, 5.4, colors="black", linewidths=0.8)

    # String labels on the left
    for string_num in range(1, 7):
        y = STRING_Y[string_num]
        ax.text(-0.05, y, STRING_NAMES[string_num - 1],
                ha="right", va="center", fontsize=7, fontweight="bold")

    # Bar number labels at the bottom
    for b in range(num_bars_in_row):
        bar_num = row_bar_start + b + 1
        ax.text((b + 0.5) * bar_width, -0.7, f"{bar_num}",
                ha="center", va="top", fontsize=6, color="gray")

    # Place fret numbers
    for string_num, notes in timeline.items():
        y = STRING_Y.get(string_num)
        if y is None:
            continue
        for tick, fret in notes:
            bar_of_note = tick // ticks_per_bar
            bar_in_row = bar_of_note - row_bar_start
            if bar_in_row < 0 or bar_in_row >= num_bars_in_row:
                continue
            tick_in_bar = tick % ticks_per_bar
            x = bar_in_row * bar_width + (tick_in_bar / ticks_per_bar) * bar_width
            ax.text(x, y, str(fret),
                    ha="center", va="center", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    ax.set_xlim(-0.12, total_width + 0.02)
    ax.set_ylim(-1.0, 6.0)
    ax.axis("off")


def render_tab_image(events, title="Guitar Tablature",
                     ticks_per_beat=480, bars_per_row=4,
                     output_file="tab_visualization.png"):
    """
    Render TAB events as a full-song guitar tablature image.

    Args:
        events: list of FTEvent
        title: figure title
        ticks_per_beat: MIDI ticks per quarter note (default 480)
        bars_per_row: number of bars per row
        output_file: path to save PNG
    """
    timeline, total_ticks = events_to_timeline(events, ticks_per_beat)

    ticks_per_bar = ticks_per_beat * 4
    if total_ticks == 0:
        print(f"  [warn] No TAB events found for '{title}', skipping.")
        return

    total_bars = max(1, (total_ticks + ticks_per_bar - 1) // ticks_per_bar)
    num_rows = (total_bars + bars_per_row - 1) // bars_per_row

    # Figure size: width scales with bars_per_row, height scales with rows
    fig_w = max(8, bars_per_row * 2.5)
    fig_h = max(3, num_rows * 1.8 + 1.0)
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_w, fig_h),
                             gridspec_kw={"hspace": 0.8})
    if num_rows == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.0)

    for row_idx in range(num_rows):
        row_bar_start = row_idx * bars_per_row
        num_bars_in_row = min(bars_per_row, total_bars - row_bar_start)
        ax = axes[row_idx]
        render_tab_axes(
            ax, timeline, total_ticks,
            ticks_per_beat=ticks_per_beat,
            bars_per_row=bars_per_row,
            row_index=row_idx,
            row_bar_start=row_bar_start,
            num_bars_in_row=num_bars_in_row
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def ids_to_events(token_ids, output_vocab):
    """Convert a 1D tensor of token IDs to FTEvent list."""
    PAD_ID = 0
    EOS_ID = 2
    tokens = []
    for t in token_ids.tolist():
        if t in (PAD_ID, EOS_ID):
            continue
        tok_str = output_vocab.id_to_token.get(t)
        if tok_str:
            tokens.append(tok_str)
    return decode_tokens_to_events(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model-predicted guitar tab as an image."
    )
    parser.add_argument("--output-dir", default="outputs/inference_folder",
                        help="Directory containing predictions.pt and targets.pt")
    parser.add_argument("--segment", type=int, default=0,
                        help="Index of segment to visualize (0-based)")
    parser.add_argument("--show", choices=["pred", "gt", "both"], default="both",
                        help="Which tab to render: prediction, ground truth, or both")
    parser.add_argument("--bars-per-row", type=int, default=4,
                        help="Number of bars per row in the output image")
    parser.add_argument("--ticks-per-beat", type=int, default=480,
                        help="MIDI ticks per beat (default 480)")
    parser.add_argument("--out", default=None,
                        help="Output PNG filename (default: tab_<show>_seg<N>.png in output-dir)")
    # Vocab loading parameters
    parser.add_argument("--data-dir", default="DadaGP-v1.1",
                        help="Path to DadaGP data directory")
    parser.add_argument("--train-split", default="data_splits/train_files.json",
                        help="Path to train split JSON (for vocab)")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-pitch", type=int, default=127)
    parser.add_argument("--max-time-shift", type=int, default=500)
    parser.add_argument("--num-strings", type=int, default=6)
    parser.add_argument("--num-frets", type=int, default=21)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pred_path = output_dir / "predictions.pt"
    tgt_path  = output_dir / "targets.pt"

    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found. Run inference.py first.")
        sys.exit(1)

    print(f"Loading predictions from {pred_path}")
    predictions = torch.load(pred_path, map_location="cpu")   # [N, L]
    targets     = torch.load(tgt_path,  map_location="cpu") if tgt_path.exists() else None

    n_segments = predictions.shape[0]
    print(f"Total segments: {n_segments}")

    # Load segment→source file mapping (saved by inference.py)
    sources_path = output_dir / "segment_sources.json"
    segment_sources = None
    if sources_path.exists():
        import json
        with open(sources_path) as f:
            segment_sources = json.load(f)

    def get_song_title(seg_idx):
        if segment_sources is None or seg_idx >= len(segment_sources):
            return f"Segment {seg_idx}"
        src = segment_sources[seg_idx]
        try:
            with open(src) as f:
                first_line = f.readline().strip()
            if first_line.startswith("artist:"):
                return first_line[len("artist:"):].replace("_", " ")
        except Exception:
            pass
        return Path(src).stem

    if args.segment >= n_segments:
        print(f"ERROR: --segment {args.segment} out of range (0–{n_segments-1})")
        sys.exit(1)

    # Load vocab
    print(f"Loading vocab from train split: {args.train_split}")
    sys.path.insert(0, str(Path(__file__).parent))
    from src.dataloader import create_dataset
    train_dataset = create_dataset(
        data_dir=args.data_dir,
        token_pattern="**/*.tokens.txt",
        selected_files_json=args.train_split,
        max_sequence_length=args.max_sequence_length,
        max_pitch=args.max_pitch,
        max_time_shift=args.max_time_shift,
        num_strings=args.num_strings,
        num_frets=args.num_frets,
        max_files=1  # only need vocab, load minimal data
    )
    output_vocab = train_dataset.output_vocab

    seg_idx = args.segment

    # Determine output path
    def out_path(label):
        if args.out:
            stem = Path(args.out).stem
            suffix = Path(args.out).suffix or ".png"
            name = f"{stem}_{label}{suffix}" if args.show == "both" else args.out
        else:
            name = f"tab_{label}_seg{seg_idx}.png"
        return output_dir / name

    song_title = get_song_title(seg_idx)

    if args.show in ("pred", "both"):
        pred_events = ids_to_events(predictions[seg_idx], output_vocab)
        n_tab = sum(1 for e in pred_events if e.type == "TAB")
        print(f"Segment {seg_idx} prediction: {len(pred_events)} events, {n_tab} TAB tokens")
        render_tab_image(
            pred_events,
            title=f"Predicted Tab — {song_title} (seg {seg_idx})",
            ticks_per_beat=args.ticks_per_beat,
            bars_per_row=args.bars_per_row,
            output_file=str(out_path("pred"))
        )

    if args.show in ("gt", "both") and targets is not None:
        gt_events = ids_to_events(targets[seg_idx], output_vocab)
        n_tab = sum(1 for e in gt_events if e.type == "TAB")
        print(f"Segment {seg_idx} ground truth: {len(gt_events)} events, {n_tab} TAB tokens")
        render_tab_image(
            gt_events,
            title=f"Ground Truth — {song_title} (seg {seg_idx})",
            ticks_per_beat=args.ticks_per_beat,
            bars_per_row=args.bars_per_row,
            output_file=str(out_path("gt"))
        )

    print("Done.")


if __name__ == "__main__":
    main()
