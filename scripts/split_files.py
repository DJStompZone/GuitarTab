#!/usr/bin/env python3
"""
Split selected_files.json into train/val/test splits at FILE level.

This ensures no file appears in multiple splits (no data leakage).
Run this ONCE before training.
"""

import json
import random
from pathlib import Path


def split_files(
    input_json: str = "selected_files.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = "data_splits"
):
    """
    Split selected files into train/val/test at FILE level.

    Args:
        input_json: Path to selected_files.json
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for testing (default 0.1)
        seed: Random seed for reproducibility
        output_dir: Directory to save split files
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    print("=" * 80)
    print("File Splitting for Train/Val/Test")
    print("=" * 80)

    # Load selected files
    print(f"\nLoading files from {input_json}...")
    with open(input_json, 'r') as f:
        all_files = json.load(f)

    print(f"Total files: {len(all_files):,}")

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled_files = all_files.copy()
    random.shuffle(shuffled_files)

    # Split
    n_train = int(len(shuffled_files) * train_ratio)
    n_val = int(len(shuffled_files) * val_ratio)

    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train:n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_files):,} files ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"  Val:   {len(val_files):,} files ({len(val_files)/len(all_files)*100:.1f}%)")
    print(f"  Test:  {len(test_files):,} files ({len(test_files)/len(all_files)*100:.1f}%)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save splits
    print(f"\nSaving splits to {output_dir}/...")

    train_path = output_path / "train_files.json"
    val_path = output_path / "val_files.json"
    test_path = output_path / "test_files.json"

    with open(train_path, 'w') as f:
        json.dump(train_files, f, indent=2)
    print(f"  ✓ {train_path}")

    with open(val_path, 'w') as f:
        json.dump(val_files, f, indent=2)
    print(f"  ✓ {val_path}")

    with open(test_path, 'w') as f:
        json.dump(test_files, f, indent=2)
    print(f"  ✓ {test_path}")

    # Verify no overlap
    print(f"\nVerifying no overlap...")
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap:
        print(f"  ❌ Train/Val overlap: {len(train_val_overlap)} files!")
        return False

    if train_test_overlap:
        print(f"  ❌ Train/Test overlap: {len(train_test_overlap)} files!")
        return False

    if val_test_overlap:
        print(f"  ❌ Val/Test overlap: {len(val_test_overlap)} files!")
        return False

    print(f"  ✓ No overlap between splits")

    # Verify all files accounted for
    total_split_files = len(train_files) + len(val_files) + len(test_files)
    if total_split_files != len(all_files):
        print(f"  ❌ File count mismatch: {total_split_files} != {len(all_files)}")
        return False

    print(f"  ✓ All {len(all_files):,} files accounted for")

    print("\n" + "=" * 80)
    print("✅ File splitting complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Use these splits in your training:")
    print("     python train.py data.selected_files_json=data_splits/train_files.json")
    print("  2. For inference, use test split:")
    print("     python inference.py data.selected_files_json=data_splits/test_files.json")
    print()

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split files into train/val/test")
    parser.add_argument("--input", default="selected_files.json", help="Input JSON file")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data_splits", help="Output directory")

    args = parser.parse_args()

    success = split_files(
        input_json=args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )

    exit(0 if success else 1)
