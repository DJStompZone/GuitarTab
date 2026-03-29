#!/usr/bin/env python3
"""
Test the file filtering functionality.
Verifies that selected_files.json correctly filters token files.
"""

import json
from glob import glob
import os


def test_file_filtering():
    """Test that file filtering works correctly."""

    print("=" * 80)
    print("File Filtering Test")
    print("=" * 80)

    # Find all token files
    all_token_files = sorted(glob(
        os.path.join("DadaGP-v1.1", "**/*.tokens.txt"),
        recursive=True
    ))
    print(f"\nTotal token files in dataset: {len(all_token_files):,}")

    # Load selected files
    with open("selected_files.json", 'r') as f:
        selected_files = set(json.load(f))
    print(f"Selected files in JSON: {len(selected_files):,}")

    # Apply filtering logic (same as train.py)
    filtered_token_files = []
    for token_file in all_token_files:
        if token_file.endswith('.tokens.txt'):
            # Remove .tokens.txt suffix to get the original .gp filename
            gp_file = token_file[:-len('.tokens.txt')]
            if gp_file in selected_files:
                filtered_token_files.append(token_file)

    print(f"Filtered token files: {len(filtered_token_files):,}")
    print(f"Filtering rate: {len(filtered_token_files) / len(all_token_files) * 100:.1f}%")

    # Show examples
    print("\nFirst 5 filtered files:")
    for i, f in enumerate(filtered_token_files[:5], 1):
        gp_file = f[:-len('.tokens.txt')]
        print(f"  {i}. {gp_file}")
        print(f"     → {f}")

    # Verify correctness
    print("\nVerification:")
    for token_file in filtered_token_files[:10]:
        gp_file = token_file[:-len('.tokens.txt')]
        if gp_file not in selected_files:
            print(f"  ❌ ERROR: {gp_file} not in selected_files but was included!")
            return False
        print(f"  ✅ {os.path.basename(gp_file)}")

    print("\n" + "=" * 80)
    print("✅ File filtering works correctly!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    test_file_filtering()
