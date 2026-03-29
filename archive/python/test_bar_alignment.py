#!/usr/bin/env python3
"""
Test that bar-aligned splitting maintains semantic alignment between input and output.
"""

from glob import glob
from src.tab_dataset import TabDataset


def main():
    print("=" * 80)
    print("Testing Bar-Aligned Splitting")
    print("=" * 80)

    # Find token files
    token_files = glob("DadaGP-v1.1/**/*.tokens.txt", recursive=True)[:5]

    if not token_files:
        print("No .tokens.txt files found!")
        return

    print(f"\nUsing {len(token_files)} files for testing...")

    # Create dataset
    dataset = TabDataset(
        token_files=token_files,
        max_sequence_length=512
    )

    print(f"\nDataset: {len(dataset)} segments")

    # Check first few segments
    print("\nSegment length statistics:")
    print("-" * 80)
    print(f"{'Segment':<10} {'Input Len':<12} {'Output Len':<12} {'Difference':<12}")
    print("-" * 80)

    for i in range(min(10, len(dataset))):
        input_ids, output_ids = dataset[i]
        diff = len(output_ids) - len(input_ids)
        print(f"{i:<10} {len(input_ids):<12} {len(output_ids):<12} {diff:<12}")

    # Verify semantic alignment
    print("\n" + "=" * 80)
    print("Verifying Semantic Alignment (Segment 0)")
    print("=" * 80)

    input_ids, output_ids = dataset[0]

    # Count event types in both sequences
    def count_event_types(ids, vocab):
        counts = {'NOTE_ON': 0, 'NOTE_OFF': 0, 'TIME_SHIFT': 0, 'TAB': 0}
        for token_id in ids:
            token = vocab.id_to_token.get(token_id, "UNK")
            if token.startswith('NOTE_ON_'):
                counts['NOTE_ON'] += 1
            elif token.startswith('NOTE_OFF_'):
                counts['NOTE_OFF'] += 1
            elif token.startswith('TIME_SHIFT_'):
                counts['TIME_SHIFT'] += 1
            elif token.startswith('TAB_'):
                counts['TAB'] += 1
        return counts

    input_counts = count_event_types(input_ids, dataset.input_vocab)
    output_counts = count_event_types(output_ids, dataset.output_vocab)

    print("\nEvent counts:")
    print(f"  Input:  NOTE_ON={input_counts['NOTE_ON']}, NOTE_OFF={input_counts['NOTE_OFF']}, TIME_SHIFT={input_counts['TIME_SHIFT']}")
    print(f"  Output: NOTE_ON={output_counts['NOTE_ON']}, NOTE_OFF={output_counts['NOTE_OFF']}, TIME_SHIFT={output_counts['TIME_SHIFT']}, TAB={output_counts['TAB']}")

    # Verify alignment
    print("\nAlignment checks:")
    if input_counts['NOTE_ON'] == output_counts['NOTE_ON']:
        print("  ✓ Same number of NOTE_ON events")
    else:
        print(f"  ✗ NOTE_ON mismatch: {input_counts['NOTE_ON']} vs {output_counts['NOTE_ON']}")

    if input_counts['NOTE_OFF'] == output_counts['NOTE_OFF']:
        print("  ✓ Same number of NOTE_OFF events")
    else:
        print(f"  ✗ NOTE_OFF mismatch: {input_counts['NOTE_OFF']} vs {output_counts['NOTE_OFF']}")

    if input_counts['TIME_SHIFT'] == output_counts['TIME_SHIFT']:
        print("  ✓ Same number of TIME_SHIFT events")
    else:
        print(f"  ✗ TIME_SHIFT mismatch: {input_counts['TIME_SHIFT']} vs {output_counts['TIME_SHIFT']}")

    if output_counts['TAB'] == output_counts['NOTE_ON']:
        print("  ✓ One TAB token per NOTE_ON (correct format)")
    else:
        print(f"  ✗ TAB/NOTE_ON mismatch: {output_counts['TAB']} TABs vs {output_counts['NOTE_ON']} NOTE_ONs")

    print("\n" + "=" * 80)
    print("✓ Bar-aligned splitting test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
