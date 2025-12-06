#!/usr/bin/env python3
"""
Test script for TabDataset.
"""

from glob import glob
from src.tab_dataset import TabDataset


def main():
    print("=" * 80)
    print("Testing TabDataset")
    print("=" * 80)

    # Find some token files
    token_files = glob("DadaGP-v1.1/**/*.tokens.txt", recursive=True)[:10]

    if not token_files:
        print("No .tokens.txt files found!")
        print("Please check the DadaGP-v1.1 directory.")
        return

    print(f"\nFound {len(token_files)} token files")
    print(f"Using first 10 for testing...")

    # Create dataset
    print(len(token_files))
    dataset = TabDataset(
        token_files=token_files,
        max_sequence_length=512
    )

    print(f"\nDataset created:")
    print(f"  Total segments: {len(dataset)}")
    print(f"  Input vocab size: {dataset.input_vocab.vocab_size}")
    print(f"  Output vocab size: {dataset.output_vocab.vocab_size}")

    if len(dataset) > 0:
        print("\nFirst segment:")
        input_ids, output_ids = dataset[0]
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {output_ids.shape}")
        print(f"  Input IDs (first 20): {input_ids[:20]}")
        print(f"  Output IDs (first 20): {output_ids[:20]}")

        # Decode first few tokens
        print("\n  Decoded input tokens (first 10):")
        for i in range(min(10, len(input_ids))):
            token = dataset.input_vocab.id_to_token.get(input_ids[i], "UNK")
            print(f"    {i}: {input_ids[i]:4d} -> {token}")

        print("\n  Decoded output tokens (first 10):")
        for i in range(min(10, len(output_ids))):
            token = dataset.output_vocab.id_to_token.get(output_ids[i], "UNK")
            print(f"    {i}: {output_ids[i]:4d} -> {token}")

    print("\n" + "=" * 80)
    print("✓ Dataset test complete!")


if __name__ == "__main__":
    main()
