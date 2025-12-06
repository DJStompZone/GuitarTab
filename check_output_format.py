#!/usr/bin/env python3
import torch
from src.dataloader import create_dataset

# Load predictions
preds = torch.load("outputs/2025-12-07_03-56-inference/raw_predictions.pt")
dataset = create_dataset(
    data_dir="DadaGP-v1.1",
    token_pattern="**/*.tokens.txt",
    selected_files_json="data_splits/mini_test_files.json",
    max_sequence_length=512,
    max_pitch=127,
    max_time_shift=500,
    num_strings=6,
    num_frets=21
)

# Check first sequence
first_seq = preds[0]
tokens = [dataset.output_vocab.id_to_token[idx.item()] for idx in first_seq[:50] if idx.item() != 0]
print("First 50 non-PAD tokens from model output:")
for i, tok in enumerate(tokens[:20]):
    print(f"{i}: {tok}")

print(f"\nIn first 50 tokens:")
print("TAB count:", sum(1 for t in tokens if "TAB_" in t))
print("NOTE_ON count:", sum(1 for t in tokens if "NOTE_ON_" in t))
print("NOTE_OFF count:", sum(1 for t in tokens if "NOTE_OFF_" in t))
