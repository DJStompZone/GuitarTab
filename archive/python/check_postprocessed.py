#!/usr/bin/env python3
import torch
from src.dataloader import create_dataset

# Load predictions (use latest run)
raw_preds = torch.load("outputs/2025-12-07_04-33-inference/raw_predictions.pt")
post_preds = torch.load("outputs/2025-12-07_04-33-inference/postprocessed_predictions.pt")

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

print("First sequence comparison:")
print("\nRAW predictions (first 20 tokens):")
raw_tokens = [dataset.output_vocab.id_to_token[idx.item()] for idx in raw_preds[0][:20] if idx.item() != 0]
for i, tok in enumerate(raw_tokens):
    print(f"  {i}: {tok}")

print("\nPOST-PROCESSED predictions (first 20 tokens):")
post_tokens = [dataset.output_vocab.id_to_token.get(idx.item(), f"UNK({idx.item()})") for idx in post_preds[0][:20] if idx.item() != 0]
for i, tok in enumerate(post_tokens):
    print(f"  {i}: {tok}")

print(f"\nToken ID comparison:")
print(f"Raw  [0]: {raw_preds[0][0].item()} -> {dataset.output_vocab.id_to_token.get(raw_preds[0][0].item(), 'UNK')}")
print(f"Post [0]: {post_preds[0][0].item()} -> {dataset.output_vocab.id_to_token.get(post_preds[0][0].item(), 'UNK')}")
