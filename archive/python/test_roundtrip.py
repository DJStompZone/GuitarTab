#!/usr/bin/env python3
"""Test round-trip conversion: dataset tokens → post-processor → dataset tokens."""
import torch
from src.dataloader import create_dataset
from src.postprocessing_bridge import PostProcessingBridge
from fretting_postprocessor import GuitarConfig

# Load dataset and predictions
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

raw_preds = torch.load("outputs/2025-12-07_04-43-inference/raw_predictions.pt")

# Create bridge
guitar_config = GuitarConfig()
bridge = PostProcessingBridge(
    dataset.input_vocab,
    dataset.output_vocab,
    guitar_config
)

# Test first sequence - round trip without any post-processing
print("=== Round-Trip Test (No Post-Processing) ===\n")
pred_ids_orig = raw_preds[0][:30].clone()  # First 30 tokens

# Step 1: IDs → Tokens (dataset → post-processor format)
tokens_pp = bridge.ids_to_token_strings(pred_ids_orig, dataset.output_vocab)
print("Original (dataset IDs → post-processor tokens):")
for i, tok in enumerate(tokens_pp[:20]):
    print(f"  {i}: {tok}")

# Step 2: Parse mixed format (use empty input sequence for testing)
from fretting_postprocessor import FrettingPostProcessor
from fretting_postprocessor.sequence import NoteSequence
processor = FrettingPostProcessor(guitar_config)

# Create empty input sequence (just for parsing, not used in conversion test)
empty_input = NoteSequence([])

parsed = processor.parser.parse_mixed_format_output(
    tokens_pp,
    empty_input,
    guitar_config
)

print(f"\nParsed {len(parsed)} notes")

# Step 3: Serialize back to mixed format
from fretting_postprocessor.serializer import TokenSerializer
serializer = TokenSerializer()
tokens_reserialized = serializer.serialize_to_mixed_format(parsed)

print(f"\nReserialized ({len(tokens_reserialized)} tokens):")
for i, tok in enumerate(tokens_reserialized[:20]):
    print(f"  {i}: {tok}")

# Step 4: Tokens → IDs (post-processor → dataset format)
pred_ids_roundtrip = bridge.token_strings_to_ids(tokens_reserialized, dataset.output_vocab)

print(f"\nRound-trip IDs (first 20):")
for i in range(min(20, len(pred_ids_roundtrip))):
    orig_id = pred_ids_orig[i].item()
    rt_id = pred_ids_roundtrip[i].item()
    orig_tok = dataset.output_vocab.id_to_token.get(orig_id, 'UNK')
    rt_tok = dataset.output_vocab.id_to_token.get(rt_id, 'UNK')
    match = "✓" if orig_id == rt_id else "✗"
    print(f"  {i}: {orig_tok:20s} → {rt_tok:20s} {match}")

# Compute accuracy
matches = sum(1 for i in range(min(len(pred_ids_orig), len(pred_ids_roundtrip)))
              if pred_ids_orig[i].item() == pred_ids_roundtrip[i].item())
total = min(len(pred_ids_orig), len(pred_ids_roundtrip))
accuracy = matches / total * 100

print(f"\nRound-trip accuracy: {accuracy:.2f}% ({matches}/{total})")
print(f"Original length: {len(pred_ids_orig)}, Round-trip length: {len(pred_ids_roundtrip)}")
