#!/usr/bin/env python3
"""Debug script to test mixed format detection and processing."""
import torch
from src.dataloader import create_dataset
from src.postprocessing_bridge import PostProcessingBridge
from fretting_postprocessor import GuitarConfig

# Load predictions
raw_preds = torch.load("outputs/2025-12-07_04-01-inference/raw_predictions.pt")

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

# Create bridge
guitar_config = GuitarConfig()
bridge = PostProcessingBridge(
    dataset.input_vocab,
    dataset.output_vocab,
    guitar_config
)

# Test first sequence
print("=== Testing Format Detection ===")
pred_ids = raw_preds[0][:50]  # First 50 tokens
pred_tokens = bridge.ids_to_token_strings(pred_ids, dataset.output_vocab)

print(f"\nFirst 20 tokens (post-processor format):")
for i, tok in enumerate(pred_tokens[:20]):
    print(f"  {i}: {tok}")

# Detect format
tab_count = sum(1 for t in pred_tokens if t.startswith('TAB<'))
note_on_count = sum(1 for t in pred_tokens if t.startswith('NOTE_ON<'))

print(f"\nToken counts: TAB={tab_count}, NOTE_ON={note_on_count}")

detected_format = bridge.processor._detect_output_format(pred_tokens)
print(f"Detected format: {detected_format}")

# Test processing
print("\n=== Testing Processing ===")
# Create dummy input (just for testing)
input_ids = torch.zeros_like(pred_ids)

try:
    result = bridge.processor.process_tokens(
        model_output_tokens=pred_tokens[:30],  # Limit for testing
        input_note_tokens=pred_tokens[:30],    # Use same for simplicity
        method='overlap',
        output_format='auto'
    )

    print(f"\nProcessed output (first 20 tokens):")
    for i, tok in enumerate(result[:20]):
        print(f"  {i}: {tok}")

    # Check output format
    out_tab = sum(1 for t in result if t.startswith('TAB<'))
    out_note = sum(1 for t in result if t.startswith('NOTE_ON<'))
    print(f"\nOutput counts: TAB={out_tab}, NOTE_ON={out_note}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
