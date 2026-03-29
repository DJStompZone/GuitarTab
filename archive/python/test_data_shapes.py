#!/usr/bin/env python3
"""
Test script to verify tensor shapes in the data pipeline only.
Does not require model dependencies.
"""

import numpy as np
from functools import partial
from torch.utils.data import DataLoader

from src.tab_dataset import TabDataset, collate_fn


def test_data_shapes():
    """Test and print tensor shapes in data pipeline."""

    print("=" * 80)
    print("Data Pipeline Tensor Shapes Verification")
    print("=" * 80)

    # 1. Dataset
    print("\n### 1. Dataset __getitem__")
    token_files = ["DadaGP-v1.1/M/M/M - La fleur (live).gp3.tokens.txt"]
    dataset = TabDataset(
        token_files=token_files,
        max_sequence_length=512,
        max_pitch=127,
        max_time_shift=500,
        num_strings=6,
        num_frets=21
    )

    input_ids, output_ids = dataset[0]
    print(f"  input_ids shape:  {input_ids.shape}  (dtype: {input_ids.dtype})")
    print(f"  output_ids shape: {output_ids.shape}  (dtype: {output_ids.dtype})")
    print(f"  Ratio: {len(output_ids) / len(input_ids):.2f}x (output has extra TAB tokens)")

    # 2. Collate function
    print("\n### 2. Collate Function (batching)")
    batch_items = [dataset[i] for i in range(min(4, len(dataset)))]
    print(f"  Batch size: {len(batch_items)}")
    print(f"\n  Individual sample lengths:")
    input_lens = []
    output_lens = []
    for i, (inp, out) in enumerate(batch_items):
        input_lens.append(len(inp))
        output_lens.append(len(out))
        print(f"    Sample {i}: input={len(inp):4d}, output={len(out):4d}, ratio={len(out)/len(inp):.2f}x")

    input_pad_id, output_pad_id = dataset.get_pad_ids()
    collate_fn_partial = partial(collate_fn, input_pad_id=input_pad_id, output_pad_id=output_pad_id)
    batch = collate_fn_partial(batch_items)

    print(f"\n  After collate_fn (padded to max in batch):")
    print(f"    input_ids:               {batch['input_ids'].shape}")
    print(f"    output_ids:              {batch['output_ids'].shape}")
    print(f"    attention_mask:          {batch['attention_mask'].shape}")
    print(f"    decoder_attention_mask:  {batch['decoder_attention_mask'].shape}")

    # Verify padding
    print(f"\n  Padding verification:")
    max_input = max(input_lens)
    max_output = max(output_lens)
    print(f"    Max input length in batch:  {max_input}")
    print(f"    Max output length in batch: {max_output}")
    print(f"    Expected input_ids shape:   ({len(batch_items)}, {max_input})")
    print(f"    Expected output_ids shape:  ({len(batch_items)}, {max_output})")
    assert batch['input_ids'].shape == (len(batch_items), max_input)
    assert batch['output_ids'].shape == (len(batch_items), max_output)
    print(f"    ✅ Shapes match!")

    # Check actual padding
    print(f"\n  Mask verification (sample 0):")
    sample_input_len = input_lens[0]
    sample_output_len = output_lens[0]
    input_mask = batch['attention_mask'][0]
    output_mask = batch['decoder_attention_mask'][0]

    # Convert to numpy for counting
    input_mask_np = input_mask.numpy()
    output_mask_np = output_mask.numpy()

    print(f"    Input:  {np.sum(input_mask_np == 1)} real tokens, {np.sum(input_mask_np == 0)} padding")
    print(f"    Output: {np.sum(output_mask_np == 1)} real tokens, {np.sum(output_mask_np == 0)} padding")
    assert np.sum(input_mask_np == 1) == sample_input_len
    assert np.sum(output_mask_np == 1) == sample_output_len
    print(f"    ✅ Masks are correct!")

    # 3. DataLoader
    print("\n### 3. DataLoader")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_partial, shuffle=False)
    batch = next(iter(dataloader))

    print(f"  Batch from DataLoader:")
    print(f"    input_ids:               {batch['input_ids'].shape}")
    print(f"    output_ids:              {batch['output_ids'].shape}")
    print(f"    attention_mask:          {batch['attention_mask'].shape}")
    print(f"    decoder_attention_mask:  {batch['decoder_attention_mask'].shape}")

    # 4. Vocabulary
    print("\n### 4. Vocabularies")
    input_vocab_size, output_vocab_size = dataset.get_vocab_sizes()
    print(f"  Input vocabulary size:   {input_vocab_size:4d} tokens")
    print(f"  Output vocabulary size:  {output_vocab_size:4d} tokens")
    print(f"  Difference:              {output_vocab_size - input_vocab_size:4d} tokens (126 TAB tokens)")

    print("\n" + "=" * 80)
    print("✅ All data pipeline shapes verified correctly!")
    print("=" * 80)

    # Summary table
    print("\n### Shape Flow Summary")
    print("─" * 80)
    print(f"{'Stage':<30} {'Input Shape':<25} {'Output Shape':<25}")
    print("─" * 80)
    print(f"{'Dataset __getitem__':<30} {str(input_ids.shape):<25} {str(output_ids.shape):<25}")
    print(f"{'Collate (batch=4)':<30} {str(batch['input_ids'].shape):<25} {str(batch['output_ids'].shape):<25}")
    print(f"{'After teacher forcing*':<30} {'[B, L_enc]':<25} {'[B, L_dec-1]':<25}")
    print(f"{'Model forward*':<30} {'[B, L_enc]':<25} {'[B, L_dec-1, V_out]':<25}")
    print("─" * 80)
    print("* Shapes after shifting and model forward (see TENSOR_SHAPES.md for details)")
    print()


if __name__ == "__main__":
    test_data_shapes()
