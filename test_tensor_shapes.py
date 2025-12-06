#!/usr/bin/env python3
"""
Test script to verify tensor shapes throughout the pipeline.
Traces shapes from dataset to model forward pass.
"""

import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

from src.tab_dataset import TabDataset, collate_fn
from src.model import FrettingTransformer
from omegaconf import DictConfig


def test_tensor_shapes():
    """Test and print tensor shapes at each stage."""

    print("=" * 80)
    print("Tensor Shapes Verification")
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
    print(f"  Note: output is longer due to extra TAB tokens")

    # 2. Collate function
    print("\n### 2. Collate Function (batching)")
    batch_items = [dataset[i] for i in range(min(3, len(dataset)))]
    print(f"  Batch size: {len(batch_items)}")
    print(f"  Individual lengths:")
    for i, (inp, out) in enumerate(batch_items):
        print(f"    Sample {i}: input={len(inp)}, output={len(out)}")

    input_pad_id, output_pad_id = dataset.get_pad_ids()
    collate_fn_partial = partial(collate_fn, input_pad_id=input_pad_id, output_pad_id=output_pad_id)
    batch = collate_fn_partial(batch_items)

    print(f"\n  After collate_fn:")
    print(f"    input_ids:               {batch['input_ids'].shape}  (padded to max input len)")
    print(f"    output_ids:              {batch['output_ids'].shape}  (padded to max output len)")
    print(f"    attention_mask:          {batch['attention_mask'].shape}")
    print(f"    decoder_attention_mask:  {batch['decoder_attention_mask'].shape}")

    # 3. DataLoader
    print("\n### 3. DataLoader")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_partial)
    batch = next(iter(dataloader))

    print(f"  input_ids:               {batch['input_ids'].shape}")
    print(f"  output_ids:              {batch['output_ids'].shape}")
    print(f"  attention_mask:          {batch['attention_mask'].shape}")
    print(f"  decoder_attention_mask:  {batch['decoder_attention_mask'].shape}")

    # 4. Training loop (move to device and shift)
    print("\n### 4. Training Loop (after shifting)")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_ids = torch.from_numpy(batch["input_ids"]).to(device)
    attention_mask = torch.from_numpy(batch["attention_mask"]).to(device)
    labels = torch.from_numpy(batch["output_ids"]).to(device)
    decoder_attention_mask = torch.from_numpy(batch["decoder_attention_mask"]).to(device)

    print(f"  Before shift:")
    print(f"    input_ids:               {input_ids.shape}")
    print(f"    labels:                  {labels.shape}")

    # Shift for teacher forcing
    decoder_input_ids = labels[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()

    print(f"\n  After shift (teacher forcing):")
    print(f"    decoder_input_ids:       {decoder_input_ids.shape}  (removed last token)")
    print(f"    labels:                  {labels.shape}  (removed first token)")
    print(f"    decoder_attention_mask:  {decoder_attention_mask.shape}")

    # 5. Model forward pass
    print("\n### 5. Model Forward Pass")
    input_vocab_size, output_vocab_size = dataset.get_vocab_sizes()
    print(f"  Vocabulary sizes: input={input_vocab_size}, output={output_vocab_size}")

    model_config = {
        'd_model': 128,
        'd_ff': 1024,
        'num_layers': 3,
        'num_heads': 4,
        'dropout_rate': 0.1,
        'pretrained': False
    }

    model = FrettingTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        model_config=model_config
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    print(f"\n  Model outputs:")
    print(f"    loss:                    {outputs.loss.shape}  (scalar)")
    print(f"    logits:                  {outputs.logits.shape}  [B, L_dec-1, vocab_size]")

    # Verify shapes
    B, L_enc = input_ids.shape
    _, L_dec_minus_1 = decoder_input_ids.shape
    _, _, V_out = outputs.logits.shape

    print(f"\n  Shape verification:")
    print(f"    Batch size (B):          {B}")
    print(f"    Encoder length (L_enc):  {L_enc}")
    print(f"    Decoder length (L_dec-1):{L_dec_minus_1}")
    print(f"    Output vocab (V_out):    {V_out}")
    print(f"    Expected output vocab:   {output_vocab_size}")

    assert V_out == output_vocab_size, f"Vocab size mismatch: {V_out} != {output_vocab_size}"
    assert outputs.logits.shape == (B, L_dec_minus_1, V_out), "Logits shape mismatch"

    # 6. Test generation
    print("\n### 6. Generation (inference)")
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=1
    )
    print(f"  Generated tokens:        {generated.shape}  [B, L_gen]")

    print("\n" + "=" * 80)
    print("✅ All tensor shapes verified correctly!")
    print("=" * 80)

    # Summary table
    print("\n### Shape Summary")
    print("─" * 80)
    print(f"{'Stage':<30} {'Shape':<40} {'Notes'}")
    print("─" * 80)
    print(f"{'Dataset __getitem__':<30} {str(input_ids.shape):<40} {'Variable length'}")
    print(f"{'Collate function':<30} {str(batch['input_ids'].shape):<40} {'Padded to max'}")
    print(f"{'After shift':<30} {str(decoder_input_ids.shape):<40} {'Removed 1 token'}")
    print(f"{'Model logits':<30} {str(outputs.logits.shape):<40} {'+ vocab dim'}")
    print(f"{'Loss':<30} {'scalar':<40} {'Cross-entropy'}")
    print(f"{'Generated':<30} {str(generated.shape):<40} {'Autoregressive'}")
    print("─" * 80)


if __name__ == "__main__":
    test_tensor_shapes()
