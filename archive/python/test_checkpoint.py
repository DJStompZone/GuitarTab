#!/usr/bin/env python3
"""
Quick test to verify checkpoint can be loaded and used for inference.
"""

import torch
import numpy as np
from src.model import FrettingTransformer
from src.tab_dataset import TabDataset


def test_checkpoint():
    """Test loading checkpoint and running inference."""

    # Load checkpoint
    checkpoint = torch.load('outputs/fretting_transformer/best_model.pt', map_location='cpu')

    print("=" * 80)
    print("Checkpoint Verification")
    print("=" * 80)
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint['train_loss']:.4f}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")

    # Get config
    config = checkpoint['config']
    model_config = config['model']

    # Create model
    print("\nLoading model...")
    model = FrettingTransformer(
        input_vocab_size=760,
        output_vocab_size=886,
        model_config=model_config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test inference with dummy input
    print("\nTesting inference with dummy input...")
    batch_size = 2
    seq_len = 64

    dummy_input = torch.randint(0, 760, (batch_size, seq_len))
    dummy_attention_mask = torch.ones_like(dummy_input)

    with torch.no_grad():
        # Generate output
        generated = model.generate(
            input_ids=dummy_input,
            attention_mask=dummy_attention_mask,
            max_length=128,
            num_beams=1
        )

    print(f"Input shape: {dummy_input.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens (first sequence, first 20): {generated[0, :20].tolist()}")

    print("\n" + "=" * 80)
    print("✅ Checkpoint loaded successfully and model is functional!")
    print("=" * 80)


if __name__ == "__main__":
    test_checkpoint()
