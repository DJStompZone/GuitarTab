#!/usr/bin/env python3
"""
Inference script for Fretting-Transformer.
Runs autoregressive generation and computes accuracy metrics.
"""

import os
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from functools import partial

from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy
from src.dataloader import create_dataset, create_dataloader


def load_checkpoint(checkpoint_path: str, model, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    return model


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    """Run inference and compute accuracy."""

    print("=" * 80)
    print("Fretting-Transformer Inference")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Use test split by default for inference
    if cfg.data.selected_files_json is None:
        print("No selected_files_json specified, using test split by default...")
        cfg.data.selected_files_json = "data_splits/test_files.json"

    # Create dataset
    print(f"Loading dataset from {cfg.data.selected_files_json}")
    dataset = create_dataset(
        data_dir=cfg.data.data_dir,
        token_pattern=cfg.data.token_pattern,
        selected_files_json=cfg.data.selected_files_json,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
        max_files=cfg.data.get('max_files', None)
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    print(f"Dataset: {len(dataset)} segments\n")

    input_vocab_size, output_vocab_size = dataset.get_vocab_sizes()

    # Create model
    print("Initializing model...")
    from omegaconf import OmegaConf
    model = FrettingTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        model_config=OmegaConf.to_container(cfg.model)
    ).to(device)

    # Load checkpoint
    checkpoint_path = cfg.get('checkpoint_path', 'outputs/fretting_transformer/best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_checkpoint(checkpoint_path, model, device)

    # Run inference
    print("\n" + "=" * 80)
    print(f"Evaluating on {cfg.data.selected_files_json}")
    print("=" * 80)

    metrics, (input_ids, targets, predictions) = generate_and_compute_accuracy(
        model=model,
        dataloader=dataloader,
        output_vocab=dataset.output_vocab,
        device=device,
        max_length=cfg.training.get('ar_eval_max_length', 1024),
        num_beams=cfg.training.get('ar_eval_num_beams', 1),
        max_batches=cfg.get('max_eval_batches', None)  # None = all batches
    )

    # Save predictions and targets
    input_ids_file = cfg.output_dir / "input_ids.pt"
    targets_file = cfg.output_dir / "targets.pt"
    predictions_file = cfg.output_dir / "predictions.pt"
    torch.save(input_ids, input_ids_file)
    torch.save(targets, targets_file)
    torch.save(predictions, predictions_file)


    print(f"\nResults:")
    print(f"  Token Accuracy:  {metrics.token_accuracy:.2%}")
    print(f"  Pitch Accuracy:  {metrics.pitch_accuracy:.2%}")
    print(f"  Tab Accuracy:    {metrics.tab_accuracy:.2%}")
    print(f"  Total Tokens:    {metrics.total_tokens:,}")
    print(f"  Total Notes:     {metrics.total_notes:,}")

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
