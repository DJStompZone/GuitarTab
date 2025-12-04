#!/usr/bin/env python3
"""
Training script for Fretting-Transformer using Hydra configuration.
"""

import os
from pathlib import Path
from glob import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial

from src.tab_dataset import TabDataset, collate_fn
from src.model import FrettingTransformer


from typing import TypeAlias
from src.tab_dataset import TabDatasetBatchInput

TabDataLoader: TypeAlias = DataLoader[TabDatasetBatchInput]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(cfg: DictConfig) -> tuple[TabDataLoader, TabDataLoader, TabDataLoader, TabDataset]:
    """
    Create train/val/test dataloaders.

    Args:
        cfg: Hydra config

    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    # Find all token files
    token_files = sorted(
        glob(os.path.join(cfg.data.data_dir, cfg.data.token_pattern), recursive=True)
    )

    # Filter by selected files if provided
    if cfg.data.selected_files_json is not None:
        print(f"Loading selected files from {cfg.data.selected_files_json}")
        with open(cfg.data.selected_files_json, "r") as f:
            selected_files = set(json.load(f))

        # Filter token files by removing .tokens.txt suffix and checking against selected files
        filtered_token_files = []
        for token_file in token_files:
            # Remove .tokens.txt suffix to get the original .gp filename
            if token_file.endswith(".tokens.txt"):
                gp_file = token_file[: -len(".tokens.txt")]
                if gp_file in selected_files:
                    filtered_token_files.append(token_file)

        token_files = filtered_token_files
        print(f"Filtered to {len(token_files)} selected files")

    if cfg.data.max_files is not None:
        token_files = token_files[: cfg.data.max_files]

    print(f"Using {len(token_files)} token files")

    if not token_files:
        raise ValueError(f"No token files found in {cfg.data.data_dir}")

    # Create dataset
    dataset = TabDataset(
        token_files=token_files,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
    )

    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * cfg.data.train_ratio)
    val_size = int(total_size * cfg.data.val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    print(
        f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    # Get pad IDs
    input_pad_id, output_pad_id = dataset.get_pad_ids()

    # Create collate function with pad IDs
    collate_fn_with_pads = partial(
        collate_fn, input_pad_id=input_pad_id, output_pad_id=output_pad_id
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn_with_pads,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn_with_pads,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn_with_pads,
    )

    return train_loader, val_loader, test_loader, dataset


def create_optimizer(model: nn.Module, cfg: DictConfig):
    """Create optimizer based on config."""
    if cfg.training.optimizer.name == "adafactor":
        from transformers import Adafactor

        optimizer = Adafactor(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            scale_parameter=cfg.training.optimizer.scale_parameter,
            relative_step=cfg.training.optimizer.relative_step,
            warmup_init=cfg.training.optimizer.warmup_init,
        )
    elif cfg.training.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer.name}")

    return optimizer


def train_epoch(
    model: FrettingTransformer,
    train_loader: TabDataLoader,
    optimizer,
    device: str,
    epoch: int,
    cfg: DictConfig,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        # Move to device
        input_ids = torch.from_numpy(batch["input_ids"]).to(device)
        attention_mask = torch.from_numpy(batch["attention_mask"]).to(device)
        labels = torch.from_numpy(batch["output_ids"]).to(device)
        decoder_attention_mask = torch.from_numpy(batch["decoder_attention_mask"]).to(
            device
        )

        # Shift labels right for decoder input (teacher forcing)
        decoder_input_ids = labels[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
        decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        if cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model: FrettingTransformer, val_loader: DataLoader, device: str):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            input_ids = torch.from_numpy(batch["input_ids"]).to(device)
            attention_mask = torch.from_numpy(batch["attention_mask"]).to(device)
            labels = torch.from_numpy(batch["output_ids"]).to(device)
            decoder_attention_mask = torch.from_numpy(
                batch["decoder_attention_mask"]
            ).to(device)

            # Shift labels right
            decoder_input_ids = labels[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 80)
    print("Fretting-Transformer Training")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(cfg)

    # Get vocabulary sizes
    input_vocab_size, output_vocab_size = dataset.get_vocab_sizes()

    # Create model
    print("\nInitializing model...")
    model = FrettingTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        model_config=OmegaConf.to_container(cfg.model),
    )
    model = model.to(device)

    # Create optimizer
    optimizer = create_optimizer(model, cfg)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{cfg.training.num_epochs}")
        print(f"{'=' * 80}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, cfg)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": OmegaConf.to_container(cfg),
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final evaluation on test set")
    print("=" * 80)
    test_loss = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
