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
from transformers.modeling_outputs import Seq2SeqLMOutput
from omegaconf import DictConfig, OmegaConf
from functools import partial

from src.tab_dataset import TabDataset
from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy
from src.dataloader import create_dataset, create_dataloader
from src.training_logger import TrainingLogger, save_generated_samples


from typing import TypeAlias
from src.tab_dataset import TabDatasetBatchInput


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader, TabDataset]:
    """
    Create train/val/test dataloaders from pre-split file lists.

    Args:
        cfg: Hydra config

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset)
    """
    selected_file = cfg.data.selected_files_json

    # Check if using pre-split files
    if selected_file and any(split in selected_file for split in ['train_files.json', 'val_files.json', 'test_files.json']):
        # Using pre-split files - load each split separately
        split_dir = Path(selected_file).parent
        train_file = str(split_dir / 'train_files.json')
        val_file = str(split_dir / 'val_files.json')
        test_file = str(split_dir / 'test_files.json')

        print(f"Loading pre-split files:")
        print(f"  Train: {train_file}")
        print(f"  Val:   {val_file}")
        print(f"  Test:  {test_file}")

        # Create three separate datasets (NO random splitting!)
        train_dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=train_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            max_files=cfg.data.max_files
        )

        val_dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=val_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            max_files=None
        )

        test_dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=test_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            max_files=None
        )

        print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    else:
        # Fallback: old behavior for debug mode
        dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=selected_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            max_files=cfg.data.max_files
        )

        # Split at segment level (for debug only)
        total_size = len(dataset)
        train_size = int(total_size * cfg.data.train_ratio)
        val_size = int(total_size * cfg.data.val_ratio)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        print(f"Dataset split (segment-level): train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset if isinstance(train_dataset, TabDataset) else train_dataset.dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    val_loader = create_dataloader(
        val_dataset if isinstance(val_dataset, TabDataset) else val_dataset.dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    test_loader = create_dataloader(
        test_dataset if isinstance(test_dataset, TabDataset) else test_dataset.dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    # Return train_dataset for vocabulary access
    return_dataset = train_dataset if isinstance(train_dataset, TabDataset) else train_dataset.dataset
    return train_loader, val_loader, test_loader, return_dataset




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
    train_loader: DataLoader,
    optimizer,
    device: str,
    epoch: int,
    cfg: DictConfig,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar: tqdm[TabDatasetBatchInput] = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        # Move to device
        # Shapes from DataLoader (collate_fn output):
        #   input_ids: [B, L_enc] - Encoder input token IDs
        #   attention_mask: [B, L_enc] - Encoder padding mask (1=real, 0=pad)
        #   output_ids: [B, L_dec] - Full decoder sequence (for teacher forcing)
        #   decoder_attention_mask: [B, L_dec] - Decoder padding mask (1=real, 0=pad)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["output_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        # Shift labels right for decoder input (teacher forcing)
        # T5 expects: decoder_input = [BOS, tok1, tok2, ..., tokN-1]
        #             labels = [tok1, tok2, ..., tokN, EOS]
        # Shapes after shifting:
        #   decoder_input_ids: [B, L_dec-1] - Decoder input (all except last token)
        #   labels: [B, L_dec-1] - Target labels (all except first token)
        #   decoder_attention_mask: [B, L_dec-1] - Mask for decoder input
        decoder_input_ids = labels[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
        decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()

        # Forward pass
        outputs: Seq2SeqLMOutput = model(
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
        batch: TabDatasetBatchInput
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            # Shapes: [B, L_enc], [B, L_enc], [B, L_dec], [B, L_dec]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["output_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)

            # Shift labels right
            # Shapes after: [B, L_dec-1], [B, L_dec-1], [B, L_dec-1]
            decoder_input_ids = labels[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()

            # Forward pass
            outputs: Seq2SeqLMOutput = model(
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
    # print("\nConfiguration:")
    # print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"Saved config to {config_path}")

    # Initialize training logger
    logger = TrainingLogger(log_file=output_dir / "training_log.json")
    print(f"Logging to {output_dir / 'training_log.json'}")

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

        # Evaluate (teacher forcing loss)
        val_loss = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")

        # Log epoch metrics
        logger.log_epoch(epoch=epoch, train_loss=train_loss, val_loss=val_loss)

        # Autoregressive evaluation (real accuracy)
        if cfg.training.get('ar_eval_enabled', False) and cfg.training.get('ar_eval_frequency', 0) > 0:
            if epoch % cfg.training.ar_eval_frequency == 0:
                print(f"\nRunning AR evaluation (generation + accuracy)...")
                ar_result = generate_and_compute_accuracy(
                    model=model,
                    dataloader=val_loader,
                    output_vocab=dataset.output_vocab,
                    device=device,
                    max_length=cfg.training.get('ar_eval_max_length', 1024),
                    num_beams=cfg.training.get('ar_eval_num_beams', 1),
                    max_batches=cfg.training.get('ar_eval_max_batches', None),
                    return_predictions=True
                )
                ar_metrics, (predictions, targets) = ar_result
                print(f"AR Metrics: {ar_metrics}")

                # Save generated samples
                samples_file = output_dir / f"generated_samples_epoch_{epoch}.json"
                save_generated_samples(
                    predictions=predictions.cpu().numpy(),
                    targets=targets.cpu().numpy(),
                    output_vocab=dataset.output_vocab,
                    output_file=samples_file,
                    max_samples=10
                )

                # Log AR eval to history
                logger.log_ar_eval(
                    epoch=epoch,
                    token_accuracy=ar_metrics.token_accuracy,
                    pitch_accuracy=ar_metrics.pitch_accuracy,
                    tab_accuracy=ar_metrics.tab_accuracy,
                    total_tokens=ar_metrics.total_tokens,
                    total_notes=ar_metrics.total_notes
                )

        # Save best checkpoint
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

        # Save checkpoint every N epochs
        checkpoint_every_n = cfg.training.get('checkpoint_every_n_epochs', 0)
        if checkpoint_every_n > 0 and epoch % checkpoint_every_n == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
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
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final evaluation on test set")
    print("=" * 80)
    test_loss = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")

    # Log test loss
    logger.log_test(test_loss=test_loss)

    # Final AR evaluation on test set
    if cfg.training.get('ar_eval_enabled', False):
        print(f"\nFinal AR evaluation on test set...")
        test_ar_result = generate_and_compute_accuracy(
            model=model,
            dataloader=test_loader,
            output_vocab=dataset.output_vocab,
            device=device,
            max_length=cfg.training.get('ar_eval_max_length', 1024),
            num_beams=cfg.training.get('ar_eval_num_beams', 1),
            max_batches=None,  # Use all batches for final evaluation
            return_predictions=True
        )
        test_ar_metrics, (predictions, targets) = test_ar_result
        print(f"\nTest Set AR Metrics:")
        print(f"  Token Accuracy:  {test_ar_metrics.token_accuracy:.2%}")
        print(f"  Pitch Accuracy:  {test_ar_metrics.pitch_accuracy:.2%}")
        print(f"  Tab Accuracy:    {test_ar_metrics.tab_accuracy:.2%}")

        # Save final test generated samples
        samples_file = output_dir / "test_generated_samples.json"
        save_generated_samples(
            predictions=predictions.cpu().numpy(),
            targets=targets.cpu().numpy(),
            output_vocab=dataset.output_vocab,
            output_file=samples_file,
            max_samples=20  # Save more samples for final test
        )

        # Log test AR eval to history
        logger.log_test_ar_eval(
            token_accuracy=test_ar_metrics.token_accuracy,
            pitch_accuracy=test_ar_metrics.pitch_accuracy,
            tab_accuracy=test_ar_metrics.tab_accuracy,
            total_tokens=test_ar_metrics.total_tokens,
            total_notes=test_ar_metrics.total_notes
        )

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
