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
from pathlib import Path

from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy
from src.dataloader import create_dataset, create_dataloader
from src.visualization import *

def load_checkpoint(checkpoint_path: str, model, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    return model

class FTEvent:
    def __init__(self, type, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)

def decode_tokens_to_events(tokens):
    """
    Convert Fretting-Transformer tokens into event objects compatible with render_as_tablature().
    """
    events = []
    for t in tokens:
        if t.startswith("NOTE_ON_"):
            pitch = int(t.split("_")[2])
            events.append(FTEvent("NOTE_ON", pitch=pitch))
        elif t.startswith("NOTE_OFF_"):
            pitch = int(t.split("_")[2])
            events.append(FTEvent("NOTE_OFF", pitch=pitch))
        elif t.startswith("TIME_SHIFT_"):
            delta = int(t.split("_")[2])
            events.append(FTEvent("TIME_SHIFT", delta=delta))
        elif t.startswith("TAB_"):
            _, s, f = t.split("_")
            events.append(FTEvent("TAB", string=int(s), fret=int(f)))
        else:
            pass  # ignore
    return events


def visualize_tab_sample(gt_ids, pred_ids, output_vocab, max_bars=2):
    """
    Render predicted vs ground truth tablature for inspection.

    Args:
        gt_ids: Tensor of token IDs (ground truth)
        pred_ids: Tensor of token IDs (model prediction)
        output_vocab: Vocabulary object (must support id_to_token)
        max_bars: Number of bars to render in tablature
    """

    # Convert token IDs to token strings
    gt_tokens = [output_vocab.id_to_token[t.item()] for t in gt_ids]
    pred_tokens = [output_vocab.id_to_token[t.item()] for t in pred_ids]

    # Convert to events
    gt_events = decode_tokens_to_events(gt_tokens)
    pred_events = decode_tokens_to_events(pred_tokens)

    # Visualization using the official tab renderer
    gt_tab = render_as_tablature(gt_events, max_bars=4, chars_per_beat=8, bars_per_row=4)
    pred_tab = render_as_tablature(pred_events, max_bars=4, chars_per_beat=8, bars_per_row=4)

    print("\n" + "=" * 80)
    print("GROUND TRUTH TAB")
    print("=" * 80)
    print(gt_tab)

    print("\n" + "=" * 80)
    print("PREDICTED TAB")
    print("=" * 80)
    print(pred_tab)


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
        output_format=cfg.data.get('output_format', 'v1'),
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
    
    # debug
    # print(f"input_vocab_size = {input_vocab_size}")
    # print(f"output_vocab_size = {output_vocab_size}")

    # print("cfg.data:")
    # print(cfg.data)
    # breakpoint()

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
    # covert to Path
    cfg.output_dir = Path(cfg.output_dir)

    # Make sure folder exists
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  Difficulty:      {metrics.difficulty:.5}")
    print(f"  Total Tokens:    {metrics.total_tokens:,}")
    print(f"  Total Notes:     {metrics.total_notes:,}")


    # ======================================================================
    # VISUALIZATION
    # Show ground truth vs predicted tablature for a few samples
    # ======================================================================

    print("\n" + "=" * 80)
    print("Visualizing Predicted vs Ground Truth Tablature")
    print("=" * 80)

    num_samples_to_show = min(3, len(predictions))  # Change if needed

    for idx in range(num_samples_to_show):
        print(f"\n--- SAMPLE {idx} ---")
        visualize_tab_sample(
            gt_ids=targets[idx],
            pred_ids=predictions[idx],
            output_vocab=dataset.output_vocab,
            max_bars=2
        )


    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
