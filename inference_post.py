#!/usr/bin/env python3
"""
Inference script for Fretting-Transformer.
Runs autoregressive generation and computes accuracy metrics.
"""

import os
import json
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from functools import partial

from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy, compute_postprocessing_metrics
from src.dataloader import create_dataset, create_dataloader
from src.postprocessing_bridge import PostProcessingBridge
from fretting_postprocessor import GuitarConfig
from fretting_postprocessor.config import (
    STANDARD_TUNING,
    DROP_D_TUNING,
    HALF_STEP_DOWN,
    FULL_STEP_DOWN
)


def load_checkpoint(checkpoint_path: str, model, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    return model


def create_guitar_config(cfg) -> GuitarConfig:
    """
    Create GuitarConfig from Hydra configuration.

    Args:
        cfg: Hydra config with postprocessing.guitar settings

    Returns:
        GuitarConfig for post-processor
    """
    tuning_map = {
        'standard': STANDARD_TUNING,
        'drop_d': DROP_D_TUNING,
        'half_step_down': HALF_STEP_DOWN,
        'full_step_down': FULL_STEP_DOWN,
    }

    tuning = tuning_map.get(cfg.postprocessing.guitar.tuning, STANDARD_TUNING)

    return GuitarConfig(
        num_strings=cfg.postprocessing.guitar.num_strings,
        tuning=tuning,
        capo_fret=cfg.postprocessing.guitar.capo_fret,
        min_fret=cfg.postprocessing.guitar.min_fret,
        max_fret=cfg.postprocessing.guitar.max_fret
    )


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
        num_beams=cfg.training.get('ar_eval_num_beams', 1),  # Ignored but kept for compat
        max_batches=cfg.get('max_eval_batches', None),  # None = all batches
        use_teacher_forcing=cfg.training.get('ar_eval_use_teacher_forcing', True),  # NEW
        temperature=cfg.training.get('ar_eval_temperature', 1.0),  # NEW
    )

    # Post-processing integration
    if cfg.postprocessing.enabled:
        print("\n" + "=" * 80)
        print(f"Applying Post-Processing ({cfg.postprocessing.method})")
        print("=" * 80)

        # Create output directory
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Initialize bridge
        guitar_config = create_guitar_config(cfg)
        bridge = PostProcessingBridge(
            dataset.input_vocab,
            dataset.output_vocab,
            guitar_config
        )

        # Batch post-processing
        postprocessed_predictions = bridge.process_batch(
            input_ids=input_ids,
            predictions=predictions,
            method=cfg.postprocessing.method
        )

        # Compute comparison metrics
        pp_metrics = compute_postprocessing_metrics(
            raw_predictions=predictions,
            postprocessed_predictions=postprocessed_predictions,
            targets=targets,
            output_vocab=dataset.output_vocab,
            method=cfg.postprocessing.method
        )

        # Display results
        if cfg.postprocessing.verbose:
            print(f"\nPost-Processing Results:")
            print(f"  Raw Model     - Pitch: {pp_metrics.raw_pitch_accuracy:.2%}, Tab: {pp_metrics.raw_tab_accuracy:.2%}")
            print(f"  Post-Processed - Pitch: {pp_metrics.post_pitch_accuracy:.2%}, Tab: {pp_metrics.post_tab_accuracy:.2%}")
            print(f"  Improvement   - Pitch: {pp_metrics.pitch_improvement:+.2%}, Tab: {pp_metrics.tab_improvement:+.2%}")

        # Save results
        if cfg.postprocessing.save_intermediate:
            raw_predictions_file = os.path.join(cfg.output_dir, "raw_predictions.pt")
            torch.save(predictions, raw_predictions_file)
            print(f"\nSaved raw predictions to: {raw_predictions_file}")

        postprocessed_file = os.path.join(cfg.output_dir, "postprocessed_predictions.pt")
        torch.save(postprocessed_predictions, postprocessed_file)
        print(f"Saved post-processed predictions to: {postprocessed_file}")

        comparison_file = os.path.join(cfg.output_dir, "postprocessing_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump({
                'method': pp_metrics.method,
                'raw_metrics': {
                    'token_accuracy': pp_metrics.raw_token_accuracy,
                    'pitch_accuracy': pp_metrics.raw_pitch_accuracy,
                    'tab_accuracy': pp_metrics.raw_tab_accuracy,
                },
                'postprocessed_metrics': {
                    'token_accuracy': pp_metrics.post_token_accuracy,
                    'pitch_accuracy': pp_metrics.post_pitch_accuracy,
                    'tab_accuracy': pp_metrics.post_tab_accuracy,
                },
                'improvements': {
                    'token': pp_metrics.token_improvement,
                    'pitch': pp_metrics.pitch_improvement,
                    'tab': pp_metrics.tab_improvement,
                },
                'counts': {
                    'total_tokens': pp_metrics.total_tokens,
                    'total_notes': pp_metrics.total_notes,
                }
            }, f, indent=2)
        print(f"Saved comparison metrics to: {comparison_file}")

        # Update predictions to post-processed version
        predictions = postprocessed_predictions
        print("\nFinal predictions are post-processed.")

    # Save predictions and targets


    print(f"\nResults:")
    print(f"  Token Accuracy:  {metrics.token_accuracy:.2%}")
    print(f"  Pitch Accuracy:  {metrics.pitch_accuracy:.2%}")
    print(f"  Tab Accuracy:    {metrics.tab_accuracy:.2%}")
    print(f"  Total Tokens:    {metrics.total_tokens:,}")
    print(f"  Total Notes:     {metrics.total_notes:,}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    input_ids_file = cfg.output_dir  + "/input_ids.pt"
    targets_file = cfg.output_dir + "/targets.pt"
    predictions_file = cfg.output_dir + "/predictions.pt"
    torch.save(input_ids, input_ids_file)
    torch.save(targets, targets_file)
    torch.save(predictions, predictions_file)

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
