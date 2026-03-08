#!/usr/bin/env python3
"""
Standalone script to run post-processing on model predictions.

Usage:
    python run_post_processing.py --output-dir out-v1-500epoch/ --method fret
    python run_post_processing.py --output-dir out-v1-500epoch/ --method reverse
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import importlib.util

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tab_dataset import build_vocabulary
from src.metrics import compute_tablature_accuracy

def load_post_processing_module(method: str):
    """
    Load the appropriate post-processing module based on method.

    Args:
        method: 'fret' or 'reverse'

    Returns:
        post_process_pitch_alignment function from the selected module
    """
    if method == 'fret':
        module_path = Path(__file__).parent / "src" / "post_processing_fret.py"
    elif method == 'reverse':
        module_path = Path(__file__).parent / "src" / "post_processing_reverse.py"
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'fret' or 'reverse'")

    if not module_path.exists():
        raise FileNotFoundError(f"Post-processing module not found: {module_path}")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location(
        f"post_processing_{method}",
        str(module_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.post_process_pitch_alignment

def load_tensors(output_dir: Path):
    """Load input_ids and predictions from output directory."""
    predictions_path = output_dir / "predictions.pt"
    input_ids_path = output_dir / "input_ids.pt"
    targets_path = output_dir / "targets.pt"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    if not input_ids_path.exists():
        raise FileNotFoundError(f"Input IDs not found: {input_ids_path}")
    
    print(f"Loading predictions from: {predictions_path}")
    predictions = torch.load(predictions_path, map_location="cpu")
    
    print(f"Loading input_ids from: {input_ids_path}")
    input_ids = torch.load(input_ids_path, map_location="cpu")
    
    targets = None
    if targets_path.exists():
        print(f"Loading targets from: {targets_path}")
        targets = torch.load(targets_path, map_location="cpu")
        
    return input_ids, predictions, targets

def main():
    parser = argparse.ArgumentParser(description="Run post-processing on model predictions")

    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        required=True,
        help="Directory containing predictions.pt and input_ids.pt"
    )

    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=['fret', 'reverse'],
        required=True,
        help="Post-processing method: 'fret' (simple alignment) or 'reverse' (timeline-based matching)"
    )

    # Vocabulary parameters (defaults matching common configs)
    parser.add_argument("--max-pitch", type=int, default=127)
    parser.add_argument("--max-time-shift", type=int, default=500)
    parser.add_argument("--num-strings", type=int, default=6)
    parser.add_argument("--num-frets", type=int, default=21)
    parser.add_argument("--output-format", type=str, default="v1")

    # Parameters for 'reverse' method (timeline-based matching)
    parser.add_argument("--pitch-threshold", type=int, default=1,
                        help="Pitch tolerance for exact matching (semitones) - only for 'reverse' method")
    parser.add_argument("--time-threshold", type=int, default=240,
                        help="Time window for matching (ticks) - only for 'reverse' method")

    # Try to load config from directory if possible
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to base config")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print(f"Post-Processing Method: {args.method.upper()}")
    print(f"{'='*60}\n")
    
    # Load tensors
    input_ids_tensor, predictions_tensor, targets_tensor = load_tensors(output_dir)
    
    print(f"Input shape: {input_ids_tensor.shape}")
    print(f"Predictions shape: {predictions_tensor.shape}")
    
    # Build vocabulary
    # First check if there is a config in the output dir (often saved by Hydra)
    hydra_config_path = output_dir / ".hydra" / "config.yaml"
    if hydra_config_path.exists():
        print(f"Loading config from {hydra_config_path}")
        cfg = OmegaConf.load(hydra_config_path)
        data_cfg = cfg.get("data", {})
        max_pitch = data_cfg.get("max_pitch", args.max_pitch)
        max_time_shift = data_cfg.get("max_time_shift", args.max_time_shift)
        num_strings = data_cfg.get("num_strings", args.num_strings)
        num_frets = data_cfg.get("num_frets", args.num_frets)
        output_format = data_cfg.get("output_format", args.output_format)
    else:
        print("Using command line/default arguments for vocabulary")
        max_pitch = args.max_pitch
        max_time_shift = args.max_time_shift
        num_strings = args.num_strings
        num_frets = args.num_frets
        output_format = args.output_format

    print("\nBuilding vocabulary...")
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=max_pitch,
        max_time_shift=max_time_shift,
        num_strings=num_strings,
        num_frets=num_frets,
        output_format=output_format,
    )

    # Load the appropriate post-processing function
    print(f"\nLoading post-processing module: {args.method}")
    post_process_pitch_alignment = load_post_processing_module(args.method)

    # Print method-specific information
    if args.method == 'fret':
        print("Method: FRET (Simple Index-based Alignment)")
        print("  - Fast O(n) algorithm")
        print("  - Assumes TAB order matches input order")
        print("  - Best for high-quality model outputs")
    elif args.method == 'reverse':
        print("Method: REVERSE (Timeline-based Matching)")
        print("  - Three-tier matching strategy")
        print(f"  - Pitch threshold: {args.pitch_threshold} semitones")
        print(f"  - Time threshold: {args.time_threshold} ticks")
        print("  - Robust handling of timing errors")
        print("  - Generates missing TABs if needed")

    # Process predictions
    aligned_predictions = []
    
    input_ids_list = input_ids_tensor.tolist()
    predictions_list = predictions_tensor.tolist()
    
    total_swaps = 0
    total_forced = 0
    
    print("\nApplying pitch alignment post-processing...")
    for i in tqdm(range(len(predictions_list)), desc="Processing"):
        single_input_ids = input_ids_list[i]
        single_pred_ids = predictions_list[i]
        
        # Un-pad
        try:
            input_end_idx = single_input_ids.index(input_vocab.pad_id)
            input_tokens = single_input_ids[:input_end_idx]
        except ValueError:
            input_tokens = single_input_ids

        try:
            pred_end_idx = single_pred_ids.index(output_vocab.pad_id)
            pred_tokens = single_pred_ids[:pred_end_idx]
        except ValueError:
            pred_tokens = single_pred_ids

        # Apply alignment with method-specific parameters
        if args.method == 'reverse':
            # Timeline-based matching with additional parameters
            aligned_ids, stats = post_process_pitch_alignment(
                input_ids=input_tokens,
                pred_ids=pred_tokens,
                input_vocab=input_vocab,
                output_vocab=output_vocab,
                pitch_threshold=args.pitch_threshold,
                time_threshold=args.time_threshold,
                output_format=output_format
            )
        else:  # 'fret' method
            # Simple alignment
            aligned_ids, stats = post_process_pitch_alignment(
                input_ids=input_tokens,
                pred_ids=pred_tokens,
                input_vocab=input_vocab,
                output_vocab=output_vocab,
                output_format=output_format
            )

        aligned_predictions.append(aligned_ids)
        total_swaps += stats.get('swaps_made', 0)
        total_forced += stats.get('forced_corrections', 0)
    
    print(f"\nTotal Corrections: Swaps={total_swaps}, Forced={total_forced}")

    # Re-pad
    max_len = predictions_tensor.shape[1]
    padded_aligned_predictions = torch.full_like(predictions_tensor, output_vocab.pad_id)
    
    for i, seq in enumerate(aligned_predictions):
        seq_len = min(len(seq), max_len)
        padded_aligned_predictions[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
        
    # Save results with method suffix
    output_path = output_dir / f"predictions_post-{args.method}.pt"
    print(f"\nSaving post-processed predictions to: {output_path}")
    torch.save(padded_aligned_predictions, output_path)
    
    # Compute metrics if targets are available
    if targets_tensor is not None:
        print("\nComputing metrics for aligned predictions...")
        # Ensure targets and preds are on same device/dtype
        if padded_aligned_predictions.device != targets_tensor.device:
            padded_aligned_predictions = padded_aligned_predictions.to(targets_tensor.device)
            
        metrics_aligned = compute_tablature_accuracy(
            predictions=padded_aligned_predictions,
            targets=targets_tensor,
            output_vocab=output_vocab
        )
        
        print(f"\nResults (After Post-Processing):")
        print(f"  Token Accuracy:  {metrics_aligned.token_accuracy:.2%}")
        print(f"  Pitch Accuracy:  {metrics_aligned.pitch_accuracy:.2%}")
        print(f"  Tab Accuracy:    {metrics_aligned.tab_accuracy:.2%}")
        print(f"  Difficulty:      {metrics_aligned.difficulty:.5}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
