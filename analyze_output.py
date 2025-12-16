#!/usr/bin/env python3
"""
Standalone CLI script to analyze Fretting-Transformer output format.

Usage:
    python analyze_output.py path/to/folder/
    
    # Or with specific files
    python analyze_output.py \
        --predictions outputs/predictions.pt \
        --targets outputs/targets.pt \
        --output outputs/analysis_result.json \
        --config configs/inference.yaml
    
    # Run analysis on both predictions and predictions_post (if exists)
    python analyze_output.py path/to/folder/ --post
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tab_dataset import build_vocabulary
from src.output_analysis import (
    analyze_all_predictions,
    save_results_json,
    print_summary,
    print_issue_summary,
    print_timeline_summary,
    generate_timeline_report,
)
from src.metrics import compute_tablature_accuracy, TabAccuracyMetrics


def load_tensors(predictions_path: str, targets_path: str, input_ids_path: str = None):
    """Load prediction and target tensors."""
    print(f"Loading predictions from: {predictions_path}")
    predictions = torch.load(predictions_path, map_location="cpu")

    print(f"Loading targets from: {targets_path}")
    targets = torch.load(targets_path, map_location="cpu")

    input_ids = None
    if input_ids_path and os.path.exists(input_ids_path):
        print(f"Loading input_ids from: {input_ids_path}")
        input_ids = torch.load(input_ids_path, map_location="cpu")

    return predictions, targets, input_ids


def tensor_to_list(tensor: torch.Tensor) -> list:
    """Convert tensor to list of lists."""
    if tensor.dim() == 1:
        return [tensor.tolist()]
    return tensor.tolist()


def run_analysis_and_save(
    predictions_tensor,
    targets_tensor,
    output_vocab,
    args,
    analysis_dir: Path,
    label: str = "Standard"
):
    print(f"\n{'='*80}")
    print(f"Running Analysis: {label}")
    print(f"{'='*80}")

    # Convert to lists
    pred_list = tensor_to_list(predictions_tensor)
    target_list = tensor_to_list(targets_tensor)

    # Limit samples if requested
    if args.max_samples:
        pred_list = pred_list[:args.max_samples]
        target_list = target_list[:args.max_samples]
        # Tensor slicing for metrics
        predictions_tensor = predictions_tensor[:args.max_samples]
        targets_tensor = targets_tensor[:args.max_samples]

    # Run analysis
    print(f"\nAnalyzing {len(pred_list)} samples...")
    result = analyze_all_predictions(
        predictions=pred_list,
        targets=target_list,
        vocab=output_vocab,
        output_format=args.output_format,
        timeline_tolerance=args.timeline_tolerance,
        verbose=not args.quiet,
        calc_levenshtein=args.calc_levenshtein,
    )

    # Ensure directory exists
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Print and Save Issue Summary
    summary_file = analysis_dir / "issue_summary.txt"
    print(f"\nSaving issue summary to: {summary_file}")

    with open(summary_file, "w", encoding="utf-8") as f:
        # We need to redirect stdout to capture the print functions
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            print_issue_summary(result, output_format=args.output_format)
            print_detailed_issue_statistics(result, top_k=args.top_k)
            # Also save position distribution to issue summary
            from src.output_analysis import print_position_distribution
            print_position_distribution(result)
        finally:
            sys.stdout = original_stdout

    # 2. Print and Save Timeline Summary
    timeline_summary_file = analysis_dir / "timeline_summary.txt"
    print(f"Saving timeline summary to: {timeline_summary_file}")
    
    with open(timeline_summary_file, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            print_timeline_summary(result)
            print_detailed_timeline_statistics(result)
        finally:
            sys.stdout = original_stdout

    # Print summaries to console
    print_issue_summary(result, output_format=args.output_format)
    print_detailed_issue_statistics(result, top_k=args.top_k)

    # Print position distribution to console
    from src.output_analysis import print_position_distribution
    print_position_distribution(result)

    # 3. Generate and Save Timeline Report (Detailed)
    timeline_report_file = analysis_dir / "timeline_analysis.txt"
    print(f"\nGenerating detailed timeline report for top {args.top_k} worst samples...")
    report_content = generate_timeline_report(result, top_k=args.top_k)
    
    with open(timeline_report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved detailed timeline report to: {timeline_report_file}")

    # 4. Compute and Save Tab Accuracy Metrics
    print(f"\nComputing detailed tab accuracy metrics...")
    
    tab_metrics = compute_tablature_accuracy(
        predictions=predictions_tensor,
        targets=targets_tensor,
        output_vocab=output_vocab,
    )
    
    metrics_file = analysis_dir / "accuracy_metrics.txt"
    print(f"Saving accuracy metrics to: {metrics_file}")
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Tablature Accuracy Metrics ({label})\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Token Accuracy:   {tab_metrics.token_accuracy:.2%}\n")
        f.write(f"Pitch Accuracy:   {tab_metrics.pitch_accuracy:.2%} (Notes with correct pitch)\n")
        f.write(f"Tab Accuracy:     {tab_metrics.tab_accuracy:.2%} (Notes with correct string/fret)\n")
        f.write(f"Difficulty Score: {tab_metrics.difficulty:.4f}\n")
        f.write(f"\nTotal Tokens:     {tab_metrics.total_tokens}\n")
        f.write(f"Total Notes:      {tab_metrics.total_notes}\n")
        f.write("\n" + "=" * 80 + "\n")

    # Also print to console
    print("\n" + "=" * 80)
    print(f"Tablature Accuracy Metrics ({label})")
    print("=" * 80)
    print(f"Token Accuracy:   {tab_metrics.token_accuracy:.2%}")
    print(f"Pitch Accuracy:   {tab_metrics.pitch_accuracy:.2%}")
    print(f"Tab Accuracy:     {tab_metrics.tab_accuracy:.2%}")
    print(f"Difficulty Score: {tab_metrics.difficulty:.4f}")
    print("=" * 80)

    # 5. Save JSON results
    # Use output filename from args if it's the standard run, otherwise derive from dir
    if label == "Standard" and args.output != "analysis_result.json":
         json_path = args.output
    else:
         json_path = str(analysis_dir / "analysis_result.json")

    print(f"\nSaving detailed JSON results to: {json_path}")
    save_results_json(result, json_path)

    print(f"\nAll analysis outputs for {label} have been saved to: {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Fretting-Transformer output format"
    )

    # Positional argument for folder
    parser.add_argument(
        "folder",
        nargs="?",
        type=str,
        help="Input directory containing predictions.pt and targets.pt"
    )

    # File paths
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        help="Path to predictions.pt file"
    )
    parser.add_argument(
        "--targets", "-t",
        type=str,
        help="Path to targets.pt file"
    )
    parser.add_argument(
        "--input-ids", "-i",
        type=str,
        help="Path to input_ids.pt file (optional)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="analysis_result.json",
        help="Output JSON file path (default: analysis_result.json)"
    )

    # Convenience: specify output directory from inference (Legacy support)
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        help="Hydra output directory containing predictions.pt, targets.pt, etc."
    )

    # Post-processing flag
    parser.add_argument(
        "--post",
        action="store_true",
        help="Also analyze predictions_post.pt if available"
    )
    parser.add_argument(
        "--post-output-dir",
        type=str,
        help="Directory to save post-processing analysis results"
    )

    # Config for vocabulary
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/inference.yaml",
        help="Path to config file for vocabulary settings"
    )

    # Vocabulary parameters (if not using config)
    parser.add_argument(
        "--max-pitch", type=int, default=127,
        help="Maximum MIDI pitch (default: 127)"
    )
    parser.add_argument(
        "--max-time-shift", type=int, default=500,
        help="Maximum time shift (default: 500)"
    )
    parser.add_argument(
        "--num-strings", type=int, default=6,
        help="Number of guitar strings (default: 6)"
    )
    parser.add_argument(
        "--num-frets", type=int, default=21,
        help="Number of frets (default: 21)"
    )
    parser.add_argument(
        "--output-format", type=str, default="v1",
        choices=["v1", "v2", "v3"],
        help="Output format version (default: v1)"
    )

    # Analysis options
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to analyze (default: all)"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top error samples to show (default: 10)"
    )
    parser.add_argument(
        "--timeline-tolerance", type=int, default=10,
        help="Tolerance (in ticks) for timeline alignment matching (default: 10)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--calc-levenshtein",
        action="store_true",
        help="Calculate Levenshtein distance (slow)"
    )

    args = parser.parse_args()

    # Resolve file paths
    folder = args.folder or args.output_dir
    
    if folder:
        output_dir = Path(folder)
        
        # Determine predictions path (plural or singular)
        # Priority: args.predictions > folder/predictions.pt > folder/prediction.pt
        if args.predictions:
            predictions_path = args.predictions
        elif (output_dir / "predictions.pt").exists():
            predictions_path = str(output_dir / "predictions.pt")
        elif (output_dir / "prediction.pt").exists():
            predictions_path = str(output_dir / "prediction.pt")
        else:
             # Default to predictions.pt even if missing, let validation fail later
             predictions_path = str(output_dir / "predictions.pt")

        # Determine targets path (plural or singular)
        # Priority: args.targets > folder/targets.pt > folder/target.pt
        if args.targets:
            targets_path = args.targets
        elif (output_dir / "targets.pt").exists():
            targets_path = str(output_dir / "targets.pt")
        elif (output_dir / "target.pt").exists():
            targets_path = str(output_dir / "target.pt")
        else:
             targets_path = str(output_dir / "targets.pt")

        if args.input_ids:
            input_ids_path = args.input_ids
        else:
            input_ids_path = str(output_dir / "input_ids.pt")
        
        # Determine analysis output directory
        analysis_dir = output_dir / "analysis_report"
        
        if not args.output or args.output == "analysis_result.json":
            args.output = str(analysis_dir / "analysis_result.json")
    else:
        predictions_path = args.predictions
        targets_path = args.targets
        input_ids_path = args.input_ids
        output_dir = None # No base folder if using direct paths
        
        # If output path is provided, use its parent directory as base
        if args.output:
            analysis_dir = Path(args.output).parent
        else:
            analysis_dir = Path("analysis_report")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(analysis_dir / "analysis_result.json")

    # Validate paths
    if not predictions_path or not os.path.exists(predictions_path):
        print(f"Error: predictions file not found: {predictions_path}")
        if folder:
             print(f"Checked both 'predictions.pt' and 'prediction.pt' in {folder}")
        sys.exit(1)
    if not targets_path or not os.path.exists(targets_path):
        print(f"Error: targets file not found: {targets_path}")
        if folder:
             print(f"Checked both 'targets.pt' and 'target.pt' in {folder}")
        sys.exit(1)

    # Load tensors
    predictions, targets, input_ids = load_tensors(
        predictions_path, targets_path, input_ids_path
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    if input_ids is not None:
        print(f"Input IDs shape: {input_ids.shape}")

    # Build vocabulary
    print("\nBuilding vocabulary...")

    # Try to load config if exists
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        cfg = OmegaConf.load(config_path)
        data_cfg = cfg.get("data", {})
        max_pitch = data_cfg.get("max_pitch", args.max_pitch)
        max_time_shift = data_cfg.get("max_time_shift", args.max_time_shift)
        num_strings = data_cfg.get("num_strings", args.num_strings)
        num_frets = data_cfg.get("num_frets", args.num_frets)
        output_format = data_cfg.get("output_format", args.output_format)
    else:
        print(f"Config not found at {config_path}, using command line args")
        max_pitch = args.max_pitch
        max_time_shift = args.max_time_shift
        num_strings = args.num_strings
        num_frets = args.num_frets
        output_format = args.output_format

    _, output_vocab = build_vocabulary(
        max_pitch=max_pitch,
        max_time_shift=max_time_shift,
        num_strings=num_strings,
        num_frets=num_frets,
        output_format=output_format,
    )

    print(f"Output vocab size: {output_vocab.vocab_size}")
    print(f"Output format: {output_format}")

    # 1. Run Standard Analysis
    run_analysis_and_save(
        predictions_tensor=predictions,
        targets_tensor=targets,
        output_vocab=output_vocab,
        args=args,
        analysis_dir=analysis_dir,
        label="Standard"
    )

    # 2. Run Post-Processing Analysis (if requested and available)
    if args.post:
        print("\nChecking for post-processed predictions...")
        post_path = None
        
        # Determine path for predictions_post.pt
        if output_dir:
            if (output_dir / "predictions_post.pt").exists():
                post_path = output_dir / "predictions_post.pt"
            elif (output_dir / "prediction_post.pt").exists():
                post_path = output_dir / "prediction_post.pt"
        else:
            # Try to infer from predictions_path
            pred_path = Path(predictions_path)
            parent = pred_path.parent
            if (parent / "predictions_post.pt").exists():
                post_path = parent / "predictions_post.pt"
            elif (parent / "prediction_post.pt").exists():
                post_path = parent / "prediction_post.pt"
        
        if post_path:
            print(f"Found post-processed predictions: {post_path}")
            predictions_post = torch.load(post_path, map_location="cpu")
            print(f"Predictions Post shape: {predictions_post.shape}")
            
            # Use a separate directory for post-analysis
            # If using folder mode, it's parallel to analysis_report
            # If using explicit output, we modify the parent dir
            if args.post_output_dir:
                analysis_dir_post = Path(args.post_output_dir)
            elif output_dir:
                analysis_dir_post = output_dir / "analysis_report_post"
            else:
                # If explicit output like "outs/res.json", standard goes to "outs", post goes to "outs_post"
                # Or just put it in a subfolder relative to where standard report went
                analysis_dir_post = analysis_dir.parent / "analysis_report_post"
            
            run_analysis_and_save(
                predictions_tensor=predictions_post,
                targets_tensor=targets,
                output_vocab=output_vocab,
                args=args,
                analysis_dir=analysis_dir_post,
                label="Post-Processed"
            )
        else:
            print("Warning: --post flag set but 'predictions_post.pt' not found.")

    print("\nDone!")


def print_detailed_issue_statistics(result, top_k: int = 5):
    """Print detailed structural issue statistics."""
    from collections import defaultdict

    print("\n" + "=" * 80)
    print("Detailed Issue Statistics")
    print("=" * 80)

    num_samples = result.total_samples
    if num_samples == 0:
        print("No samples to analyze.")
        return

    # 1. Average issues per sample by type
    print(f"\n--- Average Issues Per Sample (Total: {num_samples} samples) ---")
    print(f"{'Issue Type':<30s} {'Total':>10s} {'Avg/Sample':>12s}")
    print("-" * 54)

    issue_types = [
        "missing_tab",
        "missing_note_off",
        "pitch_mismatch",
        "orphan_note_off",
        "orphan_tab",
        "extra_note",
        "absent_note",
        "time_shift_mismatch",
    ]

    total_all_issues = 0
    for issue_type in issue_types:
        count = result.issue_counts.get(issue_type, 0)
        avg = count / num_samples
        total_all_issues += count
        if count > 0:
            print(f"{issue_type:<30s} {count:>10d} {avg:>12.2f}")

    print("-" * 54)
    print(f"{'TOTAL':<30s} {total_all_issues:>10d} {total_all_issues / num_samples:>12.2f}")

    # 2. Top-K samples with most issues
    print(f"\n--- Top {top_k} Samples with Most Issues ---")

    # Sort samples by total issue count
    sorted_samples = sorted(
        result.per_sample,
        key=lambda s: sum(s.issue_counts.values()),
        reverse=True
    )

    print(f"{'Rank':<6s} {'Sample':<10s} {'Total':<8s} | Issue Breakdown")
    print("-" * 80)

    for rank, sample in enumerate(sorted_samples[:top_k], 1):
        total_issues = sum(sample.issue_counts.values())

        # Format issue breakdown
        breakdown_parts = []
        for issue_type in issue_types:
            count = sample.issue_counts.get(issue_type, 0)
            if count > 0:
                # Shorten issue type name for display
                short_name = issue_type.replace("_", " ").replace("note off", "n_off").replace("note", "n")
                breakdown_parts.append(f"{short_name}:{count}")

        breakdown = ", ".join(breakdown_parts) if breakdown_parts else "none"

        print(f"{rank:<6d} {sample.sample_idx:<10d} {total_issues:<8d} | {breakdown}")

    # 3. Distribution summary
    print(f"\n--- Issue Distribution Summary ---")

    # Count samples by issue count ranges
    ranges = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    range_counts = defaultdict(int)

    for sample in result.per_sample:
        total = sum(sample.issue_counts.values())
        for low, high in ranges:
            if low <= total <= high:
                if high == float('inf'):
                    label = f"{low}+"
                elif low == high:
                    label = f"{low}"
                else:
                    label = f"{low}-{high}"
                range_counts[label] += 1
                break

    print(f"{'Issue Count':<15s} {'# Samples':>10s} {'Percentage':>12s}")
    print("-" * 39)

    for low, high in ranges:
        if high == float('inf'):
            label = f"{low}+"
        elif low == high:
            label = f"{low}"
        else:
            label = f"{low}-{high}"
        count = range_counts.get(label, 0)
        pct = count / num_samples * 100
        print(f"{label:<15s} {count:>10d} {pct:>11.1f}%")

    # 4. Per-Issue Distribution Analysis
    print(f"\n--- Per-Issue Distribution Analysis ---")
    
    for issue_type in issue_types:
        # Title for this issue
        short_name = issue_type.replace("_", " ").title()
        print(f"\nDistribution for: {short_name}")
        
        # Reset counts for this issue
        type_range_counts = defaultdict(int)
        
        for sample in result.per_sample:
            # Get count for this specific issue
            count = sample.issue_counts.get(issue_type, 0)
            
            for low, high in ranges:
                if low <= count <= high:
                    if high == float('inf'):
                        label = f"{low}+"
                    elif low == high:
                        label = f"{low}"
                    else:
                        label = f"{low}-{high}"
                    type_range_counts[label] += 1
                    break
        
        print(f"{'Count Range':<15s} {'# Samples':>10s} {'Percentage':>12s}")
        print("-" * 39)
        
        for low, high in ranges:
            if high == float('inf'):
                label = f"{low}+"
            elif low == high:
                label = f"{low}"
            else:
                label = f"{low}-{high}"
            
            count = type_range_counts.get(label, 0)
            pct = count / num_samples * 100
            print(f"{label:<15s} {count:>10d} {pct:>11.1f}%")

    print("=" * 80)


def print_detailed_timeline_statistics(result):
    """Print detailed timeline statistics."""
    from collections import defaultdict
    import numpy as np

    num_samples = result.total_samples
    if num_samples == 0:
        return

    # 5. Timeline Alignment Distribution
    print(f"\n--- Timeline Alignment Distribution ---")

    # Collect per-sample metrics
    coverages = []
    precisions = []
    f1s = []

    for sample in result.per_sample:
        if sample.timeline_alignment:
            coverages.append(sample.timeline_alignment.target_coverage_rate)
            precisions.append(sample.timeline_alignment.pred_precision_rate)
            f1s.append(sample.timeline_alignment.f1_score)

    if coverages:
        print(f"{'Metric':<25s} {'Min':>8s} {'Mean':>8s} {'Max':>8s} {'Std':>8s}")
        print("-" * 61)

        print(f"{'Coverage (Recall)':<25s} "
              f"{min(coverages)*100:>7.2f}% "
              f"{np.mean(coverages)*100:>7.2f}% "
              f"{max(coverages)*100:>7.2f}% "
              f"{np.std(coverages)*100:>7.2f}%")

        print(f"{'Precision':<25s} "
              f"{min(precisions)*100:>7.2f}% "
              f"{np.mean(precisions)*100:>7.2f}% "
              f"{max(precisions)*100:>7.2f}% "
              f"{np.std(precisions)*100:>7.2f}%")

        print(f"{'F1 Score':<25s} "
              f"{min(f1s)*100:>7.2f}% "
              f"{np.mean(f1s)*100:>7.2f}% "
              f"{max(f1s)*100:>7.2f}% "
              f"{np.std(f1s)*100:>7.2f}%")
        
        # 6. Timeline Metrics Distribution
        
        # Ranges: (min_inclusive, max_inclusive, label)
        dist_ranges = [
            (1.0, 1.0, "100% (Perfect)"),
            (0.95, 0.999999, "95-100%"),
            (0.90, 0.949999, "90-95%"),
            (0.80, 0.899999, "80-90%"),
            (0.50, 0.799999, "50-80%"),
            (0.00, 0.499999, "< 50%")
        ]
        
        metrics_to_plot = [
            ("Coverage", coverages),
            ("Precision", precisions),
            ("F1 Score", f1s)
        ]
        
        for metric_name, data in metrics_to_plot:
            print(f"\n--- Timeline {metric_name} Distribution ---")
            counts = defaultdict(int)
            
            for score in data:
                # Handle float precision issues slightly
                for low, high, label in dist_ranges:
                    # Special case for exact 1.0
                    if low == 1.0 and score >= 0.999999: 
                        counts[label] += 1
                        break
                    elif low <= score <= high:
                        counts[label] += 1
                        break
            
            print(f"{metric_name + ' Range':<20s} {'# Samples':>10s} {'Percentage':>12s}")
            print("-" * 44)
            
            for _, _, label in dist_ranges:
                count = counts[label]
                pct = count / num_samples * 100
                print(f"{label:<20s} {count:>10d} {pct:>11.1f}%")

    else:
        print("No timeline alignment data available.")

    print("=" * 80)


if __name__ == "__main__":
    main()
