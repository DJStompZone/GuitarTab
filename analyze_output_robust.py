#!/usr/bin/env python3
"""
Robust analysis CLI for misaligned tablature token sequences.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import create_dataset
from src.metrics import TabAccuracyMetrics, compute_tablature_accuracy
from src.robust_alignment_metrics import compute_robust_alignment_metrics


def _load_tensors(pred_path: Path, target_path: Path, input_path: Path | None):
    predictions = torch.load(pred_path, map_location="cpu")
    targets = torch.load(target_path, map_location="cpu")
    input_ids = None
    if input_path is not None and input_path.exists():
        input_ids = torch.load(input_path, map_location="cpu")
    return predictions, targets, input_ids


def _load_cfg(config_path: Path) -> DictConfig:
    """
    Match inference Hydra composition when given configs/inference.yaml;
    accept a resolved snapshot (e.g. outputs/.../.hydra/config.yaml) as plain load.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config_path = config_path.resolve()
    cfg = OmegaConf.load(config_path)
    if cfg.get("data") is not None and cfg.data.get("data_dir") is not None:
        return cfg
    # Unmerged Hydra entry (defaults not applied by OmegaConf.load alone)
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name)


def _vocabs_like_inference(cfg: DictConfig):
    """
    Same vocabulary source as inference.py: TabDataset after build_vocabulary
    inside TabDataset.__init__. Uses train split JSON for vocab (sibling of
    test json) and output_vocab from that dataset, input_vocab is identical
    mapping when hyperparameters match.
    """
    _vocab_json = "data_splits/train_files.json"
    _test_json = cfg.data.get("selected_files_json", None)
    if _test_json:
        _candidate = str(Path(str(_test_json)).parent / "train_files.json")
        if os.path.exists(_candidate):
            _vocab_json = _candidate

    # max_files=1: vocab does not depend on which files are loaded; speeds up segment prep.
    train_dataset = create_dataset(
        data_dir=cfg.data.data_dir,
        token_pattern=cfg.data.token_pattern,
        selected_files_json=_vocab_json,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
        output_format=cfg.data.get("output_format", "v1"),
        max_files=1,
    )
    input_vocab = train_dataset.input_vocab
    output_vocab = train_dataset.output_vocab
    max_time_shift = int(cfg.data.max_time_shift)
    return input_vocab, output_vocab, max_time_shift


def _write_syntax_summary(path: Path, robust_metrics):
    lines = []
    lines.append("=== Robust Syntax Issue Summary ===")
    lines.append("")
    lines.append("[Target sequence issues]")
    for k, v in sorted(robust_metrics.syntax_issues_target.items()):
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[Prediction sequence issues]")
    for k, v in sorted(robust_metrics.syntax_issues_pred.items()):
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[Dataset aligned metrics]")
    lines.append(f"coverage: {robust_metrics.coverage:.6f}")
    lines.append(f"precision: {robust_metrics.precision:.6f}")
    lines.append(f"f1: {robust_metrics.f1:.6f}")
    lines.append(f"tab_acc_aligned: {robust_metrics.tab_acc_aligned:.6f}")
    lines.append(f"pitch_acc_aligned: {robust_metrics.pitch_acc_aligned:.6f}")
    lines.append("")
    lines.append("[Strict track]")
    lines.append(f"strict_tab_acc: {robust_metrics.strict_tab_acc:.6f}")
    lines.append(f"strict_pitch_acc: {robust_metrics.strict_pitch_acc:.6f}")
    lines.append(f"syntax_penalty_pred: {robust_metrics.syntax_penalty_pred:.6f}")
    lines.append(f"strict_tab_score: {robust_metrics.strict_tab_score:.6f}")
    lines.append(f"strict_pitch_score: {robust_metrics.strict_pitch_score:.6f}")
    lines.append("")
    lines.append("[Normalized track]")
    lines.append(f"valid_event_ratio_target: {robust_metrics.valid_event_ratio_target:.6f}")
    lines.append(f"valid_event_ratio_pred: {robust_metrics.valid_event_ratio_pred:.6f}")
    lines.append(f"normalized_tab_acc: {robust_metrics.normalized_tab_acc:.6f}")
    lines.append(f"normalized_pitch_acc: {robust_metrics.normalized_pitch_acc:.6f}")
    lines.append("")
    lines.append("[Token-class tolerant metrics]")
    for cls_name, values in sorted(robust_metrics.token_class_metrics.items()):
        lines.append(
            f"{cls_name}: P={values['precision']:.6f}, R={values['recall']:.6f}, F1={values['f1']:.6f}, "
            f"matched={int(values['matched'])}/{int(values['target_total'])} target"
        )
    lines.append("")
    lines.append("[Syntax issues per 1k tokens - Target]")
    for k, v in sorted(robust_metrics.syntax_issues_per_1k_tokens_target.items()):
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append("[Syntax issues per 1k tokens - Prediction]")
    for k, v in sorted(robust_metrics.syntax_issues_per_1k_tokens_pred.items()):
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append("[Syntax issues per 1k events - Target]")
    for k, v in sorted(robust_metrics.syntax_issues_per_1k_events_target.items()):
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append("[Syntax issues per 1k events - Prediction]")
    for k, v in sorted(robust_metrics.syntax_issues_per_1k_events_pred.items()):
        lines.append(f"{k}: {v:.4f}")
    if robust_metrics.positional_error_class_rates is not None:
        lines.append("")
        lines.append("[Positional error taxonomy (index-based; T+Tab_3_1+Tab_3_2+correct=100%)]")
        r = robust_metrics.positional_error_class_rates
        c = robust_metrics.positional_error_class_counts
        lines.append(f"correct_rate:  {r['correct_rate']:.6f}  ({c['correct']}/{c['total_target']})")
        lines.append(f"Tab_3_1_rate:  {r['Tab_3_1_rate']:.6f}  ({c['Tab_3_1']})")
        lines.append(f"Tab_3_2_rate:  {r['Tab_3_2_rate']:.6f}  ({c['Tab_3_2']})")
        lines.append(f"T_rate:        {r['T_rate']:.6f}  ({c['T']})")
        lines.append(f"sum:           {r['correct_rate']+r['Tab_3_1_rate']+r['Tab_3_2_rate']+r['T_rate']:.6f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_baseline_metrics_lines(metrics: TabAccuracyMetrics) -> list[str]:
    """Same fields as inference.py after generate_and_compute_accuracy (print block)."""
    lines = [
        "Results:",
        f"  Token Accuracy:  {metrics.token_accuracy:.2%}",
        f"  Pitch Accuracy:  {metrics.pitch_accuracy:.2%}",
    ]
    if metrics.note_token_pitch_accuracy is not None:
        lines.append(
            f"  Pitch Accuracy (NOTE_ON token):  {metrics.note_token_pitch_accuracy:.2%}"
        )
    lines.extend(
        [
            f"  Tab Accuracy:    {metrics.tab_accuracy:.2%}",
            f"  Difficulty:      {metrics.difficulty:.5}",
            f"  Total Tokens:    {metrics.total_tokens:,}",
            f"  Total Notes:     {metrics.total_notes:,}",
        ]
    )
    return lines


def _parse_args():
    parser = argparse.ArgumentParser(description="Robust alignment-aware output analyzer")
    parser.add_argument("folder", nargs="?", help="Folder containing predictions.pt and targets.pt")
    parser.add_argument("--predictions", "-p", type=str, help="Path to predictions.pt")
    parser.add_argument("--targets", "-t", type=str, help="Path to targets.pt")
    parser.add_argument("--input-ids", "-i", type=str, help="Path to input_ids.pt")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        type=str,
        help=(
            "Hydra root (e.g. configs/inference.yaml) or resolved .hydra/config.yaml. "
            "Default: if FOLDER/.hydra/config.yaml exists (same run as the .pt files), use it "
            "so data.output_format matches inference; else configs/inference.yaml."
        ),
    )
    parser.add_argument(
        "--no-hydra-snapshot",
        action="store_true",
        help="Do not auto-use FOLDER/.hydra/config.yaml; require explicit --config.",
    )
    parser.add_argument("--output-dir", "-o", default=None, type=str)
    parser.add_argument("--timeline-tolerance", default=10, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    return parser.parse_args()


def main():
    args = _parse_args()
    # Same as typical `python inference.py` from repo root: relative data_dir / JSON paths.
    os.chdir(PROJECT_ROOT)

    base_folder = Path(args.folder) if args.folder else None
    pred_path = Path(args.predictions) if args.predictions else (
        base_folder / "predictions.pt" if base_folder else None
    )
    target_path = Path(args.targets) if args.targets else (
        base_folder / "targets.pt" if base_folder else None
    )
    input_path = Path(args.input_ids) if args.input_ids else (
        (base_folder / "input_ids.pt") if base_folder else None
    )

    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError(f"predictions not found: {pred_path}")
    if target_path is None or not target_path.exists():
        raise FileNotFoundError(f"targets not found: {target_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif base_folder is not None:
        output_dir = base_folder / "analysis_report_robust"
    else:
        output_dir = Path("analysis_report_robust")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, targets, input_ids = _load_tensors(pred_path, target_path, input_path)

    if args.max_samples is not None:
        predictions = predictions[: args.max_samples]
        targets = targets[: args.max_samples]
        if input_ids is not None:
            input_ids = input_ids[: args.max_samples]

    run_hydra_cfg = (
        (base_folder / ".hydra" / "config.yaml") if base_folder is not None else None
    )
    if (
        not args.no_hydra_snapshot
        and run_hydra_cfg is not None
        and run_hydra_cfg.is_file()
    ):
        config_path = run_hydra_cfg
        print(f"Using Hydra run config (matches saved tensors): {config_path}")
    elif args.config is not None:
        config_path = Path(args.config)
    else:
        config_path = PROJECT_ROOT / "configs" / "inference.yaml"

    cfg = _load_cfg(config_path)
    input_vocab, output_vocab, max_time_shift = _vocabs_like_inference(cfg)

    robust_metrics = compute_robust_alignment_metrics(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        input_ids=input_ids,
        input_vocab=input_vocab if input_ids is not None else None,
        timeline_tolerance=args.timeline_tolerance,
        output_format=str(cfg.data.get("output_format", "v1")),
        max_time_shift=max_time_shift,
    )

    baseline_metrics = compute_tablature_accuracy(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        input_ids=input_ids,
        input_vocab=input_vocab if input_ids is not None else None,
    )

    robust_json_path = output_dir / "robust_metrics.json"
    robust_json_path.write_text(
        json.dumps(
            {
                "robust_metrics": robust_metrics.to_dict(),
                "baseline_metrics": {
                    "token_accuracy": baseline_metrics.token_accuracy,
                    "pitch_accuracy": baseline_metrics.pitch_accuracy,
                    "tab_accuracy": baseline_metrics.tab_accuracy,
                    "difficulty": baseline_metrics.difficulty,
                    "total_tokens": baseline_metrics.total_tokens,
                    "total_notes": baseline_metrics.total_notes,
                    "note_token_pitch_accuracy": baseline_metrics.note_token_pitch_accuracy,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    syntax_summary_path = output_dir / "syntax_issue_summary.txt"
    _write_syntax_summary(syntax_summary_path, robust_metrics)

    diagnostics_path = output_dir / "sample_diagnostics.jsonl"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        for sample in robust_metrics.per_sample:
            f.write(json.dumps(sample.__dict__, ensure_ascii=False) + "\n")

    baseline_txt_path = output_dir / "baseline_metrics.txt"
    baseline_txt_path.write_text(
        "\n".join(_format_baseline_metrics_lines(baseline_metrics)) + "\n",
        encoding="utf-8",
    )

    print("=== Robust analysis completed ===")
    print(f"Config used:       {config_path.resolve()}")
    print(f"Input predictions: {pred_path}")
    print(f"Input targets:     {target_path}")
    if input_path is not None and input_path.exists():
        print(f"Input input_ids:   {input_path}")
    print(f"Output dir:        {output_dir}")
    print("")
    print("[Aligned robust metrics]")
    print(f"coverage:          {robust_metrics.coverage:.4f}")
    print(f"precision:         {robust_metrics.precision:.4f}")
    print(f"f1:                {robust_metrics.f1:.4f}")
    print(f"tab_acc_aligned:   {robust_metrics.tab_acc_aligned:.4f}")
    print(f"pitch_acc_aligned: {robust_metrics.pitch_acc_aligned:.4f}")
    print("")
    print("[Strict track]")
    print(f"strict_tab_acc:    {robust_metrics.strict_tab_acc:.4f}")
    print(f"strict_pitch_acc:  {robust_metrics.strict_pitch_acc:.4f}")
    print(f"syntax_penalty:    {robust_metrics.syntax_penalty_pred:.4f}")
    print(f"strict_tab_score:  {robust_metrics.strict_tab_score:.4f}")
    print(f"strict_pitch_score:{robust_metrics.strict_pitch_score:.4f}")
    print("")
    print("[Normalized track]")
    print(f"valid_ratio_tgt:   {robust_metrics.valid_event_ratio_target:.4f}")
    print(f"valid_ratio_pred:  {robust_metrics.valid_event_ratio_pred:.4f}")
    print(f"norm_tab_acc:      {robust_metrics.normalized_tab_acc:.4f}")
    print(f"norm_pitch_acc:    {robust_metrics.normalized_pitch_acc:.4f}")
    print("")
    if robust_metrics.positional_error_class_rates is not None:
        r = robust_metrics.positional_error_class_rates
        c = robust_metrics.positional_error_class_counts
        print("[Positional error taxonomy (index-based; T+Tab_3_1+Tab_3_2+correct=100%)]")
        print(f"correct_rate:  {r['correct_rate']:.4f}  ({c['correct']}/{c['total_target']})")
        print(f"Tab_3_1_rate:  {r['Tab_3_1_rate']:.4f}  ({c['Tab_3_1']})")
        print(f"Tab_3_2_rate:  {r['Tab_3_2_rate']:.4f}  ({c['Tab_3_2']})")
        print(f"T_rate:        {r['T_rate']:.4f}  ({c['T']})")
        print(f"sum:           {r['correct_rate']+r['Tab_3_1_rate']+r['Tab_3_2_rate']+r['T_rate']:.4f}")
        print("")
    print("[Baseline metrics — same as inference.py TabAccuracyMetrics]")
    print("")
    for line in _format_baseline_metrics_lines(baseline_metrics):
        print(line)
    print("")
    print(f"Saved: {robust_json_path}")
    print(f"Saved: {syntax_summary_path}")
    print(f"Saved: {diagnostics_path}")
    print(f"Saved: {baseline_txt_path}")


if __name__ == "__main__":
    main()
