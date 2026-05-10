#!/usr/bin/env python3
"""
Inference script for Fretting-Transformer.
Runs autoregressive generation and computes accuracy metrics.
"""

import os
import torch
import hydra
import random
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy
from src.dataloader import create_dataset, create_dataloader
from src.visualization import *

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(checkpoint_path: str, model, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model_state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  [info] Ignored unexpected keys (aux heads): {unexpected}")
    if missing:
        print(f"  [warn] Missing keys: {missing}")
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
    
    # Set seed for reproducibility
    set_seed(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Modify 2026-03-17: auto-detect the train JSON for vocab building so that
    # inference works with any dataset config (e.g. data=leduc, data=dadagp).
    # Original: hardcoded selected_files_json="data_splits/train_files.json"
    # which fails when data=leduc because the Leduc data_dir has no such file.
    _vocab_json = "data_splits/train_files.json"  # Original default (DadaGP legacy path)
    # Modify 2026-03-20: generalize sibling detection — look for train_files.json in the
    # parent directory of ANY configured selected_files_json, not just test_files.json.
    # Original:
    # if _test_json and "test_files" in str(_test_json):
    #     _candidate = str(_test_json).replace("test_files.json", "train_files.json")
    #     if os.path.exists(_candidate):
    #         _vocab_json = _candidate
    _test_json = cfg.data.get("selected_files_json", None)
    if _test_json:
        _candidate = str(Path(str(_test_json)).parent / "train_files.json")
        if os.path.exists(_candidate):
            _vocab_json = _candidate

    # Build train set dataset for vocab
    train_dataset = create_dataset(
        data_dir=cfg.data.data_dir,
        token_pattern=cfg.data.token_pattern,
        selected_files_json=_vocab_json,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
        output_format=cfg.data.get('output_format', 'v1'),
        max_files=cfg.data.max_files
    )  
    
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
        # max_files=cfg.data.get('max_files', None)
        max_files=None
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
    # breakpoint()

    # original
    # input_vocab_size, output_vocab_size = dataset.get_vocab_sizes()
    
    # modify use train dataset vocab for inference to ensure consistency with training
    input_vocab_size, output_vocab_size = train_dataset.get_vocab_sizes()
    
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
    model.eval()
    
    # Run inference
    print("\n" + "=" * 80)
    print(f"Evaluating on {cfg.data.selected_files_json}")
    print("=" * 80)

    use_constrained = cfg.get("constrained_decoding", False)
    constrained_decoding_mode = cfg.get("constrained_decoding_mode", "input_skeleton")
    constrained_decoding_pitch_mask = cfg.get("constrained_decoding_pitch_mask", True)
    disable_kv_cache = cfg.get("disable_kv_cache", False)
    save_logit_stats = cfg.get("save_logit_stats", False)
    save_structural_logit_stats = cfg.get("save_structural_logit_stats", False)

    output_format = cfg.data.get("output_format", "v1")
    if use_constrained and constrained_decoding_mode == "grammar" and output_format != "v1":
        print(
            f"[warn] grammar constrained decoding is v1-only "
            f"(output_format={output_format} has no NOTE_ON/OFF tokens in output vocab). "
            f"Falling back to unconstrained (C1 ≡ C0 for this format)."
        )
        use_constrained = False

    if use_constrained:
        print("Constrained decoding: enabled")
        print(f"Constrained decoding mode: {constrained_decoding_mode}")
    if save_logit_stats:
        mode_label = "constrained" if use_constrained else "unconstrained (stats_only)"
        print(f"TAB logit stats collection: enabled ({mode_label})")
    if save_structural_logit_stats:
        print("Structural logit stats collection: enabled (Exp A)")
    print(f"KV cache disabled: {disable_kv_cache}")

    metrics, (input_ids, targets, predictions), logit_stats, structural_logit_stats = generate_and_compute_accuracy(
        model=model,
        dataloader=dataloader,
        output_vocab=train_dataset.output_vocab,  # Use train vocab for decoding
        input_vocab=dataset.input_vocab,
        device=device,
        max_length=cfg.training.get('ar_eval_max_length', 1024),
        num_beams=cfg.training.get('ar_eval_num_beams', 1),
        max_batches=None,
        use_constrained_decoding=use_constrained,
        constrained_decoding_mode=constrained_decoding_mode,
        constrained_decoding_pitch_mask=constrained_decoding_pitch_mask,
        num_frets=cfg.data.num_frets,
        disable_kv_cache=disable_kv_cache,
        collect_logit_stats=save_logit_stats,
        collect_structural_logit_stats=save_structural_logit_stats,
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

    import json
    with open(cfg.output_dir / "segment_sources.json", "w") as f:
        json.dump(dataset.segment_sources, f)

    if save_logit_stats and logit_stats:
        logit_stats_file = cfg.output_dir / "logit_stats.pt"
        torch.save(logit_stats, logit_stats_file)
        print(f"\nLogit stats saved: {logit_stats_file}  ({len(logit_stats):,} TAB steps)")

        from src.logit_stats import aggregate_logit_stats, print_logit_stats_summary
        vocab_size = train_dataset.output_vocab.vocab_size
        summary = aggregate_logit_stats(logit_stats, vocab_size=vocab_size)
        print_logit_stats_summary(summary)

        summary_file = cfg.output_dir / "logit_stats_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Logit stats summary: {summary_file}")

    if save_structural_logit_stats and structural_logit_stats:
        struct_stats_file = cfg.output_dir / "structural_logit_stats.pt"
        torch.save(structural_logit_stats, struct_stats_file)
        print(f"\nStructural logit stats saved: {struct_stats_file}  ({len(structural_logit_stats):,} structural steps)")

        from src.logit_stats import aggregate_structural_stats, print_structural_stats_summary
        struct_summary = aggregate_structural_stats(structural_logit_stats)
        print_structural_stats_summary(struct_summary)

        struct_summary_file = cfg.output_dir / "structural_logit_stats_summary.json"
        with open(struct_summary_file, "w") as f:
            json.dump(struct_summary, f, indent=2)
        print(f"Structural logit stats summary: {struct_summary_file}")


    print(f"\nResults:")
    print(f"  Token Accuracy:  {metrics.token_accuracy:.2%}")
    print(f"  Pitch Accuracy:  {metrics.pitch_accuracy:.2%}")
    if metrics.note_token_pitch_accuracy is not None:
        print(f"  Pitch Accuracy (NOTE_ON token):  {metrics.note_token_pitch_accuracy:.2%}")
    print(f"  Tab Accuracy:    {metrics.tab_accuracy:.2%}")
    print(f"  Difficulty:      {metrics.difficulty:.5}")
    print(f"  Total Tokens:    {metrics.total_tokens:,}")
    print(f"  Total Notes:     {metrics.total_notes:,}")


    # ======================================================================
    # VISUALIZATION
    # Show ground truth vs predicted tablature for a few samples
    # ======================================================================

    # print("\n" + "=" * 80)
    # print("Visualizing Predicted vs Ground Truth Tablature")
    # print("=" * 80)

    # num_samples_to_show = min(3, len(predictions))  # Change if needed

    # for idx in range(num_samples_to_show):
    #     print(f"\n--- SAMPLE {idx} ---")
    #     visualize_tab_sample(
    #         gt_ids=targets[idx],
    #         pred_ids=predictions[idx],
    #         output_vocab=dataset.output_vocab,
    #         max_bars=2
    #     )


    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
