#!/usr/bin/env python3
"""
Exp 5: Input perturbation robustness evaluation.

Introduces controlled noise to the input token sequence and evaluates how
tab accuracy degrades under three perturbation types:
  (a) NOTE_ON pitch ±k semitone jitter (fraction p of events)
  (b) TIME_SHIFT ±k ticks noise (fraction p of events)
  (c) Random drop of NOTE_OFF tokens (fraction p)

For each model (M1/M2/M3) × condition (C2/C3) × perturbation type × noise level,
runs inference and saves results for the degradation curve analysis.

Usage:
    python scripts/run_inference_with_perturbation.py \
        --checkpoint ckpt/dadagp_v1_300epcohs_weight_decay/best_model.pt \
        --format v1 \
        --condition C3 \
        --perturb_type pitch_jitter \
        --perturb_frac 0.05 \
        --perturb_level 1 \
        --output_dir outputs/robustness/M1_C3_pitch_jitter_p05_k1
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.dataloader import create_dataset, create_dataloader
from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy, infer_tuning_from_input_output, GUITAR_TUNING
from src.robust_alignment_metrics import compute_robust_alignment_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perturb_input_ids(
    input_ids: torch.Tensor,
    input_vocab,
    perturb_type: str,
    perturb_frac: float,
    perturb_level: int,
    rng: random.Random,
) -> torch.Tensor:
    """
    Apply noise to a batch of input token ID sequences.

    Args:
        input_ids: [B, L] LongTensor
        input_vocab: Vocabulary with id_to_token mapping
        perturb_type: "pitch_jitter" | "time_noise" | "drop_note_off"
        perturb_frac: fraction of eligible tokens to perturb (0..1)
        perturb_level: magnitude — semitones for pitch_jitter, ticks for time_noise
        rng: seeded random.Random for reproducibility

    Returns:
        Perturbed input_ids (new tensor, original unchanged)
    """
    ids = input_ids.clone()

    for b in range(ids.shape[0]):
        row = ids[b].tolist()
        new_row = []
        for tok_id in row:
            tok = input_vocab.id_to_token.get(tok_id, "")
            if perturb_type == "pitch_jitter" and tok.startswith("NOTE_ON_"):
                if rng.random() < perturb_frac:
                    pitch = int(tok.split("_")[2])
                    delta = rng.choice([-perturb_level, perturb_level])
                    new_pitch = max(0, min(127, pitch + delta))
                    new_tok = f"NOTE_ON_{new_pitch}"
                    new_id = input_vocab.token_to_id.get(new_tok, tok_id)
                    new_row.append(new_id)
                    continue

            elif perturb_type == "time_noise" and tok.startswith("TIME_SHIFT_"):
                if rng.random() < perturb_frac:
                    ticks = int(tok.split("_")[2])
                    delta = rng.choice([-perturb_level, perturb_level])
                    new_ticks = max(1, ticks + delta)
                    new_tok = f"TIME_SHIFT_{new_ticks}"
                    new_id = input_vocab.token_to_id.get(new_tok, tok_id)
                    new_row.append(new_id)
                    continue

            elif perturb_type == "drop_note_off" and tok.startswith("NOTE_OFF_"):
                if rng.random() < perturb_frac:
                    continue  # drop this token

            new_row.append(tok_id)

        # Pad back to original length
        orig_len = ids.shape[1]
        new_row = new_row[:orig_len]
        while len(new_row) < orig_len:
            new_row.append(input_vocab.pad_id)
        ids[b] = torch.tensor(new_row, dtype=ids.dtype)

    return ids


def load_model(checkpoint_path: str, input_vocab_size: int, output_vocab_size: int, model_config: dict, device: str):
    model = FrettingTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        model_config=model_config,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    model.eval()
    return model


def run_perturbed_inference(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load config via Hydra
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose("inference")
    cfg = OmegaConf.to_container(cfg, resolve=True)

    from omegaconf import OmegaConf as OC
    cfg = OC.create(cfg)

    # Override data format and files
    cfg.data.output_format = args.format
    cfg.data.selected_files_json = args.test_json

    # Build vocab from train split
    vocab_json = str(Path(args.test_json).parent / "train_files.json")
    train_ds = create_dataset(
        data_dir=cfg.data.data_dir,
        token_pattern=cfg.data.token_pattern,
        selected_files_json=vocab_json,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
        output_format=args.format,
        max_files=1,
    )
    input_vocab = train_ds.input_vocab
    output_vocab = train_ds.output_vocab

    test_ds = create_dataset(
        data_dir=cfg.data.data_dir,
        token_pattern=cfg.data.token_pattern,
        selected_files_json=args.test_json,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
        output_format=args.format,
        max_files=None,
    )
    test_loader = create_dataloader(
        test_ds,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    input_vocab_size, output_vocab_size = train_ds.get_vocab_sizes()
    from omegaconf import OmegaConf as OC2
    model_config = OC2.to_container(cfg.model)
    model = load_model(args.checkpoint, input_vocab_size, output_vocab_size, model_config, device)

    # Setup perturbation RNG
    rng = random.Random(args.seed + 1)

    # Inject perturbation at dataloader level using a custom wrapper
    class PerturbedLoader:
        def __init__(self, loader, vocab, ptype, pfrac, plevel, prng):
            self._loader = loader
            self._vocab = vocab
            self._ptype = ptype
            self._pfrac = pfrac
            self._plevel = plevel
            self._rng = prng

        def __iter__(self):
            for batch in self._loader:
                batch = dict(batch)
                batch["input_ids"] = perturb_input_ids(
                    batch["input_ids"], self._vocab,
                    self._ptype, self._pfrac, self._plevel, self._rng
                )
                # Recompute attention mask (some tokens may have been dropped → length changed)
                batch["attention_mask"] = (batch["input_ids"] != self._vocab.pad_id).long()
                yield batch

        def __len__(self):
            return len(self._loader)

    if args.perturb_type != "none":
        loader = PerturbedLoader(
            test_loader, input_vocab,
            args.perturb_type, args.perturb_frac, args.perturb_level, rng
        )
    else:
        loader = test_loader

    use_constrained = args.condition in ("C2", "C3")
    cmode = "input_skeleton" if use_constrained else "input_skeleton"
    pitch_mask = (args.condition == "C3")

    metrics, (input_ids, targets, predictions), _, _ = generate_and_compute_accuracy(
        model=model,
        dataloader=loader,
        output_vocab=output_vocab,
        input_vocab=input_vocab,
        device=device,
        max_length=cfg.training.get("ar_eval_max_length", 1024),
        use_constrained_decoding=use_constrained,
        constrained_decoding_mode=cmode,
        constrained_decoding_pitch_mask=pitch_mask,
        num_frets=cfg.data.num_frets,
        max_batches=None,
    )

    robust = compute_robust_alignment_metrics(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        input_ids=input_ids,
        input_vocab=input_vocab,
        timeline_tolerance=10,
        output_format=args.format,
        max_time_shift=int(cfg.data.max_time_shift),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model_tag,
        "condition": args.condition,
        "format": args.format,
        "perturb_type": args.perturb_type,
        "perturb_frac": args.perturb_frac,
        "perturb_level": args.perturb_level,
        "tab_accuracy": metrics.tab_accuracy,
        "pitch_accuracy": metrics.pitch_accuracy,
        "token_accuracy": metrics.token_accuracy,
        "coverage": robust.coverage,
        "tab_acc_aligned": robust.tab_acc_aligned,
        "strict_tab_acc": robust.strict_tab_acc,
        "error_class_counts": robust.error_class_counts,
        "error_class_marginal_rates": robust.error_class_marginal_rates,
        "error_class_conditional_rates": robust.error_class_conditional_rates,
    }

    torch.save(input_ids, out_dir / "input_ids.pt")
    torch.save(targets, out_dir / "targets.pt")
    torch.save(predictions, out_dir / "predictions.pt")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSummary: {summary}")
    print(f"Saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--format", choices=["v1", "v2"], required=True)
    p.add_argument("--condition", choices=["C0", "C2", "C3"], default="C3")
    p.add_argument("--model_tag", default="M?")
    p.add_argument("--test_json", default="data_splits/test_files.json")
    p.add_argument("--perturb_type", choices=["none", "pitch_jitter", "time_noise", "drop_note_off"], default="none")
    p.add_argument("--perturb_frac", type=float, default=0.0, help="Fraction of eligible tokens to perturb")
    p.add_argument("--perturb_level", type=int, default=0, help="Perturbation magnitude (semitones or ticks)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    run_perturbed_inference(args)
