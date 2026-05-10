#!/usr/bin/env python3
"""
Training script for v2+aux model: v2 token sequence (TAB + TIME_SHIFT) with an
auxiliary pitch-classification head on the decoder hidden states.

The auxiliary head provides explicit pitch supervision at each TAB token position,
decoupling pitch supervision from the token format question:
  "Is v1 better because of NOTE_ON tokens, or because of pitch supervision?"

Loss = (1 - pitch_loss_weight) * CE(TAB tokens) + pitch_loss_weight * CE(pitch)

During inference, the aux head is ignored; the checkpoint is fully compatible with
the standard v2 inference pipeline (inference.py with data.output_format=v2).
"""

import os
import random
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.dadagp_parser import (
    NoteOnEvent,
    TabEvent,
    TimeShiftEvent,
    dadagp_to_events,
    parse_dadagp_file,
)
from src.dataloader import create_dataset
from src.model import FrettingTransformer
from src.tab_dataset import (
    TabDataset,
    Vocabulary,
    _remap_bar_positions_after_output_filter,
    collate_fn,
    events_to_ids,
    find_split_bar_near_length,
)
from src.training_logger import TrainingLogger


# ============================================================================
# Seed
# ============================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Dataset with auxiliary pitch labels
# ============================================================================


class V2AuxTabDataset(TabDataset):
    """
    Extends TabDataset (v2 format) to also return per-step pitch labels.

    For each output step, pitch_labels[t] is:
      - The MIDI pitch of the note (from input NOTE_ON) if output_ids[t] is a TAB token
      - -100 (ignore_index) for TIME_SHIFT, BOS, EOS, PAD positions
    """

    def __init__(self, token_files: list[str], **kwargs):
        kwargs["output_format"] = "v2"
        super().__init__(token_files, **kwargs)
        # self.segments is now list of (input_ids, output_ids, pitch_labels)

    def _prepare_segments(self) -> list:
        all_segments: list = []
        self.segment_sources: list[str] = []

        for token_file in tqdm(self.token_files, desc="Processing files (v2+aux)"):
            try:
                dadagp_tokens = parse_dadagp_file(token_file)
                input_events, full_output_events, bar_positions = dadagp_to_events(
                    dadagp_tokens, max_time_shift=self.max_time_shift
                )

                # Filter output to TAB + TIME_SHIFT (v2 format)
                filtered_output_events = [
                    ev
                    for ev in full_output_events
                    if isinstance(ev, (TabEvent, TimeShiftEvent))
                ]

                bar_positions = _remap_bar_positions_after_output_filter(
                    bar_positions,
                    full_output_events,
                    lambda ev: isinstance(ev, (TabEvent, TimeShiftEvent)),
                )

                # Build pitch label for each TAB token in order.
                # full_output_events layout: ..., NOTE_ON_p, TAB_s_f, ..., NOTE_OFF, TIME_SHIFT, ...
                # Each TAB is immediately preceded by its NOTE_ON in the full stream.
                pitch_per_tab: list[int] = []
                last_pitch: Optional[int] = None
                for ev in full_output_events:
                    if isinstance(ev, NoteOnEvent):
                        last_pitch = ev.pitch
                    elif isinstance(ev, TabEvent):
                        pitch_per_tab.append(last_pitch if last_pitch is not None else -100)
                        last_pitch = None

                # Create pitch_labels parallel to filtered_output_events
                tab_idx = 0
                pitch_labels: list[int] = []
                for ev in filtered_output_events:
                    if isinstance(ev, TabEvent):
                        p = pitch_per_tab[tab_idx] if tab_idx < len(pitch_per_tab) else -100
                        pitch_labels.append(p)
                        tab_idx += 1
                    else:
                        pitch_labels.append(-100)

                # Convert to IDs
                input_ids = events_to_ids(input_events, self.input_vocab)
                output_ids = events_to_ids(filtered_output_events, self.output_vocab)

                # Split into bar-aligned segments
                segments = self._split_with_pitch(
                    input_ids, output_ids, pitch_labels, bar_positions
                )

                for inp, out, pit in segments:
                    out = [self.output_vocab.bos_id] + out + [self.output_vocab.eos_id]
                    pit = [-100] + pit + [-100]  # BOS and EOS → ignore
                    if len(out) > self.max_sequence_length:
                        out = out[: self.max_sequence_length - 1] + [self.output_vocab.eos_id]
                        pit = pit[: self.max_sequence_length - 1] + [-100]
                    all_segments.append((inp, out, pit))
                    self.segment_sources.append(token_file)

            except Exception as e:
                print(f"Error processing {token_file}: {e}")
                continue

        return all_segments

    def _split_with_pitch(
        self,
        input_ids: list[int],
        output_ids: list[int],
        pitch_labels: list[int],
        bar_positions: list[tuple[int, int]],
    ) -> list[tuple[list[int], list[int], list[int]]]:
        segments: list = []
        if not bar_positions:
            raise ValueError("No bar positions found")

        current_bar_idx = 0
        while current_bar_idx < len(bar_positions):
            split_bar_idx = find_split_bar_near_length(
                bar_positions, self.max_sequence_length, current_bar_idx
            )
            if split_bar_idx is None:
                break

            start_inp = bar_positions[current_bar_idx][0]
            start_out = bar_positions[current_bar_idx][1]

            if split_bar_idx < len(bar_positions):
                end_inp = bar_positions[split_bar_idx][0]
                end_out = bar_positions[split_bar_idx][1]
            else:
                end_inp = len(input_ids)
                end_out = len(output_ids)

            inp_seg = input_ids[start_inp:end_inp]
            out_seg = output_ids[start_out:end_out]
            pit_seg = pitch_labels[start_out:end_out]

            if len(inp_seg) > self.max_sequence_length // 10:
                if (
                    self.input_vocab.unk_id not in inp_seg
                    and self.output_vocab.unk_id not in out_seg
                ):
                    segments.append((inp_seg, out_seg, pit_seg))

            overlap = (split_bar_idx - current_bar_idx) // 2
            current_bar_idx += max(1, overlap)

        return segments

    def __getitem__(self, idx: int):
        inp, out, pit = self.segments[idx]
        return (
            np.array(inp, dtype=np.int64),
            np.array(out, dtype=np.int64),
            np.array(pit, dtype=np.int64),
        )


def collate_fn_v2_aux(batch, input_pad_id: int = 0, output_pad_id: int = 0):
    """Collate (input_ids, output_ids, pitch_labels) triples with padding."""
    inp_list = [item[0] for item in batch]
    out_list = [item[1] for item in batch]
    pit_list = [item[2] for item in batch]

    max_inp = max(len(x) for x in inp_list)
    max_out = max(len(x) for x in out_list)
    B = len(batch)

    padded_inp = np.full((B, max_inp), input_pad_id, dtype=np.int64)
    padded_out = np.full((B, max_out), output_pad_id, dtype=np.int64)
    padded_pit = np.full((B, max_out), -100, dtype=np.int64)
    inp_mask = np.zeros((B, max_inp), dtype=np.int64)
    out_mask = np.zeros((B, max_out), dtype=np.int64)

    for i, (inp, out, pit) in enumerate(zip(inp_list, out_list, pit_list)):
        padded_inp[i, : len(inp)] = inp
        padded_out[i, : len(out)] = out
        padded_pit[i, : len(pit)] = pit
        inp_mask[i, : len(inp)] = 1
        out_mask[i, : len(out)] = 1

    return {
        "input_ids": torch.from_numpy(padded_inp),
        "output_ids": torch.from_numpy(padded_out),
        "pitch_labels": torch.from_numpy(padded_pit),
        "attention_mask": torch.from_numpy(inp_mask),
        "decoder_attention_mask": torch.from_numpy(out_mask),
    }


# ============================================================================
# Model with auxiliary pitch head
# ============================================================================


class FrettingTransformerV2Aux(FrettingTransformer):
    """
    FrettingTransformer + auxiliary pitch-classification head.

    The pitch head is applied to the decoder's last hidden state at each step.
    At TAB positions, it predicts the MIDI pitch; other positions are ignored (label=-100).

    During inference, load the checkpoint with standard FrettingTransformer
    (the pitch_head weights will be ignored via strict=False or simply skipped).
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        model_config: dict,
        max_pitch: int = 127,
        pitch_loss_weight: float = 0.5,
    ):
        super().__init__(input_vocab_size, output_vocab_size, model_config)
        d_model: int = self.model.config.d_model
        self.pitch_head = nn.Linear(d_model, max_pitch + 1)
        self.pitch_loss_weight = pitch_loss_weight
        self.max_pitch = max_pitch

    def forward_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        pitch_labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass that returns combined loss.

        pitch_labels: [B, L_dec] aligned with *labels* (already shifted by 1).
                      -100 at non-TAB positions → ignored by cross_entropy.
        """
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        tab_loss: torch.Tensor = outputs.loss  # type: ignore[assignment]

        # Last decoder hidden state: [B, L_dec, d_model]
        decoder_hidden = outputs.decoder_hidden_states[-1]  # type: ignore[index]
        pitch_logits = self.pitch_head(decoder_hidden)  # [B, L_dec, max_pitch+1]

        if pitch_labels is not None:
            pitch_loss = F.cross_entropy(
                pitch_logits.reshape(-1, self.max_pitch + 1),
                pitch_labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            pitch_loss = torch.zeros(1, device=tab_loss.device).squeeze()

        alpha = 1.0 - self.pitch_loss_weight
        beta = self.pitch_loss_weight
        total_loss = alpha * tab_loss + beta * pitch_loss

        return total_loss, tab_loss.detach(), pitch_loss.detach()


# ============================================================================
# DataLoader factory
# ============================================================================


def _make_dataloader(
    token_files: list[str],
    cfg: DictConfig,
    shuffle: bool,
    batch_size: int,
) -> tuple[DataLoader, V2AuxTabDataset]:
    ds = V2AuxTabDataset(
        token_files=token_files,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
    )
    inp_pad, out_pad = ds.get_pad_ids()
    from functools import partial

    cfn = partial(collate_fn_v2_aux, input_pad_id=inp_pad, output_pad_id=out_pad)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=cfn,
    )
    return loader, ds


def create_dataloaders(cfg: DictConfig):
    import json

    split_dir = Path(cfg.data.selected_files_json).parent
    train_json = cfg.data.selected_files_json
    val_json = str(split_dir / "val_files.json")
    test_json = str(split_dir / "test_files.json")

    def _load_files(json_path, data_dir, pattern):
        import glob as _glob

        with open(json_path) as f:
            selected = set(json.load(f))
        all_files = sorted(_glob.glob(str(Path(data_dir) / pattern), recursive=True))
        return [
            fp
            for fp in all_files
            if fp.endswith(".tokens.txt") and fp[: -len(".tokens.txt")] in selected
        ]

    print("Loading file lists ...")
    train_files = _load_files(train_json, cfg.data.data_dir, cfg.data.token_pattern)
    val_files = _load_files(val_json, cfg.data.data_dir, cfg.data.token_pattern)
    test_files = _load_files(test_json, cfg.data.data_dir, cfg.data.token_pattern)

    if cfg.data.get("max_files"):
        train_files = train_files[: cfg.data.max_files]

    print(f"Files: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    train_loader, train_ds = _make_dataloader(
        train_files, cfg, shuffle=True, batch_size=cfg.data.batch_size
    )
    val_loader, _ = _make_dataloader(
        val_files, cfg, shuffle=False, batch_size=cfg.training.eval_batch_size
    )
    test_loader, _ = _make_dataloader(
        test_files, cfg, shuffle=False, batch_size=cfg.training.eval_batch_size
    )
    return train_loader, val_loader, test_loader, train_ds


# ============================================================================
# Optimizer
# ============================================================================


def create_optimizer(model: nn.Module, cfg: DictConfig):
    opt_cfg = cfg.training.optimizer
    if opt_cfg.name == "adafactor":
        from transformers import Adafactor

        return Adafactor(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            scale_parameter=opt_cfg.scale_parameter,
            relative_step=opt_cfg.relative_step,
            warmup_init=opt_cfg.warmup_init,
        )
    elif opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {opt_cfg.name}")


# ============================================================================
# Train / eval epoch
# ============================================================================


def train_epoch(model: FrettingTransformerV2Aux, loader: DataLoader, optimizer, device: str, epoch: int, cfg: DictConfig):
    model.train()
    total_loss = total_tab = total_pitch = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["output_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        pitch_labels = batch["pitch_labels"].to(device)

        # Shift for teacher forcing
        decoder_input_ids = labels[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
        decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()
        pitch_labels = pitch_labels[:, 1:].contiguous()  # align with labels

        labels = labels.clone()
        labels[labels == 0] = -100

        loss, tab_loss, pitch_loss = model.forward_train(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            pitch_labels=pitch_labels,
        )

        loss.backward()
        if cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_tab += tab_loss.item()
        total_pitch += pitch_loss.item()
        n += 1
        pbar.set_postfix({"loss": loss.item(), "tab": tab_loss.item(), "pitch": pitch_loss.item()})

    return total_loss / n, total_tab / n, total_pitch / n


def evaluate(model: FrettingTransformerV2Aux, loader: DataLoader, device: str):
    model.eval()
    total_loss = total_tab = total_pitch = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["output_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)
            pitch_labels = batch["pitch_labels"].to(device)

            decoder_input_ids = labels[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()
            pitch_labels = pitch_labels[:, 1:].contiguous()

            labels = labels.clone()
            labels[labels == 0] = -100

            loss, tab_loss, pitch_loss = model.forward_train(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                pitch_labels=pitch_labels,
            )

            total_loss += loss.item()
            total_tab += tab_loss.item()
            total_pitch += pitch_loss.item()
            n += 1

    return total_loss / n, total_tab / n, total_pitch / n


# ============================================================================
# Main
# ============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="train_v2_aux")
def main(cfg: DictConfig):
    print("=" * 80)
    print("Fretting-Transformer v2+aux Training")
    print("=" * 80)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger = TrainingLogger(log_file=output_dir / "training_log.json")

    train_loader, val_loader, test_loader, train_ds = create_dataloaders(cfg)

    input_vocab_size, output_vocab_size = train_ds.get_vocab_sizes()
    print(f"Vocab: input={input_vocab_size}, output={output_vocab_size}")

    model = FrettingTransformerV2Aux(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        model_config=OmegaConf.to_container(cfg.model),
        max_pitch=cfg.data.max_pitch,
        pitch_loss_weight=cfg.get("pitch_loss_weight", 0.5),
    ).to(device)

    optimizer = create_optimizer(model, cfg)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\n{'='*80}\nEpoch {epoch}/{cfg.training.num_epochs}\n{'='*80}")

        tr_loss, tr_tab, tr_pitch = train_epoch(model, train_loader, optimizer, device, epoch, cfg)
        print(f"Train  loss={tr_loss:.4f}  tab={tr_tab:.4f}  pitch={tr_pitch:.4f}")

        vl_loss, vl_tab, vl_pitch = evaluate(model, val_loader, device)
        print(f"Val    loss={vl_loss:.4f}  tab={vl_tab:.4f}  pitch={vl_pitch:.4f}")

        logger.log_epoch(epoch=epoch, train_loss=tr_loss, val_loss=vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": tr_loss,
                    "val_loss": vl_loss,
                    "config": OmegaConf.to_container(cfg),
                },
                ckpt_path,
            )
            print(f"Saved best model → {ckpt_path}")

        chk_n = cfg.training.get("checkpoint_every_n_epochs", 0)
        if chk_n > 0 and epoch % chk_n == 0:
            ep_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": tr_loss,
                    "val_loss": vl_loss,
                },
                ep_path,
            )
            print(f"Saved checkpoint → {ep_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
