"""
TAB-focused and pitch-restricted losses for alignment with input_skeleton decoding.

v1 output format: decoder_input_ids[t] should be NOTE_ON_* when labels[t] is TAB_*.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from src.constrained_decoding import PitchToTabMapping, STANDARD_TUNING
from src.tab_dataset import Vocabulary

LossMode = Literal["full", "tab_only", "tab_restricted", "composite"]


def _parse_note_on_pitch_token_str(token_str: str) -> Optional[int]:
    if not token_str.startswith("NOTE_ON_"):
        return None
    try:
        return int(token_str.split("_")[2])
    except (IndexError, ValueError):
        return None


def build_is_tab_mask(output_vocab: Vocabulary, device: torch.device) -> torch.Tensor:
    """vocab_size bool tensor: True iff id is TAB_*."""
    m = torch.zeros(output_vocab.vocab_size, dtype=torch.bool, device=device)
    for tid, tok in output_vocab.id_to_token.items():
        if isinstance(tok, str) and tok.startswith("TAB_"):
            if 0 <= int(tid) < output_vocab.vocab_size:
                m[int(tid)] = True
    return m


def build_pitch_to_tab_token_ids(
    output_vocab: Vocabulary,
    num_frets: int,
    tuning: Optional[List[int]] = None,
) -> Dict[int, List[int]]:
    """Map MIDI pitch -> output vocab ids for playable (string, fret) TAB tokens."""
    tuning = tuning or STANDARD_TUNING
    mapping = PitchToTabMapping.build(tuning, num_frets)
    tab_ids: Dict[tuple, int] = {}
    for token, token_id in output_vocab.token_to_id.items():
        if not token.startswith("TAB_"):
            continue
        try:
            parts = token.split("_")
            string_n, fret = int(parts[1]), int(parts[2])
            tab_ids[(string_n, fret)] = token_id
        except (IndexError, ValueError):
            continue
    out: Dict[int, List[int]] = {}
    for pitch, pairs in mapping.pitch_to_tabs.items():
        ids = [tab_ids[p] for p in pairs if p in tab_ids]
        if ids:
            out[pitch] = ids
    return out


def mask_labels_tab_only(
    labels: torch.Tensor,
    output_vocab: Vocabulary,
    is_tab: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Set non-TAB target positions to ignore_index (-100). PAD should already be -100.

    Args:
        labels: [B, L] shifted labels (post output_ids[:, 1:])
        is_tab: optional precomputed mask from build_is_tab_mask(..., labels.device)
    """
    ignore = -100
    if is_tab is None:
        is_tab = build_is_tab_mask(output_vocab, labels.device)
    out = labels.clone()
    valid = out != ignore
    # Clamp only for indexing; positions with ignore are overwritten by valid mask
    safe = out.clamp(min=0)
    keep = (~valid) | is_tab[safe]
    out[~keep] = ignore
    return out


def _apply_pitch_mask_to_logits_row(
    logits_row: torch.Tensor,
    pitch: int,
    pitch_to_tab_token_ids: Dict[int, List[int]],
    gold_tab_id: Optional[int] = None,
) -> None:
    """
    In-place: invalid TAB logits -> -inf; if pitch unknown, no change.

    If gold_tab_id is set and not in the pitch-legal TAB set, skip masking entirely
    so CE stays finite (dataset pitch vs fingering can disagree; see tab_restricted inf).
    """
    valid_ids = pitch_to_tab_token_ids.get(pitch)
    if not valid_ids:
        return
    if gold_tab_id is not None and int(gold_tab_id) not in valid_ids:
        return
    V = logits_row.shape[-1]
    mask_keep = torch.zeros(V, dtype=torch.bool, device=logits_row.device)
    mask_keep[valid_ids] = True
    logits_row[~mask_keep] = float("-inf")


def cross_entropy_shifted(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def compute_tab_focused_loss(
    logits: torch.Tensor,
    labels_shifted: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    output_vocab: Vocabulary,
    mode: LossMode,
    num_frets: int,
    tuning: Optional[List[int]] = None,
    composite_lambda: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Single forward logits vs shifted labels (same convention as train.py).

    - full: standard CE on all non-padding targets
    - tab_only: CE only on TAB_* targets
    - tab_restricted: tab_only + pitch-masked logits at TAB positions (v1)
    - composite: CE_full + composite_lambda * CE_tab_component
        where CE_tab_component is tab_restricted if mode implies restriction...
        Here: composite = full + lambda * (tab_restricted loss on same logits with tab-only labels + mask)
    """
    tuning = tuning or STANDARD_TUNING
    is_tab = build_is_tab_mask(output_vocab, logits.device)
    pitch_map = build_pitch_to_tab_token_ids(output_vocab, num_frets, tuning)

    if mode == "full":
        return cross_entropy_shifted(logits, labels_shifted, ignore_index)

    labels_tab = mask_labels_tab_only(labels_shifted, output_vocab, is_tab=is_tab)

    if mode in ("tab_only", "tab_restricted") and not (labels_tab != ignore_index).any():
        return logits.sum() * 0.0

    if mode == "tab_only":
        return cross_entropy_shifted(logits, labels_tab, ignore_index)

    if mode == "tab_restricted":
        logits_r = logits.clone()
        B, L, _ = logits_r.shape
        id_to_token = output_vocab.id_to_token
        for b in range(B):
            for t in range(L):
                lab = int(labels_tab[b, t].item())
                if lab == ignore_index:
                    continue
                prev_id = int(decoder_input_ids[b, t].item())
                pitch = _parse_note_on_pitch_token_str(id_to_token.get(prev_id, ""))
                if pitch is None:
                    continue
                _apply_pitch_mask_to_logits_row(
                    logits_r[b, t], pitch, pitch_map, gold_tab_id=lab
                )
        return cross_entropy_shifted(logits_r, labels_tab, ignore_index)

    if mode == "composite":
        loss_full = cross_entropy_shifted(logits, labels_shifted, ignore_index)
        logits_r = logits.clone()
        B, L, _ = logits_r.shape
        id_to_token = output_vocab.id_to_token
        for b in range(B):
            for t in range(L):
                lab = int(labels_tab[b, t].item())
                if lab == ignore_index:
                    continue
                prev_id = int(decoder_input_ids[b, t].item())
                pitch = _parse_note_on_pitch_token_str(id_to_token.get(prev_id, ""))
                if pitch is None:
                    continue
                _apply_pitch_mask_to_logits_row(
                    logits_r[b, t], pitch, pitch_map, gold_tab_id=lab
                )
        if (labels_tab != ignore_index).any():
            loss_tab = cross_entropy_shifted(logits_r, labels_tab, ignore_index)
        else:
            loss_tab = logits.new_tensor(0.0)
        return loss_full + composite_lambda * loss_tab

    raise ValueError(f"Unknown loss mode: {mode}")
