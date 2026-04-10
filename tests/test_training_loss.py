"""Tests for TAB-focused training loss (src/training_loss.py)."""

import torch

from src.tab_dataset import Vocabulary
from src.training_loss import (
    build_is_tab_mask,
    compute_tab_focused_loss,
    mask_labels_tab_only,
)


def _tiny_vocab_v1() -> Vocabulary:
    """Minimal output vocab: NOTE_ON_40, TAB_1_0 (pitch 40), TAB_2_5 (wrong pitch for 40)."""
    token_to_id = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "NOTE_ON_40": 4,
        "TAB_1_0": 10,
        "TAB_2_5": 11,
    }
    id_to_token = {v: k for k, v in token_to_id.items()}
    return Vocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        vocab_size=12,
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )


def test_mask_labels_tab_only_sets_non_tab_to_ignore():
    voc = _tiny_vocab_v1()
    labels = torch.tensor([[4, 10, 2, -100]])  # NOTE_ON, TAB, EOS, ignore
    masked = mask_labels_tab_only(labels, voc)
    assert masked[0, 0].item() == -100
    assert masked[0, 1].item() == 10
    assert masked[0, 2].item() == -100
    assert masked[0, 3].item() == -100


def test_build_is_tab_mask():
    voc = _tiny_vocab_v1()
    m = build_is_tab_mask(voc, torch.device("cpu"))
    assert m[10] and m[11]
    assert not m[4]


def test_tab_restricted_finite_when_gold_tab_not_pitch_legal():
    """
    Real data can pair NOTE_ON pitch with a TAB that is not in STANDARD_TUNING geometry.
    Restricted loss must not mask away the gold class (would yield CE inf).
    """
    voc = _tiny_vocab_v1()
    B, L, V = 1, 1, voc.vocab_size
    logits = torch.randn(B, L, V, requires_grad=True)
    # NOTE_ON_40 -> legal TAB in tiny vocab is only TAB_1_0; TAB_2_5 is a different pitch
    labels = torch.tensor([[11]])  # TAB_2_5
    decoder_input_ids = torch.tensor([[4]])  # NOTE_ON_40
    loss = compute_tab_focused_loss(
        logits,
        labels_shifted=labels,
        decoder_input_ids=decoder_input_ids,
        output_vocab=voc,
        mode="tab_restricted",
        num_frets=25,
    )
    assert torch.isfinite(loss).item()
    loss.backward()


def test_tab_restricted_prefers_legal_tab_despite_illegal_high_logit():
    """
    Wrong TAB has highest raw logit; after pitch mask only TAB_1_0 is legal for NOTE_ON_40,
    so CE should match gold TAB_1_0.
    """
    voc = _tiny_vocab_v1()
    B, L, V = 1, 1, voc.vocab_size
    logits = torch.full((B, L, V), -10.0, requires_grad=True)
    # Illegal tab very high
    logits.data[0, 0, 11] = 100.0
    # Gold tab low
    logits.data[0, 0, 10] = 0.0

    labels = torch.tensor([[10]])  # TAB_1_0
    decoder_input_ids = torch.tensor([[4]])  # NOTE_ON_40

    loss = compute_tab_focused_loss(
        logits,
        labels_shifted=labels,
        decoder_input_ids=decoder_input_ids,
        output_vocab=voc,
        mode="tab_restricted",
        num_frets=25,
    )
    loss.backward()
    assert loss.item() < 0.01


def test_tab_only_matches_full_vocab_ce_when_only_tab_supervised():
    voc = _tiny_vocab_v1()
    logits = torch.randn(1, 1, voc.vocab_size, requires_grad=True)
    labels = torch.tensor([[10]])
    dec = torch.tensor([[4]])
    loss = compute_tab_focused_loss(
        logits,
        labels_shifted=labels,
        decoder_input_ids=dec,
        output_vocab=voc,
        mode="tab_only",
        num_frets=25,
    )
    import torch.nn.functional as F

    labels_m = mask_labels_tab_only(labels, voc)
    expected = F.cross_entropy(
        logits.reshape(-1, voc.vocab_size), labels_m.reshape(-1), ignore_index=-100
    )
    assert torch.allclose(loss, expected)
