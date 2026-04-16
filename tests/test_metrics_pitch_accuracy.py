import torch

from src.metrics import compute_tablature_accuracy
from src.tab_dataset import build_vocabulary


def _token(vocab, token_str: str) -> int:
    return vocab.token_to_id[token_str]


def test_v1_pitch_accuracy_tab_pitch_vs_target_tab():
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=90,
        max_time_shift=120,
        num_frets=25,
        output_format="v1",
    )
    # Shifted tuning (+1): pitch 41 should map to TAB_1_0
    input_ids = torch.tensor(
        [[_token(input_vocab, "NOTE_ON_41"), _token(input_vocab, "NOTE_OFF_41"), input_vocab.eos_id]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[
            _token(output_vocab, "NOTE_ON_41"),
            _token(output_vocab, "TAB_1_0"),
            _token(output_vocab, "NOTE_OFF_41"),
            output_vocab.eos_id,
        ]],
        dtype=torch.long,
    )
    predictions = torch.tensor(
        [[
            _token(output_vocab, "NOTE_ON_41"),
            _token(output_vocab, "TAB_1_1"),  # wrong pitch vs target TAB under same tuning
            _token(output_vocab, "NOTE_OFF_41"),
            output_vocab.eos_id,
        ]],
        dtype=torch.long,
    )

    metrics = compute_tablature_accuracy(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        input_ids=input_ids,
        input_vocab=input_vocab,
    )
    assert metrics.total_notes == 1
    assert metrics.pitch_accuracy == 0.0
    assert metrics.note_token_pitch_accuracy == 1.0
    assert metrics.tab_accuracy == 0.0


def test_v2_pitch_accuracy_tab_pitch_vs_target_tab():
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=90,
        max_time_shift=120,
        num_frets=25,
        output_format="v2",
    )
    input_ids = torch.tensor(
        [[_token(input_vocab, "NOTE_ON_41"), _token(input_vocab, "NOTE_OFF_41"), input_vocab.eos_id]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[_token(output_vocab, "TAB_1_0"), output_vocab.eos_id]],
        dtype=torch.long,
    )
    predictions = torch.tensor(
        [[_token(output_vocab, "TAB_1_1"), output_vocab.eos_id]],
        dtype=torch.long,
    )

    metrics = compute_tablature_accuracy(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        input_ids=input_ids,
        input_vocab=input_vocab,
    )
    assert metrics.total_notes == 1
    assert metrics.pitch_accuracy == 0.0
    assert metrics.note_token_pitch_accuracy is None
    assert metrics.tab_accuracy == 0.0


def test_v1_same_pitch_different_tab_tokens_counts_pitch_not_tab():
    """E4: TAB_6_0 vs TAB_5_5 under standard tuning."""
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=90,
        max_time_shift=120,
        num_frets=25,
        output_format="v1",
    )
    input_ids = torch.tensor(
        [[_token(input_vocab, "NOTE_ON_64"), input_vocab.eos_id]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[
            _token(output_vocab, "NOTE_ON_64"),
            _token(output_vocab, "TAB_6_0"),
            output_vocab.eos_id,
        ]],
        dtype=torch.long,
    )
    predictions = torch.tensor(
        [[
            _token(output_vocab, "NOTE_ON_64"),
            _token(output_vocab, "TAB_5_5"),
            output_vocab.eos_id,
        ]],
        dtype=torch.long,
    )
    metrics = compute_tablature_accuracy(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        input_ids=input_ids,
        input_vocab=input_vocab,
    )
    assert metrics.total_notes == 1
    assert metrics.pitch_accuracy == 1.0
    assert metrics.tab_accuracy == 0.0
