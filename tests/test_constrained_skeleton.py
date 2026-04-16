import torch

from src.constrained_decoding import (
    build_steps_from_input_ids,
    create_constrained_processor,
)
from src.tab_dataset import build_vocabulary


def _token(vocab, token_str: str) -> int:
    return vocab.token_to_id[token_str]


def test_build_steps_from_input_ids_inserts_tab_after_note_on():
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=80,
        max_time_shift=120,
        num_frets=25,
        output_format="v1",
    )
    input_ids = torch.tensor(
        [
            _token(input_vocab, "NOTE_ON_64"),
            _token(input_vocab, "NOTE_OFF_64"),
            _token(input_vocab, "TIME_SHIFT_120"),
            input_vocab.pad_id,
        ],
        dtype=torch.long,
    )

    steps = build_steps_from_input_ids(input_ids, input_vocab, output_vocab)
    assert [step.kind for step in steps] == [
        "FIXED_TOKEN",
        "TAB_FOR_PITCH",
        "FIXED_TOKEN",
        "FIXED_TOKEN",
    ]
    assert steps[0].token_id == _token(output_vocab, "NOTE_ON_64")
    assert steps[1].pitch == 64
    assert steps[2].token_id == _token(output_vocab, "NOTE_OFF_64")
    assert steps[3].token_id == _token(output_vocab, "TIME_SHIFT_120")


def test_build_steps_from_input_ids_v2_skips_note_on_off_fixed_tokens():
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=80,
        max_time_shift=120,
        num_frets=25,
        output_format="v2",
    )
    input_ids = torch.tensor(
        [
            _token(input_vocab, "NOTE_ON_64"),
            _token(input_vocab, "NOTE_OFF_64"),
            _token(input_vocab, "TIME_SHIFT_120"),
            input_vocab.pad_id,
        ],
        dtype=torch.long,
    )

    steps = build_steps_from_input_ids(input_ids, input_vocab, output_vocab)
    assert [step.kind for step in steps] == [
        "TAB_FOR_PITCH",
        "FIXED_TOKEN",
    ]
    assert steps[0].pitch == 64
    assert steps[1].token_id == _token(output_vocab, "TIME_SHIFT_120")


def test_input_skeleton_processor_enforces_exact_structure_and_eos():
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=80,
        max_time_shift=120,
        num_frets=25,
        output_format="v1",
    )
    input_ids = torch.tensor(
        [[
            _token(input_vocab, "NOTE_ON_64"),
            _token(input_vocab, "NOTE_OFF_64"),
            _token(input_vocab, "TIME_SHIFT_120"),
            input_vocab.pad_id,
        ]],
        dtype=torch.long,
    )

    processor = create_constrained_processor(
        input_ids=input_ids,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        mode="input_skeleton",
        num_frets=25,
    )
    p = processor.processors[0]

    scores = torch.zeros(output_vocab.vocab_size)
    mask = p._compute_valid_token_mask(scores.device)
    assert mask[_token(output_vocab, "NOTE_ON_64")]
    assert int(mask.sum()) == 1

    p.update_state(_token(output_vocab, "NOTE_ON_64"))
    mask = p._compute_valid_token_mask(scores.device)
    valid_tabs = [i for i, ok in enumerate(mask.tolist()) if ok]
    assert len(valid_tabs) >= 1
    assert _token(output_vocab, "TAB_1_24") in valid_tabs

    p.update_state(_token(output_vocab, "TAB_1_24"))
    mask = p._compute_valid_token_mask(scores.device)
    assert mask[_token(output_vocab, "NOTE_OFF_64")]
    assert int(mask.sum()) == 1

    p.update_state(_token(output_vocab, "NOTE_OFF_64"))
    mask = p._compute_valid_token_mask(scores.device)
    assert mask[_token(output_vocab, "TIME_SHIFT_120")]
    assert int(mask.sum()) == 1

    p.update_state(_token(output_vocab, "TIME_SHIFT_120"))
    mask = p._compute_valid_token_mask(scores.device)
    assert mask[output_vocab.eos_id]
    assert int(mask.sum()) == 1


def test_input_skeleton_processor_v2_enforces_tab_then_time_shift_then_eos():
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=80,
        max_time_shift=120,
        num_frets=25,
        output_format="v2",
    )
    input_ids = torch.tensor(
        [[
            _token(input_vocab, "NOTE_ON_64"),
            _token(input_vocab, "NOTE_OFF_64"),
            _token(input_vocab, "TIME_SHIFT_120"),
            input_vocab.pad_id,
        ]],
        dtype=torch.long,
    )

    processor = create_constrained_processor(
        input_ids=input_ids,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        mode="input_skeleton",
        num_frets=25,
    )
    p = processor.processors[0]

    scores = torch.zeros(output_vocab.vocab_size)
    mask = p._compute_valid_token_mask(scores.device)
    valid_tabs = [i for i, ok in enumerate(mask.tolist()) if ok]
    assert len(valid_tabs) >= 1
    assert _token(output_vocab, "TAB_1_24") in valid_tabs
    assert _token(output_vocab, "TIME_SHIFT_120") not in valid_tabs

    p.update_state(_token(output_vocab, "TAB_1_24"))
    mask = p._compute_valid_token_mask(scores.device)
    assert mask[_token(output_vocab, "TIME_SHIFT_120")]
    assert int(mask.sum()) == 1

    p.update_state(_token(output_vocab, "TIME_SHIFT_120"))
    mask = p._compute_valid_token_mask(scores.device)
    assert mask[output_vocab.eos_id]
    assert int(mask.sum()) == 1


def test_input_skeleton_processor_v2_respects_custom_tuning_batch():
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=90,
        max_time_shift=120,
        num_frets=25,
        output_format="v2",
    )
    input_ids = torch.tensor(
        [[
            _token(input_vocab, "NOTE_ON_41"),
            _token(input_vocab, "NOTE_OFF_41"),
            input_vocab.pad_id,
        ]],
        dtype=torch.long,
    )

    # Standard tuning maps pitch 41 -> TAB_1_1
    processor_std = create_constrained_processor(
        input_ids=input_ids,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        mode="input_skeleton",
        num_frets=25,
    )
    p_std = processor_std.processors[0]
    scores = torch.zeros(output_vocab.vocab_size)
    mask_std = p_std._compute_valid_token_mask(scores.device)
    assert mask_std[_token(output_vocab, "TAB_1_1")]
    assert not mask_std[_token(output_vocab, "TAB_1_0")]

    # Shifted tuning (+1 semitone) maps pitch 41 -> TAB_1_0
    processor_shift = create_constrained_processor(
        input_ids=input_ids,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        mode="input_skeleton",
        tuning_batch=[[41, 46, 51, 56, 60, 65]],
        num_frets=25,
    )
    p_shift = processor_shift.processors[0]
    mask_shift = p_shift._compute_valid_token_mask(scores.device)
    assert mask_shift[_token(output_vocab, "TAB_1_0")]
