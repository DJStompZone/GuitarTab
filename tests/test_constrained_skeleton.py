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
