"""Sanity: output schedule from encoder input matches v1 dadagp-aligned structure."""

import torch

from src.constrained_decoding import (
    SchedNoteOff,
    SchedNoteOn,
    SchedTab,
    SchedTimeShift,
    build_output_schedule_from_input,
)
from src.tab_dataset import build_vocabulary


def test_build_schedule_two_notes_off_shift():
    input_vocab, output_vocab = build_vocabulary(
        output_format="v1", num_frets=25, max_time_shift=500
    )
    inp = [
        input_vocab.token_to_id["NOTE_ON_40"],
        input_vocab.token_to_id["NOTE_ON_45"],
        input_vocab.token_to_id["NOTE_OFF_40"],
        input_vocab.token_to_id["NOTE_OFF_45"],
        input_vocab.token_to_id["TIME_SHIFT_120"],
    ]
    sched = build_output_schedule_from_input(
        torch.tensor(inp, dtype=torch.long), input_vocab, output_vocab
    )
    assert len(sched) == 7
    assert isinstance(sched[0], SchedNoteOn) and sched[0].pitch == 40
    assert isinstance(sched[1], SchedTab) and sched[1].pitch == 40
    assert isinstance(sched[2], SchedNoteOn) and sched[2].pitch == 45
    assert isinstance(sched[3], SchedTab) and sched[3].pitch == 45
    assert isinstance(sched[4], SchedNoteOff) and sched[4].pitch == 40
    assert isinstance(sched[5], SchedNoteOff) and sched[5].pitch == 45
    assert isinstance(sched[6], SchedTimeShift)
    assert sched[6].token_id == output_vocab.token_to_id["TIME_SHIFT_120"]


def test_schedule_matches_gt_body_length():
    """GT output body (no BOS/EOS) should have same length as schedule for aligned v1 pair."""
    input_vocab, output_vocab = build_vocabulary(
        output_format="v1", num_frets=25, max_time_shift=500
    )
    inp = [
        input_vocab.token_to_id["NOTE_ON_64"],
        input_vocab.token_to_id["NOTE_OFF_64"],
        input_vocab.token_to_id["TIME_SHIFT_240"],
    ]
    sched = build_output_schedule_from_input(
        torch.tensor(inp, dtype=torch.long), input_vocab, output_vocab
    )
    tab_id = output_vocab.token_to_id["TAB_1_0"]
    body = [
        output_vocab.token_to_id["NOTE_ON_64"],
        tab_id,
        output_vocab.token_to_id["NOTE_OFF_64"],
        output_vocab.token_to_id["TIME_SHIFT_240"],
    ]
    assert len(sched) == len(body)
