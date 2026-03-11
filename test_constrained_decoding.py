#!/usr/bin/env python3
"""
Test script for constrained decoding module.

Tests the TablatureLogitsProcessor and BatchTablatureLogitsProcessor classes
to ensure they correctly enforce tablature grammar constraints.
"""

import torch
from src.tab_dataset import build_vocabulary
from src.constrained_decoding import (
    TablatureLogitsProcessor,
    BatchTablatureLogitsProcessor,
    PitchToTabMapping,
    extract_pitches_from_input_ids,
    extract_pitches_batch,
    create_constrained_processor,
    STANDARD_TUNING,
)


def test_pitch_to_tab_mapping():
    """Test the PitchToTabMapping class."""
    print("Testing PitchToTabMapping...")
    
    mapping = PitchToTabMapping.build()
    
    # Test known pitch mappings
    # E2 (40) = open string 1
    tabs_40 = mapping.get_valid_tabs(40)
    assert (1, 0) in tabs_40, "E2 (40) should be playable on string 1, fret 0"
    
    # A2 (45) = open string 2, or string 1 fret 5
    tabs_45 = mapping.get_valid_tabs(45)
    assert (2, 0) in tabs_45, "A2 (45) should be playable on string 2, fret 0"
    assert (1, 5) in tabs_45, "A2 (45) should be playable on string 1, fret 5"
    
    # Test unplayable pitch (too low)
    tabs_20 = mapping.get_valid_tabs(20)
    assert len(tabs_20) == 0, "Pitch 20 should not be playable on standard guitar"
    
    print("  PitchToTabMapping: PASSED")


def test_processor_initialization():
    """Test TablatureLogitsProcessor initialization."""
    print("Testing TablatureLogitsProcessor initialization...")
    
    _, output_vocab = build_vocabulary()
    input_pitches = [40, 45, 50]  # E2, A2, D3
    
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches
    )
    
    assert processor.vocab == output_vocab
    assert processor.input_pitches == input_pitches
    assert len(processor.active_notes) == 0
    assert processor.last_token_type == 'START'
    assert processor.pitch_idx == 0
    
    # Check pre-computed mappings
    assert len(processor.note_on_ids) > 0, "Should have NOTE_ON token IDs"
    assert len(processor.note_off_ids) > 0, "Should have NOTE_OFF token IDs"
    assert len(processor.tab_ids) > 0, "Should have TAB token IDs"
    assert len(processor.time_shift_ids) > 0, "Should have TIME_SHIFT token IDs"
    
    print("  TablatureLogitsProcessor initialization: PASSED")


def test_state_machine_start():
    """Test state machine behavior from START state."""
    print("Testing state machine START state...")
    
    _, output_vocab = build_vocabulary()
    input_pitches = [60]  # C4
    
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        device='cpu'
    )
    
    # From START, should only allow NOTE_ON_60 or EOS
    scores = torch.zeros(output_vocab.vocab_size)
    masked_scores = processor(torch.tensor([1]), scores)  # [BOS]
    
    # Count valid tokens (not -inf)
    valid_mask = masked_scores > float('-inf')
    valid_count = valid_mask.sum().item()
    
    # Should allow NOTE_ON_60 and EOS (2 tokens)
    assert valid_count == 2, f"Expected 2 valid tokens from START, got {valid_count}"
    
    # Check that NOTE_ON_60 is valid
    note_on_60_id = output_vocab.token_to_id.get("NOTE_ON_60")
    if note_on_60_id:
        assert valid_mask[note_on_60_id], "NOTE_ON_60 should be valid from START"
    
    # Check that EOS is valid
    assert valid_mask[output_vocab.eos_id], "EOS should be valid from START"
    
    print("  State machine START state: PASSED")


def test_state_machine_after_note_on():
    """Test state machine behavior after NOTE_ON."""
    print("Testing state machine after NOTE_ON...")
    
    _, output_vocab = build_vocabulary()
    input_pitches = [45]  # A2
    
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        device='cpu'
    )
    
    # Simulate generating NOTE_ON_45
    processor.update_state(output_vocab.token_to_id["NOTE_ON_45"])
    
    assert processor.last_token_type == 'NOTE_ON'
    assert processor.last_pitch == 45
    assert 45 in processor.active_notes
    assert processor.pitch_idx == 1
    
    # After NOTE_ON, should only allow valid TABs for pitch 45
    scores = torch.zeros(output_vocab.vocab_size)
    masked_scores = processor(torch.tensor([1]), scores)
    
    valid_mask = masked_scores > float('-inf')
    valid_count = valid_mask.sum().item()
    
    # A2 (45) can be played on string 1 fret 5, string 2 fret 0
    # So should have at least 2 valid TAB tokens
    assert valid_count >= 2, f"Expected at least 2 valid TABs for A2, got {valid_count}"
    
    # Check specific valid TABs
    tab_1_5_id = output_vocab.token_to_id.get("TAB_1_5")
    tab_2_0_id = output_vocab.token_to_id.get("TAB_2_0")
    
    if tab_1_5_id:
        assert valid_mask[tab_1_5_id], "TAB_1_5 should be valid for A2 (45)"
    if tab_2_0_id:
        assert valid_mask[tab_2_0_id], "TAB_2_0 should be valid for A2 (45)"
    
    print("  State machine after NOTE_ON: PASSED")


def test_state_machine_after_tab():
    """Test state machine behavior after TAB."""
    print("Testing state machine after TAB...")
    
    _, output_vocab = build_vocabulary()
    input_pitches = [45, 50]  # A2, D3
    
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        device='cpu'
    )
    
    # Simulate generating NOTE_ON_45 -> TAB_2_0
    processor.update_state(output_vocab.token_to_id["NOTE_ON_45"])
    processor.update_state(output_vocab.token_to_id["TAB_2_0"])
    
    assert processor.last_token_type == 'TAB'
    
    # After TAB, can generate NOTE_ON (for next pitch) or NOTE_OFF
    scores = torch.zeros(output_vocab.vocab_size)
    masked_scores = processor(torch.tensor([1]), scores)
    
    valid_mask = masked_scores > float('-inf')
    
    # Should allow NOTE_ON_50 (next input pitch) and NOTE_OFF_45 (close current note)
    note_on_50_id = output_vocab.token_to_id.get("NOTE_ON_50")
    note_off_45_id = output_vocab.token_to_id.get("NOTE_OFF_45")
    
    if note_on_50_id:
        assert valid_mask[note_on_50_id], "NOTE_ON_50 should be valid after TAB"
    if note_off_45_id:
        assert valid_mask[note_off_45_id], "NOTE_OFF_45 should be valid after TAB"
    
    print("  State machine after TAB: PASSED")


def test_state_machine_after_note_off():
    """Test state machine behavior after NOTE_OFF."""
    print("Testing state machine after NOTE_OFF...")
    
    _, output_vocab = build_vocabulary()
    input_pitches = [45]  # A2
    
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        device='cpu'
    )
    
    # Simulate full note cycle: NOTE_ON_45 -> TAB_2_0 -> NOTE_OFF_45
    processor.update_state(output_vocab.token_to_id["NOTE_ON_45"])
    processor.update_state(output_vocab.token_to_id["TAB_2_0"])
    processor.update_state(output_vocab.token_to_id["NOTE_OFF_45"])
    
    assert processor.last_token_type == 'NOTE_OFF'
    assert 45 not in processor.active_notes, "Note 45 should be closed"
    assert len(processor.active_notes) == 0, "No active notes remaining"
    
    # After NOTE_OFF with no active notes, can generate TIME_SHIFT, NOTE_ON, or EOS
    scores = torch.zeros(output_vocab.vocab_size)
    masked_scores = processor(torch.tensor([1]), scores)
    
    valid_mask = masked_scores > float('-inf')
    
    # Should allow EOS
    assert valid_mask[output_vocab.eos_id], "EOS should be valid after all notes closed"
    
    # Should allow TIME_SHIFT
    time_shift_valid = any(valid_mask[tid] for tid in processor.time_shift_ids)
    assert time_shift_valid, "TIME_SHIFT should be valid after all notes closed"
    
    print("  State machine after NOTE_OFF: PASSED")


def test_batch_processor():
    """Test BatchTablatureLogitsProcessor."""
    print("Testing BatchTablatureLogitsProcessor...")
    
    _, output_vocab = build_vocabulary()
    input_pitches_batch = [
        [40, 45],  # E2, A2
        [50, 55],  # D3, G3
    ]
    
    processor = BatchTablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches_batch=input_pitches_batch,
        device='cpu'
    )
    
    assert len(processor.processors) == 2
    
    # Test batch processing
    batch_size = 2
    input_ids = torch.tensor([[1, 0], [1, 0]])  # [BOS, PAD]
    scores = torch.zeros(batch_size, output_vocab.vocab_size)
    
    masked_scores = processor(input_ids, scores)
    
    # Check that each sample has different valid tokens
    valid_mask_0 = masked_scores[0] > float('-inf')
    valid_mask_1 = masked_scores[1] > float('-inf')
    
    # Sample 0 should allow NOTE_ON_40, sample 1 should allow NOTE_ON_50
    note_on_40_id = output_vocab.token_to_id.get("NOTE_ON_40")
    note_on_50_id = output_vocab.token_to_id.get("NOTE_ON_50")
    
    if note_on_40_id:
        assert valid_mask_0[note_on_40_id], "Sample 0 should allow NOTE_ON_40"
    if note_on_50_id:
        assert valid_mask_1[note_on_50_id], "Sample 1 should allow NOTE_ON_50"
    
    print("  BatchTablatureLogitsProcessor: PASSED")


def test_extract_pitches():
    """Test pitch extraction from input IDs."""
    print("Testing extract_pitches_from_input_ids...")
    
    input_vocab, _ = build_vocabulary()
    
    # Create input sequence with NOTE_ON_40, NOTE_OFF_40, TIME_SHIFT_100, NOTE_ON_45
    input_ids = torch.tensor([
        input_vocab.token_to_id["NOTE_ON_40"],
        input_vocab.token_to_id["NOTE_OFF_40"],
        input_vocab.token_to_id["TIME_SHIFT_100"],
        input_vocab.token_to_id["NOTE_ON_45"],
        input_vocab.token_to_id["NOTE_OFF_45"],
        input_vocab.pad_id,
    ])
    
    pitches = extract_pitches_from_input_ids(input_ids, input_vocab)
    
    assert pitches == [40, 45], f"Expected [40, 45], got {pitches}"
    
    print("  extract_pitches_from_input_ids: PASSED")


def test_extract_pitches_batch():
    """Test batch pitch extraction."""
    print("Testing extract_pitches_batch...")
    
    input_vocab, _ = build_vocabulary()
    
    input_ids_batch = torch.tensor([
        [
            input_vocab.token_to_id["NOTE_ON_40"],
            input_vocab.token_to_id["NOTE_OFF_40"],
            input_vocab.pad_id,
        ],
        [
            input_vocab.token_to_id["NOTE_ON_50"],
            input_vocab.token_to_id["NOTE_ON_55"],
            input_vocab.pad_id,
        ],
    ])
    
    pitches_batch = extract_pitches_batch(input_ids_batch, input_vocab)
    
    assert pitches_batch == [[40], [50, 55]], f"Expected [[40], [50, 55]], got {pitches_batch}"
    
    print("  extract_pitches_batch: PASSED")


def test_create_constrained_processor():
    """Test the convenience function for creating processor."""
    print("Testing create_constrained_processor...")
    
    input_vocab, output_vocab = build_vocabulary()
    
    input_ids = torch.tensor([
        [
            input_vocab.token_to_id["NOTE_ON_40"],
            input_vocab.token_to_id["NOTE_OFF_40"],
        ],
    ])
    
    processor = create_constrained_processor(
        input_ids=input_ids,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        device='cpu'
    )
    
    assert isinstance(processor, BatchTablatureLogitsProcessor)
    assert len(processor.processors) == 1
    assert processor.processors[0].input_pitches == [40]
    
    print("  create_constrained_processor: PASSED")


def test_pitch_tab_consistency():
    """Test that only TABs matching pitch are allowed."""
    print("Testing pitch-TAB consistency...")
    
    _, output_vocab = build_vocabulary()
    
    # Test with pitch 64 (high E4) - only playable on string 6 fret 0
    input_pitches = [64]
    
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        device='cpu'
    )
    
    # Generate NOTE_ON_64
    processor.update_state(output_vocab.token_to_id["NOTE_ON_64"])
    
    # Get valid TABs
    scores = torch.zeros(output_vocab.vocab_size)
    masked_scores = processor(torch.tensor([1]), scores)
    valid_mask = masked_scores > float('-inf')
    
    # Check that TAB_6_0 is valid
    tab_6_0_id = output_vocab.token_to_id.get("TAB_6_0")
    assert valid_mask[tab_6_0_id], "TAB_6_0 should be valid for E4 (64)"
    
    # Check that wrong TABs are invalid
    tab_1_0_id = output_vocab.token_to_id.get("TAB_1_0")  # Would give E2 (40)
    assert not valid_mask[tab_1_0_id], "TAB_1_0 should NOT be valid for E4 (64)"
    
    print("  Pitch-TAB consistency: PASSED")


def test_time_shift_constraint():
    """Test that TIME_SHIFT is only allowed when all notes are closed."""
    print("Testing TIME_SHIFT constraint...")
    
    _, output_vocab = build_vocabulary()
    input_pitches = [45]
    
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        device='cpu'
    )
    
    # Generate NOTE_ON_45 -> TAB_2_0 (note still open)
    processor.update_state(output_vocab.token_to_id["NOTE_ON_45"])
    processor.update_state(output_vocab.token_to_id["TAB_2_0"])
    
    # After TAB with active note, TIME_SHIFT should NOT be valid
    scores = torch.zeros(output_vocab.vocab_size)
    masked_scores = processor(torch.tensor([1]), scores)
    valid_mask = masked_scores > float('-inf')
    
    time_shift_valid_before = any(valid_mask[tid] for tid in processor.time_shift_ids)
    assert not time_shift_valid_before, "TIME_SHIFT should NOT be valid with active notes"
    
    # Now close the note
    processor.update_state(output_vocab.token_to_id["NOTE_OFF_45"])
    
    # After NOTE_OFF with no active notes, TIME_SHIFT SHOULD be valid
    scores = torch.zeros(output_vocab.vocab_size)
    masked_scores = processor(torch.tensor([1]), scores)
    valid_mask = masked_scores > float('-inf')
    
    time_shift_valid_after = any(valid_mask[tid] for tid in processor.time_shift_ids)
    assert time_shift_valid_after, "TIME_SHIFT should be valid after all notes closed"
    
    print("  TIME_SHIFT constraint: PASSED")


def main():
    print("=" * 60)
    print("Constrained Decoding Tests")
    print("=" * 60)
    print()
    
    test_pitch_to_tab_mapping()
    test_processor_initialization()
    test_state_machine_start()
    test_state_machine_after_note_on()
    test_state_machine_after_tab()
    test_state_machine_after_note_off()
    test_batch_processor()
    test_extract_pitches()
    test_extract_pitches_batch()
    test_create_constrained_processor()
    test_pitch_tab_consistency()
    test_time_shift_constraint()
    
    print()
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
