"""
Tests for Parser and Serializer (Phase 2)
==========================================

Tests for token parsing and serialization.
"""

import sys
sys.path.insert(0, '/work/b10502010/GuitarTab')

from fretting_postprocessor.parser import TokenParser
from fretting_postprocessor.serializer import TokenSerializer
from fretting_postprocessor.config import GuitarConfig
from fretting_postprocessor.datatypes import Note
from fretting_postprocessor.sequence import NoteSequence


def test_parse_input_tokens():
    """Test parsing input tokens (NOTE_ON/OFF format)"""
    print("Testing parse_input_tokens...")

    parser = TokenParser()

    tokens = [
        "NOTE_ON<60>",
        "TIME_SHIFT<240>",
        "NOTE_OFF<60>",
        "NOTE_ON<62>",
        "TIME_SHIFT<240>",
        "NOTE_OFF<62>"
    ]

    sequence = parser.parse_input_tokens(tokens)

    assert len(sequence) == 2

    # First note
    assert sequence[0].pitch == 60
    assert sequence[0].onset_ticks == 0
    assert sequence[0].duration_ticks == 240

    # Second note
    assert sequence[1].pitch == 62
    assert sequence[1].onset_ticks == 240
    assert sequence[1].duration_ticks == 240

    print("  ✓ parse_input_tokens works")


def test_parse_input_chord():
    """Test parsing chord (multiple notes at same time)"""
    print("\nTesting parse_input_chord...")

    parser = TokenParser()

    tokens = [
        "NOTE_ON<60>",
        "NOTE_ON<64>",
        "NOTE_ON<67>",
        "TIME_SHIFT<480>",
        "NOTE_OFF<60>",
        "NOTE_OFF<64>",
        "NOTE_OFF<67>"
    ]

    sequence = parser.parse_input_tokens(tokens)

    assert len(sequence) == 3

    # All notes start at same time
    assert sequence[0].onset_ticks == 0
    assert sequence[1].onset_ticks == 0
    assert sequence[2].onset_ticks == 0

    # All notes have same duration
    assert all(note.duration_ticks == 480 for note in sequence)

    # Check pitches (should be sorted)
    pitches = [note.pitch for note in sequence]
    assert pitches == [60, 64, 67]

    print("  ✓ parse_input_chord works")


def test_parse_output_tokens():
    """Test parsing output tokens (TAB format)"""
    print("\nTesting parse_output_tokens...")

    parser = TokenParser()
    config = GuitarConfig()

    # First create input sequence for reference
    input_tokens = [
        "NOTE_ON<55>",  # G3
        "TIME_SHIFT<480>",
        "NOTE_OFF<55>"
    ]
    input_sequence = parser.parse_input_tokens(input_tokens)

    # Parse output (TAB format)
    output_tokens = [
        "TAB<3,0>",  # G string, open (pitch 55)
        "TIME_SHIFT<480>"
    ]

    output_sequence = parser.parse_output_tokens(
        output_tokens,
        input_sequence,
        config
    )

    assert len(output_sequence) == 1
    note = output_sequence[0]

    # Check tablature
    assert note.string == 3
    assert note.fret == 0

    # Check calculated pitch
    assert note.pitch == 55  # G3

    # Check duration (matched from input)
    assert note.duration_ticks == 480

    print("  ✓ parse_output_tokens works")


def test_serialize_to_input_format():
    """Test serializing to input format"""
    print("\nTesting serialize_to_input_format...")

    serializer = TokenSerializer()

    notes = [
        Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80),
        Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80),
    ]
    sequence = NoteSequence(notes)

    tokens = serializer.serialize_to_input_format(sequence)

    expected = [
        "NOTE_ON<60>",
        "TIME_SHIFT<480>",
        "NOTE_OFF<60>",
        "NOTE_ON<62>",
        "TIME_SHIFT<480>",
        "NOTE_OFF<62>"
    ]

    assert tokens == expected

    print("  ✓ serialize_to_input_format works")


def test_serialize_to_output_format():
    """Test serializing to output format"""
    print("\nTesting serialize_to_output_format...")

    serializer = TokenSerializer()

    notes = [
        Note(pitch=60, onset_ticks=0, duration_ticks=480,
             velocity=80, string=3, fret=5),
        Note(pitch=62, onset_ticks=480, duration_ticks=480,
             velocity=80, string=3, fret=7),
    ]
    sequence = NoteSequence(notes)

    tokens = serializer.serialize_to_output_format(sequence)

    expected = [
        "TAB<3,5>",
        "TIME_SHIFT<480>",
        "TAB<3,7>",
        "TIME_SHIFT<480>"
    ]

    print(f"  Generated tokens: {tokens}")
    print(f"  Expected tokens: {expected}")
    assert tokens == expected

    print("  ✓ serialize_to_output_format works")


def test_serialize_chord():
    """Test serializing chord (multiple notes at same time)"""
    print("\nTesting serialize_chord...")

    serializer = TokenSerializer()

    # C major chord at time 0
    notes = [
        Note(pitch=60, onset_ticks=0, duration_ticks=480,
             velocity=80, string=1, fret=3),  # C on A string
        Note(pitch=64, onset_ticks=0, duration_ticks=480,
             velocity=80, string=2, fret=2),  # E on D string
        Note(pitch=67, onset_ticks=0, duration_ticks=480,
             velocity=80, string=3, fret=0),  # G on G string (open)
    ]
    sequence = NoteSequence(notes)

    tokens = serializer.serialize_to_output_format(sequence)

    # All TAB tokens should appear before TIME_SHIFT
    # Sorted by string number
    expected_start = ["TAB<1,3>", "TAB<2,2>", "TAB<3,0>", "TIME_SHIFT<480>"]

    assert tokens[:4] == expected_start

    print("  ✓ serialize_chord works")


def test_round_trip():
    """Test parse -> serialize round trip"""
    print("\nTesting round trip (parse -> serialize)...")

    parser = TokenParser()
    serializer = TokenSerializer()

    # Original tokens
    original_tokens = [
        "NOTE_ON<60>",
        "TIME_SHIFT<480>",
        "NOTE_OFF<60>",
        "NOTE_ON<62>",
        "NOTE_ON<64>",  # Chord
        "TIME_SHIFT<480>",
        "NOTE_OFF<62>",
        "NOTE_OFF<64>"
    ]

    # Parse
    sequence = parser.parse_input_tokens(original_tokens)

    # Serialize
    reconstructed_tokens = serializer.serialize_to_input_format(sequence)

    # Should match
    assert reconstructed_tokens == original_tokens

    print("  ✓ round trip works")


def test_tokens_to_string():
    """Test token list to string conversion"""
    print("\nTesting tokens_to_string...")

    serializer = TokenSerializer()

    tokens = ["NOTE_ON<60>", "TIME_SHIFT<480>", "NOTE_OFF<60>"]

    result = serializer.tokens_to_string(tokens)

    expected = "NOTE_ON<60> TIME_SHIFT<480> NOTE_OFF<60>"

    assert result == expected

    # Test with custom separator
    result = serializer.tokens_to_string(tokens, separator="\n")
    expected = "NOTE_ON<60>\nTIME_SHIFT<480>\nNOTE_OFF<60>"

    assert result == expected

    print("  ✓ tokens_to_string works")


def test_string_to_tokens():
    """Test string to token list conversion"""
    print("\nTesting string_to_tokens...")

    serializer = TokenSerializer()

    token_string = "NOTE_ON<60> TIME_SHIFT<480> NOTE_OFF<60>"

    result = serializer.string_to_tokens(token_string)

    expected = ["NOTE_ON<60>", "TIME_SHIFT<480>", "NOTE_OFF<60>"]

    assert result == expected

    print("  ✓ string_to_tokens works")


def run_all_tests():
    """Run all Phase 2 tests"""
    print("=" * 60)
    print("Running Phase 2 Parser/Serializer Tests")
    print("=" * 60)

    test_parse_input_tokens()
    test_parse_input_chord()
    test_parse_output_tokens()
    test_serialize_to_input_format()
    test_serialize_to_output_format()
    test_serialize_chord()
    test_round_trip()
    test_tokens_to_string()
    test_string_to_tokens()

    print("\n" + "=" * 60)
    print("✓ All Phase 2 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
