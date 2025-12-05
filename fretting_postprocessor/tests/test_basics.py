"""
Basic Tests for Phase 1 Components
===================================

Tests for datatypes, config, and sequence modules.
"""

import sys
sys.path.insert(0, '/work/b10502010/GuitarTab')

from fretting_postprocessor.datatypes import TokenType, Token, Note
from fretting_postprocessor.config import (
    GuitarConfig,
    STANDARD_TUNING,
    DROP_D_TUNING,
)
from fretting_postprocessor.sequence import NoteSequence


def test_token_types():
    """Test TokenType enum"""
    print("Testing TokenType enum...")
    assert TokenType.NOTE_ON.value == "NOTE_ON"
    assert TokenType.TAB.value == "TAB"
    print("  ✓ TokenType enum works")


def test_token_representation():
    """Test Token dataclass"""
    print("\nTesting Token dataclass...")

    # NOTE_ON token
    token = Token(TokenType.NOTE_ON, 60)
    assert str(token) == "NOTE_ON<60>"

    # TAB token
    tab_token = Token(TokenType.TAB, 0, string_fret=(2, 5))
    assert str(tab_token) == "TAB<2,5>"

    print("  ✓ Token representation works")


def test_note_basic():
    """Test Note dataclass basic functionality"""
    print("\nTesting Note dataclass...")

    note = Note(
        pitch=60,
        onset_ticks=0,
        duration_ticks=480,
        velocity=80
    )

    assert note.pitch == 60
    assert note.onset_ticks == 0
    assert note.duration_ticks == 480
    assert note.get_offset_ticks() == 480
    assert not note.has_tablature()

    print("  ✓ Note basic functionality works")


def test_note_with_tablature():
    """Test Note with tablature information"""
    print("\nTesting Note with tablature...")

    note = Note(
        pitch=50,
        onset_ticks=0,
        duration_ticks=480,
        velocity=80,
        string=1,
        fret=5
    )

    assert note.has_tablature()
    assert note.string == 1
    assert note.fret == 5

    # Test pitch calculation from tablature
    tuning = STANDARD_TUNING  # (40, 45, 50, 55, 59, 64)
    calculated_pitch = note.get_pitch_from_tablature(tuning)
    assert calculated_pitch == 50  # A2 (45) + 5 frets = D3 (50)

    print("  ✓ Note with tablature works")


def test_guitar_config_basic():
    """Test GuitarConfig basic functionality"""
    print("\nTesting GuitarConfig...")

    config = GuitarConfig()

    assert config.num_strings == 6
    assert config.tuning == STANDARD_TUNING
    assert config.capo_fret == 0
    assert config.is_valid_string(0)
    assert config.is_valid_string(5)
    assert not config.is_valid_string(6)
    assert config.is_valid_fret(0)
    assert config.is_valid_fret(24)
    assert not config.is_valid_fret(25)

    print("  ✓ GuitarConfig basic functionality works")


def test_guitar_config_capo():
    """Test GuitarConfig with capo"""
    print("\nTesting GuitarConfig with capo...")

    config = GuitarConfig(capo_fret=2)

    effective = config.get_effective_tuning()
    # Each string should be raised by 2 semitones
    assert effective == (42, 47, 52, 57, 61, 66)

    print("  ✓ GuitarConfig capo functionality works")


def test_pitch_to_string_fret():
    """Test pitch_to_string_fret conversion"""
    print("\nTesting pitch_to_string_fret...")

    config = GuitarConfig()

    # Test A2 (pitch 45) - can be played on string 0 fret 5, OR string 1 open
    positions = config.pitch_to_string_fret(45)
    assert (0, 5) in positions
    assert (1, 0) in positions

    # Test middle C (pitch 60)
    positions = config.pitch_to_string_fret(60)
    assert (2, 10) in positions  # D string (50), fret 10
    assert (3, 5) in positions   # G string (55), fret 5
    assert (4, 1) in positions   # B string (59), fret 1

    # Test pitch too low (below E2)
    positions = config.pitch_to_string_fret(28)
    assert len(positions) == 0

    print("  ✓ pitch_to_string_fret works correctly")


def test_note_sequence_basic():
    """Test NoteSequence basic functionality"""
    print("\nTesting NoteSequence...")

    notes = [
        Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80),
        Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80),
        Note(pitch=64, onset_ticks=960, duration_ticks=480, velocity=80),
    ]

    sequence = NoteSequence(notes, source="test")

    assert len(sequence) == 3
    assert sequence.source == "test"

    # Test indexing
    assert sequence[0].pitch == 60
    assert sequence[1].pitch == 62

    # Test iteration
    pitches = [note.pitch for note in sequence]
    assert pitches == [60, 62, 64]

    print("  ✓ NoteSequence basic functionality works")


def test_note_sequence_time_queries():
    """Test NoteSequence time-based queries"""
    print("\nTesting NoteSequence time queries...")

    notes = [
        Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80),
        Note(pitch=62, onset_ticks=0, duration_ticks=480, velocity=80),  # Chord at time 0
        Note(pitch=64, onset_ticks=480, duration_ticks=480, velocity=80),
        Note(pitch=65, onset_ticks=960, duration_ticks=480, velocity=80),
    ]

    sequence = NoteSequence(notes)

    # Test get_notes_at_time
    notes_at_0 = sequence.get_notes_at_time(0)
    assert len(notes_at_0) == 2
    assert notes_at_0[0].pitch == 60
    assert notes_at_0[1].pitch == 62

    # Test get_notes_in_time_range
    notes_in_range = sequence.get_notes_in_time_range(0, 500)
    assert len(notes_in_range) == 3  # Notes at 0 and 480

    print("  ✓ NoteSequence time queries work")


def test_note_sequence_window():
    """Test NoteSequence window query (for overlap correction)"""
    print("\nTesting NoteSequence window query...")

    notes = [
        Note(pitch=60 + i, onset_ticks=i * 100, duration_ticks=100, velocity=80)
        for i in range(10)
    ]

    sequence = NoteSequence(notes)

    # Get window around index 5 (±2 notes)
    window = sequence.get_notes_in_window(500, window_size=2)

    # Should get notes at indices 3, 4, 5, 6, 7 (5 notes)
    assert len(window) == 5
    assert window[0].pitch == 63
    assert window[2].pitch == 65  # Center note
    assert window[4].pitch == 67

    print("  ✓ NoteSequence window query works")


def test_find_closest_note():
    """Test find_closest_note functionality"""
    print("\nTesting find_closest_note...")

    notes = [
        Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80),
        Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80),
        Note(pitch=64, onset_ticks=960, duration_ticks=480, velocity=80),
    ]

    sequence = NoteSequence(notes)

    # Find note closest to pitch 62 at time 500
    closest = sequence.find_closest_note(62, 500, max_time_diff=240)
    assert closest is not None
    assert closest.pitch == 62
    assert closest.onset_ticks == 480

    # Find note with no match (time window too narrow)
    closest = sequence.find_closest_note(64, 100, max_time_diff=50)
    assert closest is None  # No notes within [50, 150]

    print("  ✓ find_closest_note works")


def run_all_tests():
    """Run all Phase 1 tests"""
    print("=" * 60)
    print("Running Phase 1 Basic Tests")
    print("=" * 60)

    test_token_types()
    test_token_representation()
    test_note_basic()
    test_note_with_tablature()
    test_guitar_config_basic()
    test_guitar_config_capo()
    test_pitch_to_string_fret()
    test_note_sequence_basic()
    test_note_sequence_time_queries()
    test_note_sequence_window()
    test_find_closest_note()

    print("\n" + "=" * 60)
    print("✓ All Phase 1 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
