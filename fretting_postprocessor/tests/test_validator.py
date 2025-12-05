"""
Test Suite for PitchValidator
==============================

Tests for tablature validation and correction functionality.
"""

import pytest
from fretting_postprocessor.datatypes import Note
from fretting_postprocessor.config import GuitarConfig, STANDARD_TUNING, DROP_D_TUNING
from fretting_postprocessor.validator import (
    PitchValidator,
    validate_sequence,
    calculate_pitch_accuracy
)


class TestValidateNote:
    """Tests for PitchValidator.validate_note()"""

    def test_valid_tablature(self):
        """Test validation of correct tablature"""
        config = GuitarConfig()

        # String 1 (A2, pitch=45), fret 0 = A2 (pitch=45)
        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=1, fret=0)
        assert PitchValidator.validate_note(note, config) is True

    def test_valid_tablature_with_fret(self):
        """Test validation with non-zero fret"""
        config = GuitarConfig()

        # String 0 (E2, pitch=40), fret 5 = A2 (pitch=45)
        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=0, fret=5)
        assert PitchValidator.validate_note(note, config) is True

    def test_invalid_pitch_mismatch(self):
        """Test detection of pitch mismatch"""
        config = GuitarConfig()

        # String 1 (A2, pitch=45), fret 5 = D3 (pitch=50)
        # But note claims pitch=45 - mismatch!
        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=1, fret=5)
        assert PitchValidator.validate_note(note, config) is False

    def test_invalid_no_tablature(self):
        """Test detection of missing tablature"""
        config = GuitarConfig()

        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)
        assert PitchValidator.validate_note(note, config) is False

    def test_invalid_string_out_of_range(self):
        """Test detection of invalid string number"""
        config = GuitarConfig()

        # String 7 doesn't exist on 6-string guitar
        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=7, fret=0)
        assert PitchValidator.validate_note(note, config) is False

    def test_invalid_fret_out_of_range(self):
        """Test detection of invalid fret number"""
        config = GuitarConfig()

        # Fret 30 exceeds max_fret=24
        note = Note(pitch=70, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=5, fret=30)
        assert PitchValidator.validate_note(note, config) is False

    def test_invalid_negative_fret(self):
        """Test detection of negative fret"""
        config = GuitarConfig()

        note = Note(pitch=40, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=0, fret=-1)
        assert PitchValidator.validate_note(note, config) is False

    def test_validation_with_capo(self):
        """Test validation with capo"""
        config = GuitarConfig(capo_fret=2)

        # With capo on fret 2, open string 0 (E2) plays at fret 2 = F#2 (pitch=42)
        # So fret 0 relative to capo = pitch 42
        note = Note(pitch=42, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=0, fret=0)
        assert PitchValidator.validate_note(note, config) is True


class TestCorrectNoteTablature:
    """Tests for PitchValidator.correct_note_tablature()"""

    def test_correct_simple_pitch(self):
        """Test correction of note without tablature"""
        config = GuitarConfig()

        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)

        success = PitchValidator.correct_note_tablature(note, config)

        assert success is True
        assert note.has_tablature()
        assert note.string is not None
        assert note.fret is not None

        # Verify pitch calculation
        effective_tuning = config.get_effective_tuning()
        calculated_pitch = effective_tuning[note.string] + note.fret
        assert calculated_pitch == 45

    def test_correct_with_preferred_string(self):
        """Test correction with preferred string"""
        config = GuitarConfig()

        # Pitch 45 (A2) can be played on string 0 fret 5 OR string 1 fret 0
        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)

        success = PitchValidator.correct_note_tablature(note, config, preferred_string=1)

        assert success is True
        assert note.string == 1  # Should use preferred string
        assert note.fret == 0

    def test_correct_with_invalid_preferred_string(self):
        """Test correction when preferred string cannot play the pitch"""
        config = GuitarConfig()

        # Pitch 40 (E2) can only be played on string 0 (or higher strings with high frets)
        # String 5 (E4, pitch=64) cannot play pitch 40 (would need negative fret)
        note = Note(pitch=40, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)

        success = PitchValidator.correct_note_tablature(note, config, preferred_string=5)

        assert success is True
        assert note.string == 0  # Falls back to first valid position
        assert note.fret == 0

    def test_correct_out_of_range_pitch(self):
        """Test correction fails for out-of-range pitch"""
        config = GuitarConfig()

        # Pitch 28 (E1) is too low for standard tuning
        note = Note(pitch=28, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)

        success = PitchValidator.correct_note_tablature(note, config)

        assert success is False
        # Tablature should remain None
        assert note.string is None
        assert note.fret is None

    def test_correct_fixes_wrong_tablature(self):
        """Test correction can fix existing wrong tablature"""
        config = GuitarConfig()

        # Wrong tablature: string 2 fret 10 = pitch 60, but note says pitch 45
        note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=2, fret=10)

        assert PitchValidator.validate_note(note, config) is False

        success = PitchValidator.correct_note_tablature(note, config)

        assert success is True
        assert PitchValidator.validate_note(note, config) is True

    def test_correct_with_capo(self):
        """Test correction with capo"""
        config = GuitarConfig(capo_fret=2)

        # Pitch 42 (F#2) with capo on fret 2
        note = Note(pitch=42, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)

        success = PitchValidator.correct_note_tablature(note, config)

        assert success is True
        assert note.string == 0  # String 0 with capo
        assert note.fret == 0     # Open string (relative to capo)

    def test_correct_multiple_positions_returns_first(self):
        """Test that correction returns first valid position by default"""
        config = GuitarConfig()

        # Pitch 50 (D3) has multiple valid positions
        note = Note(pitch=50, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)

        success = PitchValidator.correct_note_tablature(note, config)

        assert success is True
        # Should get first position (lowest string)
        assert note.string == 0
        assert note.fret == 10


class TestGetAlternativePositions:
    """Tests for PitchValidator.get_alternative_positions()"""

    def test_get_alternatives_single_pitch(self):
        """Test getting alternatives for pitch with multiple positions"""
        config = GuitarConfig()

        # Pitch 50 (D3) can be played at:
        # String 0 fret 10, String 1 fret 5, String 2 fret 0
        note = Note(pitch=50, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=2, fret=0)

        alternatives = PitchValidator.get_alternative_positions(note, config)

        # Should exclude current position (2, 0)
        assert len(alternatives) == 2
        assert (0, 10) in alternatives
        assert (1, 5) in alternatives
        assert (2, 0) not in alternatives  # Excluded

    def test_get_alternatives_include_current(self):
        """Test getting alternatives including current position"""
        config = GuitarConfig()

        note = Note(pitch=50, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=2, fret=0)

        alternatives = PitchValidator.get_alternative_positions(
            note, config, exclude_current=False
        )

        # Should include current position
        assert len(alternatives) == 3
        assert (0, 10) in alternatives
        assert (1, 5) in alternatives
        assert (2, 0) in alternatives  # Included

    def test_get_alternatives_no_tablature(self):
        """Test getting alternatives when note has no tablature"""
        config = GuitarConfig()

        note = Note(pitch=50, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=None, fret=None)

        alternatives = PitchValidator.get_alternative_positions(note, config)

        # Should return all positions (no current to exclude)
        assert len(alternatives) == 3

    def test_get_alternatives_unique_position(self):
        """Test pitch with only one valid position"""
        config = GuitarConfig()

        # Pitch 40 (E2) - only string 0 fret 0
        note = Note(pitch=40, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=0, fret=0)

        alternatives = PitchValidator.get_alternative_positions(note, config)

        # Should return empty list (current position excluded, no others exist)
        assert len(alternatives) == 0

    def test_get_alternatives_high_pitch(self):
        """Test alternatives for high pitch"""
        config = GuitarConfig()

        # Pitch 64 (E4) - can be played on string 5 fret 0, or string 4 fret 5, etc.
        note = Note(pitch=64, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=5, fret=0)

        alternatives = PitchValidator.get_alternative_positions(note, config)

        # Should have alternatives (at least string 4 fret 5)
        assert len(alternatives) >= 1
        assert (4, 5) in alternatives

    def test_get_alternatives_with_capo(self):
        """Test alternatives with capo"""
        config = GuitarConfig(capo_fret=2)

        # Pitch 50 (D3) with capo on fret 2
        note = Note(pitch=50, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=1, fret=3)  # A2+2 (capo) + 3 frets = D3

        alternatives = PitchValidator.get_alternative_positions(note, config)

        # Should find alternatives considering capo
        assert len(alternatives) >= 1


class TestValidateSequence:
    """Tests for validate_sequence() utility function"""

    def test_validate_all_correct(self):
        """Test validation of sequence with all correct notes"""
        config = GuitarConfig()

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=0),
            Note(pitch=55, onset_ticks=960, duration_ticks=480,
                velocity=80, string=3, fret=0),
        ]

        valid_count, total_count = validate_sequence(notes, config)

        assert valid_count == 3
        assert total_count == 3

    def test_validate_mixed_sequence(self):
        """Test validation of sequence with some invalid notes"""
        config = GuitarConfig()

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),      # Valid
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=5),      # Invalid (should be fret 0)
            Note(pitch=55, onset_ticks=960, duration_ticks=480,
                velocity=80, string=3, fret=0),      # Valid
        ]

        valid_count, total_count = validate_sequence(notes, config)

        assert valid_count == 2
        assert total_count == 3

    def test_validate_empty_sequence(self):
        """Test validation of empty sequence"""
        config = GuitarConfig()

        notes = []

        valid_count, total_count = validate_sequence(notes, config)

        assert valid_count == 0
        assert total_count == 0


class TestCalculatePitchAccuracy:
    """Tests for calculate_pitch_accuracy() utility function"""

    def test_perfect_accuracy(self):
        """Test 100% pitch accuracy"""
        config = GuitarConfig()

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=0),
        ]

        accuracy = calculate_pitch_accuracy(notes, config)

        assert accuracy == 100.0

    def test_partial_accuracy(self):
        """Test 50% pitch accuracy"""
        config = GuitarConfig()

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),      # Valid
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=5),      # Invalid
        ]

        accuracy = calculate_pitch_accuracy(notes, config)

        assert accuracy == 50.0

    def test_zero_accuracy(self):
        """Test 0% pitch accuracy"""
        config = GuitarConfig()

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=2, fret=10),     # Invalid
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=3, fret=10),     # Invalid
        ]

        accuracy = calculate_pitch_accuracy(notes, config)

        assert accuracy == 0.0

    def test_empty_sequence_accuracy(self):
        """Test accuracy of empty sequence"""
        config = GuitarConfig()

        notes = []

        accuracy = calculate_pitch_accuracy(notes, config)

        assert accuracy == 0.0


class TestEdgeCases:
    """Edge case tests"""

    def test_drop_d_tuning(self):
        """Test validation with Drop-D tuning"""
        config = GuitarConfig(tuning=DROP_D_TUNING)

        # String 0 is now D2 (pitch=38)
        note = Note(pitch=38, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=0, fret=0)

        assert PitchValidator.validate_note(note, config) is True

    def test_extreme_high_fret(self):
        """Test validation at maximum fret"""
        config = GuitarConfig()

        # String 5 (E4, pitch=64) + fret 24 = E6 (pitch=88)
        note = Note(pitch=88, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=5, fret=24)

        assert PitchValidator.validate_note(note, config) is True

    def test_boundary_fret_range(self):
        """Test fret range boundaries"""
        config = GuitarConfig(min_fret=0, max_fret=12)

        # Fret 12 should be valid
        note = Note(pitch=52, onset_ticks=0, duration_ticks=480,
                   velocity=80, string=0, fret=12)
        assert PitchValidator.validate_note(note, config) is True

        # Fret 13 should be invalid
        note.fret = 13
        note.pitch = 53
        assert PitchValidator.validate_note(note, config) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
