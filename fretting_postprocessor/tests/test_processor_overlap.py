"""
Test Suite for PostProcessor - Overlap Correction
==================================================

Tests for the overlap correction algorithm (Section 3.5 of the paper).
"""

import pytest
from fretting_postprocessor.datatypes import Note
from fretting_postprocessor.config import GuitarConfig, STANDARD_TUNING
from fretting_postprocessor.sequence import NoteSequence
from fretting_postprocessor.processor import PostProcessor
from fretting_postprocessor.validator import calculate_pitch_accuracy


class TestFindBestMatch:
    """Tests for PostProcessor._find_best_match()"""

    def test_perfect_match(self):
        """Test matching with identical pitch and time"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80, string=1, fret=0)

        candidates = [
            Note(pitch=45, onset_ticks=100, duration_ticks=480,
                velocity=80),  # Perfect match
            Note(pitch=50, onset_ticks=200, duration_ticks=240,
                velocity=70),  # Different
        ]

        best = processor._find_best_match(model_note, candidates)

        assert best is not None
        assert best.pitch == 45
        assert best.onset_ticks == 100

    def test_close_pitch_preferred(self):
        """Test that closer pitch is preferred"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80, string=1, fret=0)

        candidates = [
            Note(pitch=47, onset_ticks=100, duration_ticks=480,
                velocity=80),  # 2 semitones off
            Note(pitch=46, onset_ticks=100, duration_ticks=480,
                velocity=80),  # 1 semitone off (better)
        ]

        best = processor._find_best_match(model_note, candidates)

        assert best is not None
        assert best.pitch == 46  # Closer pitch selected

    def test_close_time_preferred(self):
        """Test that closer time is preferred when pitch is same"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80, string=1, fret=0)

        candidates = [
            Note(pitch=45, onset_ticks=200, duration_ticks=480,
                velocity=80),  # 100 ticks off
            Note(pitch=45, onset_ticks=110, duration_ticks=480,
                velocity=80),  # 10 ticks off (better)
        ]

        best = processor._find_best_match(model_note, candidates)

        assert best is not None
        assert best.onset_ticks == 110  # Closer time selected

    def test_skip_matched_notes(self):
        """Test that already matched notes are skipped"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80, string=1, fret=0)

        candidates = [
            Note(pitch=45, onset_ticks=100, duration_ticks=480,
                velocity=80, matched=True),  # Already matched
            Note(pitch=45, onset_ticks=150, duration_ticks=480,
                velocity=80, matched=False),  # Available
        ]

        best = processor._find_best_match(model_note, candidates)

        assert best is not None
        assert best.onset_ticks == 150  # Second note selected
        assert not best.matched

    def test_no_candidates(self):
        """Test handling of empty candidate list"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80, string=1, fret=0)

        best = processor._find_best_match(model_note, [])

        assert best is None

    def test_all_candidates_matched(self):
        """Test when all candidates are already matched"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80, string=1, fret=0)

        candidates = [
            Note(pitch=45, onset_ticks=100, duration_ticks=480,
                velocity=80, matched=True),
            Note(pitch=45, onset_ticks=110, duration_ticks=480,
                velocity=80, matched=True),
        ]

        best = processor._find_best_match(model_note, candidates)

        assert best is None


class TestCreateFallbackNote:
    """Tests for PostProcessor._create_fallback_note()"""

    def test_fallback_creates_valid_tablature(self):
        """Test that fallback note has valid tablature"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80, string=None, fret=None)

        fallback = processor._create_fallback_note(model_note)

        assert fallback.has_tablature()
        assert fallback.string is not None
        assert fallback.fret is not None
        assert fallback.source == "fallback"

    def test_fallback_preserves_timing(self):
        """Test that fallback preserves model's timing"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480,
                         velocity=80)

        fallback = processor._create_fallback_note(model_note)

        assert fallback.onset_ticks == 100
        assert fallback.duration_ticks == 480
        assert fallback.pitch == 45

    def test_fallback_with_out_of_range_pitch(self):
        """Test fallback handling of out-of-range pitch"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Pitch too low for standard guitar
        model_note = Note(pitch=30, onset_ticks=0, duration_ticks=480,
                         velocity=80)

        fallback = processor._create_fallback_note(model_note)

        # Should adjust to valid range
        min_pitch, max_pitch = config.get_pitch_range()
        assert min_pitch <= fallback.pitch <= max_pitch
        assert fallback.has_tablature()


class TestOverlapCorrection:
    """Tests for PostProcessor.overlap_correction()"""

    def test_simple_single_note_correction(self):
        """Test correction of single note with pitch error"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model output has wrong pitch (47 instead of 45)
        model_output = NoteSequence([
            Note(pitch=47, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=2)  # Wrong pitch
        ])

        # Input has correct pitch
        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80)  # Ground truth: A2
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        assert len(corrected) == 1
        assert corrected.notes[0].pitch == 45  # Corrected to ground truth
        assert corrected.notes[0].source == "corrected"

    def test_preserves_model_timing(self):
        """Test that model's timing is preserved"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model has slightly different timing than input
        model_output = NoteSequence([
            Note(pitch=50, onset_ticks=100, duration_ticks=400,
                velocity=80, string=2, fret=0)
        ])

        input_sequence = NoteSequence([
            Note(pitch=50, onset_ticks=96, duration_ticks=480,
                velocity=80)  # Slightly different timing
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        # Should use model timing, not input timing
        assert corrected.notes[0].onset_ticks == 100  # Model timing
        assert corrected.notes[0].duration_ticks == 400  # Model duration

    def test_window_size_limits_search(self):
        """Test that window size limits candidate search"""
        config = GuitarConfig()
        processor = PostProcessor(config, window_size=2)  # Small window

        # Model note at position 5
        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=500, duration_ticks=480,
                velocity=80, string=1, fret=0)
        ])

        # Input notes: one close, one far
        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80),    # Far away (500 ticks)
            Note(pitch=45, onset_ticks=100, duration_ticks=480,
                velocity=80),    # Still far
            Note(pitch=45, onset_ticks=200, duration_ticks=480,
                velocity=80),    # Within range
            Note(pitch=45, onset_ticks=480, duration_ticks=480,
                velocity=80),    # Close
            Note(pitch=45, onset_ticks=520, duration_ticks=480,
                velocity=80),    # Close
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        # Should find a match within window
        assert len(corrected) == 1
        assert corrected.notes[0].pitch == 45

    def test_multiple_notes_correction(self):
        """Test correction of sequence with multiple notes"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model output with mixed correct/incorrect pitches
        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),    # Correct
            Note(pitch=52, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=2),    # Wrong (should be 50)
            Note(pitch=55, onset_ticks=960, duration_ticks=480,
                velocity=80, string=3, fret=0),    # Correct
        ])

        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=50, onset_ticks=480, duration_ticks=480, velocity=80),
            Note(pitch=55, onset_ticks=960, duration_ticks=480, velocity=80),
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        assert len(corrected) == 3
        assert corrected.notes[0].pitch == 45  # Kept correct
        assert corrected.notes[1].pitch == 50  # Corrected
        assert corrected.notes[2].pitch == 55  # Kept correct

    def test_fallback_when_no_match(self):
        """Test fallback mechanism when no match found"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model has a note
        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=1000, duration_ticks=480,
                velocity=80, string=1, fret=0)
        ])

        # Input has no notes nearby (empty sequence)
        input_sequence = NoteSequence([])

        corrected = processor.overlap_correction(model_output, input_sequence)

        assert len(corrected) == 1
        assert corrected.notes[0].source == "fallback"
        assert corrected.notes[0].pitch == 45
        assert corrected.notes[0].has_tablature()

    def test_chord_handling(self):
        """Test handling of chords (simultaneous notes)"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model output: C major chord (C, E, G)
        model_output = NoteSequence([
            Note(pitch=48, onset_ticks=0, duration_ticks=480,
                velocity=80, string=0, fret=8),   # C
            Note(pitch=52, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=7),   # E
            Note(pitch=55, onset_ticks=0, duration_ticks=480,
                velocity=80, string=2, fret=5),   # G
        ])

        # Input sequence: same chord
        input_sequence = NoteSequence([
            Note(pitch=48, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=52, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=55, onset_ticks=0, duration_ticks=480, velocity=80),
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        assert len(corrected) == 3
        # All notes should be corrected (matched)
        pitches = sorted([note.pitch for note in corrected.notes])
        assert pitches == [48, 52, 55]

    def test_avoids_duplicate_matching(self):
        """Test that input notes are not matched multiple times"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Two model notes at same time
        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
        ])

        # Only one input note
        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        assert len(corrected) == 2
        # First should match input, second should be fallback
        sources = [note.source for note in corrected.notes]
        assert "corrected" in sources
        assert "fallback" in sources

    def test_preserves_string_choice_when_valid(self):
        """Test that model's string choice is preserved if valid"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model uses string 1 for pitch 45
        model_output = NoteSequence([
            Note(pitch=47, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=2)  # Wrong pitch, but valid string
        ])

        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80)
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        # Should try to keep string 1 if possible
        assert corrected.notes[0].string == 1  # String preserved
        assert corrected.notes[0].fret == 0    # Fret recalculated for pitch 45

    def test_corrects_invalid_string_choice(self):
        """Test that invalid string choices are corrected"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model uses invalid string
        model_output = NoteSequence([
            Note(pitch=50, onset_ticks=0, duration_ticks=480,
                velocity=80, string=10, fret=0)  # Invalid string!
        ])

        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80)
        ])

        corrected = processor.overlap_correction(model_output, input_sequence)

        # Should correct to valid string
        assert corrected.notes[0].string < config.num_strings
        assert corrected.notes[0].has_tablature()


class TestProcess:
    """Tests for PostProcessor.process()"""

    def test_process_without_neighbor_search(self):
        """Test process with neighbor search disabled"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0)
        ])

        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80)
        ])

        result = processor.process(
            model_output,
            input_sequence,
            apply_neighbor_search=False
        )

        assert result.source == "overlap_corrected"

    def test_process_with_neighbor_search(self):
        """Test process with neighbor search enabled"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0)
        ])

        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80)
        ])

        result = processor.process(
            model_output,
            input_sequence,
            apply_neighbor_search=True
        )

        # Should have applied neighbor search
        assert result.source == "neighbor_search"
        # Individual notes should have refined source
        assert all(note.source == "refined" for note in result.notes if note.has_tablature())


class TestPitchAccuracy:
    """Integration tests for pitch accuracy improvement"""

    def test_pitch_accuracy_improvement(self):
        """Test that pitch accuracy improves after correction"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Create model output where tablature produces WRONG pitches
        # The pitch field doesn't match what the tablature would produce
        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),     # Correct: string 1 (A2=45) + fret 0 = 45
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=2),     # Wrong: string 2 (D3=50) + fret 2 = 52, not 50
            Note(pitch=55, onset_ticks=960, duration_ticks=480,
                velocity=80, string=3, fret=3),     # Wrong: string 3 (G3=55) + fret 3 = 58, not 55
            Note(pitch=59, onset_ticks=1440, duration_ticks=480,
                velocity=80, string=4, fret=1),     # Wrong: string 4 (B3=59) + fret 1 = 60, not 59
        ])

        # Ground truth input
        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=50, onset_ticks=480, duration_ticks=480, velocity=80),
            Note(pitch=55, onset_ticks=960, duration_ticks=480, velocity=80),
            Note(pitch=59, onset_ticks=1440, duration_ticks=480, velocity=80),
        ])

        # Calculate accuracy before correction (25% - only first note is valid)
        accuracy_before = calculate_pitch_accuracy(model_output.notes, config)

        # Apply correction
        corrected = processor.overlap_correction(model_output, input_sequence)

        # Calculate accuracy after correction (should be 100%)
        accuracy_after = calculate_pitch_accuracy(corrected.notes, config)

        # Accuracy should improve (25% → 100% in this case)
        assert accuracy_after > accuracy_before
        assert accuracy_before == 25.0  # Only 1 out of 4 was correct
        assert accuracy_after == 100.0  # All corrected
        print(f"Accuracy improved: {accuracy_before:.2f}% → {accuracy_after:.2f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
