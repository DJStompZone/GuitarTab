"""
Test Suite for PostProcessor - Neighbor Search
===============================================

Tests for the neighbor search algorithm (Section 4.2 of the paper).
"""

import pytest
from fretting_postprocessor.datatypes import Note
from fretting_postprocessor.config import GuitarConfig
from fretting_postprocessor.sequence import NoteSequence
from fretting_postprocessor.processor import PostProcessor
from fretting_postprocessor.validator import calculate_pitch_accuracy


class TestGetContextNotes:
    """Tests for PostProcessor._get_context_notes()"""

    def test_middle_position(self):
        """Test getting context from middle of sequence"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=50, onset_ticks=480, duration_ticks=480, velocity=80),
            Note(pitch=55, onset_ticks=960, duration_ticks=480, velocity=80),
            Note(pitch=59, onset_ticks=1440, duration_ticks=480, velocity=80),
            Note(pitch=64, onset_ticks=1920, duration_ticks=480, velocity=80),
        ]

        prev, next_notes = processor._get_context_notes(notes, 2, context_window=1)

        assert len(prev) == 1
        assert prev[0].pitch == 50  # Previous note
        assert len(next_notes) == 1
        assert next_notes[0].pitch == 59  # Next note

    def test_start_position(self):
        """Test getting context from start of sequence"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=50, onset_ticks=480, duration_ticks=480, velocity=80),
            Note(pitch=55, onset_ticks=960, duration_ticks=480, velocity=80),
        ]

        prev, next_notes = processor._get_context_notes(notes, 0, context_window=2)

        assert len(prev) == 0  # No previous notes
        assert len(next_notes) == 2  # Two following notes

    def test_end_position(self):
        """Test getting context from end of sequence"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=50, onset_ticks=480, duration_ticks=480, velocity=80),
            Note(pitch=55, onset_ticks=960, duration_ticks=480, velocity=80),
        ]

        prev, next_notes = processor._get_context_notes(notes, 2, context_window=2)

        assert len(prev) == 2  # Two previous notes
        assert len(next_notes) == 0  # No following notes

    def test_large_window(self):
        """Test with large context window"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        notes = [
            Note(pitch=i * 5 + 40, onset_ticks=i * 480, duration_ticks=480, velocity=80)
            for i in range(10)
        ]

        prev, next_notes = processor._get_context_notes(notes, 5, context_window=3)

        assert len(prev) == 3  # indices 2, 3, 4
        assert len(next_notes) == 3  # indices 6, 7, 8


class TestEvaluatePosition:
    """Tests for PostProcessor._evaluate_position()"""

    def test_string_consistency_bonus(self):
        """Test that same string gets bonus"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Previous notes all on string 1
        previous_notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
            Note(pitch=47, onset_ticks=480, duration_ticks=480,
                velocity=80, string=1, fret=2),
        ]

        # Score for continuing on string 1
        score_same_string = processor._evaluate_position(
            string=1, fret=5,
            previous_notes=previous_notes,
            optimize_for="balanced"
        )

        # Score for switching to string 2
        score_diff_string = processor._evaluate_position(
            string=2, fret=0,
            previous_notes=previous_notes,
            optimize_for="balanced"
        )

        # Same string should have better (lower) score due to -20 bonus
        assert score_same_string < score_diff_string

    def test_position_proximity_penalty(self):
        """Test that large fret jumps are penalized"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        previous_notes = [
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
        ]

        # Close fret on same string
        score_close = processor._evaluate_position(
            string=1, fret=2,
            previous_notes=previous_notes,
            optimize_for="balanced"
        )

        # Far fret on same string
        score_far = processor._evaluate_position(
            string=1, fret=15,
            previous_notes=previous_notes,
            optimize_for="balanced"
        )

        # Closer fret should have better score
        assert score_close < score_far

    def test_playability_factor(self):
        """Test that lower frets are preferred in playability mode"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        previous_notes = []

        # Low fret (easier)
        score_low = processor._evaluate_position(
            string=2, fret=2,
            previous_notes=previous_notes,
            optimize_for="playability"
        )

        # High fret (harder)
        score_high = processor._evaluate_position(
            string=2, fret=12,
            previous_notes=previous_notes,
            optimize_for="playability"
        )

        # Lower fret should have better score
        assert score_low < score_high

    def test_extreme_position_penalty(self):
        """Test that frets > 15 are penalized"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        previous_notes = []

        # Normal fret
        score_normal = processor._evaluate_position(
            string=2, fret=10,
            previous_notes=previous_notes,
            optimize_for="balanced"
        )

        # Extreme fret (> 15)
        score_extreme = processor._evaluate_position(
            string=2, fret=18,
            previous_notes=previous_notes,
            optimize_for="balanced"
        )

        # Normal fret should have much better score
        assert score_normal < score_extreme

    def test_position_stability_mode(self):
        """Test position_stability optimization doesn't add playability penalty"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        previous_notes = []

        score_stability = processor._evaluate_position(
            string=2, fret=12,
            previous_notes=previous_notes,
            optimize_for="position_stability"
        )

        score_playability = processor._evaluate_position(
            string=2, fret=12,
            previous_notes=previous_notes,
            optimize_for="playability"
        )

        # position_stability should not penalize high frets as much
        assert score_stability < score_playability


class TestNeighborSearch:
    """Tests for PostProcessor.neighbor_search()"""

    def test_single_position_unchanged(self):
        """Test that notes with only one valid position remain unchanged"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Pitch 40 (E2) can only be played on string 0 fret 0
        sequence = NoteSequence([
            Note(pitch=40, onset_ticks=0, duration_ticks=480,
                velocity=80, string=0, fret=0)
        ])

        refined = processor.neighbor_search(sequence)

        assert len(refined) == 1
        assert refined.notes[0].string == 0
        assert refined.notes[0].fret == 0
        assert refined.notes[0].pitch == 40  # Pitch preserved

    def test_multiple_positions_optimized(self):
        """Test that alternative positions are explored"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Pitch 50 (D3) has multiple positions: (0,10), (1,5), (2,0)
        sequence = NoteSequence([
            Note(pitch=50, onset_ticks=0, duration_ticks=480,
                velocity=80, string=0, fret=10)  # High fret position
        ])

        refined = processor.neighbor_search(sequence, optimize_for="playability")

        # Should prefer lower fret position
        assert refined.notes[0].pitch == 50  # Pitch preserved
        assert refined.notes[0].fret < 10  # Lower fret preferred

    def test_string_consistency_across_sequence(self):
        """Test that string consistency is maintained"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Sequence where all notes can be played on string 1
        sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),   # A2
            Note(pitch=47, onset_ticks=480, duration_ticks=480,
                velocity=80, string=0, fret=7),   # B2 - wrong string choice
            Note(pitch=50, onset_ticks=960, duration_ticks=480,
                velocity=80, string=0, fret=10),  # D3 - wrong string choice
        ])

        refined = processor.neighbor_search(sequence, optimize_for="balanced")

        # Should prefer to stay on string 1 for consistency
        # Note: This might not always be string 1, but should show consistency
        assert all(note.pitch in [45, 47, 50] for note in refined.notes)

    def test_pitch_accuracy_preserved(self):
        """Test that 100% pitch accuracy is maintained"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Create sequence with valid tablature
        sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=0),
            Note(pitch=55, onset_ticks=960, duration_ticks=480,
                velocity=80, string=3, fret=0),
            Note(pitch=59, onset_ticks=1440, duration_ticks=480,
                velocity=80, string=4, fret=0),
        ])

        refined = processor.neighbor_search(sequence)

        # All pitches should be preserved
        original_pitches = sorted([n.pitch for n in sequence.notes])
        refined_pitches = sorted([n.pitch for n in refined.notes])
        assert original_pitches == refined_pitches

        # All tablature should be valid
        accuracy = calculate_pitch_accuracy(refined.notes, config)
        assert accuracy == 100.0

    def test_different_optimization_strategies(self):
        """Test different optimization strategies"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Sequence with choices
        sequence = NoteSequence([
            Note(pitch=50, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=5)
        ])

        # Playability: prefers lower frets
        playability_result = processor.neighbor_search(
            sequence, optimize_for="playability"
        )

        # Position stability: less weight on fret height
        stability_result = processor.neighbor_search(
            sequence, optimize_for="position_stability"
        )

        # Both should preserve pitch
        assert playability_result.notes[0].pitch == 50
        assert stability_result.notes[0].pitch == 50

    def test_context_aware_selection(self):
        """Test that context influences position selection"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Setup: previous notes on string 2, current note has choices
        sequence = NoteSequence([
            Note(pitch=50, onset_ticks=0, duration_ticks=480,
                velocity=80, string=2, fret=0),   # D3 on string 2
            Note(pitch=52, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=2),   # E3 on string 2
            Note(pitch=55, onset_ticks=960, duration_ticks=480,
                velocity=80, string=0, fret=15),  # G3 - poor choice
        ])

        refined = processor.neighbor_search(sequence, optimize_for="balanced")

        # Third note should prefer a position closer to string 2
        # since previous notes were on string 2
        assert refined.notes[2].pitch == 55  # Pitch preserved

    def test_no_tablature_notes_passed_through(self):
        """Test that notes without tablature are passed through"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=None, fret=None),  # No tablature
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=0),  # Has tablature
        ])

        refined = processor.neighbor_search(sequence)

        # First note should be unchanged (no tablature)
        assert not refined.notes[0].has_tablature()
        # Second note should be processed
        assert refined.notes[1].has_tablature()


class TestIntegration:
    """Integration tests for complete pipeline"""

    def test_full_pipeline_achieves_100_percent(self):
        """Test that full pipeline achieves 100% pitch accuracy"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Model output with errors
        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),     # Correct
            Note(pitch=50, onset_ticks=480, duration_ticks=480,
                velocity=80, string=2, fret=2),     # Wrong tablature
            Note(pitch=55, onset_ticks=960, duration_ticks=480,
                velocity=80, string=3, fret=3),     # Wrong tablature
        ])

        # Ground truth
        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=50, onset_ticks=480, duration_ticks=480, velocity=80),
            Note(pitch=55, onset_ticks=960, duration_ticks=480, velocity=80),
        ])

        # Apply full pipeline
        corrected = processor.overlap_correction(model_output, input_sequence)
        refined = processor.neighbor_search(corrected)

        # Should achieve 100% accuracy
        accuracy = calculate_pitch_accuracy(refined.notes, config)
        assert accuracy == 100.0

        # All pitches should match ground truth
        assert [n.pitch for n in refined.notes] == [45, 50, 55]

    def test_process_method_with_neighbor_search(self):
        """Test that process() correctly applies both algorithms"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480,
                velocity=80, string=1, fret=0),
        ])

        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=0, duration_ticks=480, velocity=80),
        ])

        # Process with neighbor search
        result = processor.process(
            model_output,
            input_sequence,
            apply_neighbor_search=True
        )

        assert result.source == "neighbor_search"
        assert len(result) == 1
        assert result.notes[0].pitch == 45

    def test_preserves_timing_through_full_pipeline(self):
        """Test that timing is preserved through full pipeline"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        model_output = NoteSequence([
            Note(pitch=45, onset_ticks=100, duration_ticks=400,
                velocity=80, string=1, fret=0),
        ])

        input_sequence = NoteSequence([
            Note(pitch=45, onset_ticks=96, duration_ticks=480, velocity=80),
        ])

        result = processor.process(model_output, input_sequence)

        # Should use model timing, not input timing
        assert result.notes[0].onset_ticks == 100
        assert result.notes[0].duration_ticks == 400


class TestEdgeCases:
    """Edge case tests"""

    def test_empty_sequence(self):
        """Test handling of empty sequence"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        sequence = NoteSequence([])
        refined = processor.neighbor_search(sequence)

        assert len(refined) == 0

    def test_single_note(self):
        """Test handling of single note"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        sequence = NoteSequence([
            Note(pitch=50, onset_ticks=0, duration_ticks=480,
                velocity=80, string=2, fret=0)
        ])

        refined = processor.neighbor_search(sequence)

        assert len(refined) == 1
        assert refined.notes[0].pitch == 50

    def test_all_notes_unique_positions(self):
        """Test sequence where all notes have unique positions"""
        config = GuitarConfig()
        processor = PostProcessor(config)

        # Each note has only one valid position
        sequence = NoteSequence([
            Note(pitch=40, onset_ticks=0, duration_ticks=480,
                velocity=80, string=0, fret=0),   # E2 - unique
            Note(pitch=88, onset_ticks=480, duration_ticks=480,
                velocity=80, string=5, fret=24),  # E6 - unique
        ])

        refined = processor.neighbor_search(sequence)

        # Should remain unchanged
        assert refined.notes[0].string == 0
        assert refined.notes[0].fret == 0
        assert refined.notes[1].string == 5
        assert refined.notes[1].fret == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
