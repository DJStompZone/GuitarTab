"""
Integration Tests
==================

測試完整 pipeline 的整合測試，包括端到端工作流程。
"""

import unittest
import os
import tempfile
import json
from fretting_postprocessor import (
    Note, NoteSequence, GuitarConfig, STANDARD_TUNING,
    FrettingPostProcessor, TokenParser, TokenSerializer,
    PostProcessor, PitchValidator
)


class TestCompletePostProcessingPipeline(unittest.TestCase):
    """測試完整的 post-processing pipeline"""

    def setUp(self):
        """設定測試環境"""
        self.config = GuitarConfig(tuning=STANDARD_TUNING)
        self.parser = TokenParser()
        self.serializer = TokenSerializer()
        self.processor = PostProcessor(self.config)
        self.validator = PitchValidator()

    def test_end_to_end_single_note(self):
        """測試單音符的端到端處理"""
        # Step 1: Input tokens (ground truth pitch)
        input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>'
        ]

        # Step 2: Model output (with pitch error)
        model_output_tokens = [
            'TAB<2,6>',  # Wrong: pitch 61 instead of 60
            'TIME_SHIFT<480>'
        ]

        # Step 3: Parse
        input_sequence = self.parser.parse_input_tokens(input_tokens)
        model_output = self.parser.parse_output_tokens(
            model_output_tokens,
            input_sequence,
            self.config
        )

        # Step 4: Overlap correction
        overlap_corrected = self.processor.overlap_correction(
            model_output,
            input_sequence
        )

        # Step 5: Neighbor search
        neighbor_refined = self.processor.neighbor_search(
            overlap_corrected,
            optimize_for='balanced'
        )

        # Step 6: Validate result
        self.assertEqual(len(neighbor_refined), 1)
        final_note = neighbor_refined.notes[0]

        # Should correct to pitch 60
        self.assertEqual(final_note.pitch, 60)

        # Step 7: Serialize back to tokens
        output_tokens = self.serializer.serialize_to_output_format(neighbor_refined)

        # Verify tokens
        self.assertIsInstance(output_tokens, list)
        self.assertTrue(any('TAB<' in token for token in output_tokens))

    def test_end_to_end_multiple_notes(self):
        """測試多音符序列的端到端處理"""
        # Input: C-D-E melody
        input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<240>',
            'NOTE_OFF<60>',
            'NOTE_ON<62>',
            'TIME_SHIFT<240>',
            'NOTE_OFF<62>',
            'NOTE_ON<64>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<64>'
        ]

        # Model output with some errors
        model_output_tokens = [
            'TAB<2,5>',  # Correct: 60
            'TIME_SHIFT<240>',
            'TAB<2,8>',  # Wrong: 63 instead of 62
            'TIME_SHIFT<240>',
            'TAB<2,9>',  # Correct: 64
            'TIME_SHIFT<480>'
        ]

        # Parse
        input_sequence = self.parser.parse_input_tokens(input_tokens)
        model_output = self.parser.parse_output_tokens(
            model_output_tokens,
            input_sequence,
            self.config
        )

        # Process
        overlap_corrected = self.processor.overlap_correction(
            model_output,
            input_sequence
        )
        neighbor_refined = self.processor.neighbor_search(
            overlap_corrected,
            optimize_for='balanced'
        )

        # Validate
        self.assertEqual(len(neighbor_refined), 3)

        # Check pitches are corrected
        pitches = [note.pitch for note in neighbor_refined.notes]
        self.assertEqual(pitches, [60, 62, 64])

    def test_end_to_end_with_chord(self):
        """測試和弦（同時多音）的處理"""
        # Input: C major chord (C-E-G)
        input_tokens = [
            'NOTE_ON<60>',  # C
            'NOTE_ON<64>',  # E
            'NOTE_ON<67>',  # G
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>',
            'NOTE_OFF<64>',
            'NOTE_OFF<67>'
        ]

        # Model output
        model_output_tokens = [
            'TAB<2,5>',  # C (60)
            'TAB<1,4>',  # E (64)
            'TAB<0,3>',  # G (67)
            'TIME_SHIFT<480>'
        ]

        # Parse
        input_sequence = self.parser.parse_input_tokens(input_tokens)
        model_output = self.parser.parse_output_tokens(
            model_output_tokens,
            input_sequence,
            self.config
        )

        # Process
        overlap_corrected = self.processor.overlap_correction(
            model_output,
            input_sequence
        )
        neighbor_refined = self.processor.neighbor_search(
            overlap_corrected,
            optimize_for='balanced'
        )

        # Validate
        self.assertEqual(len(neighbor_refined), 3)

        # Check all pitches are present
        pitches = sorted([note.pitch for note in neighbor_refined.notes])
        self.assertEqual(pitches, [60, 64, 67])

    def test_empty_sequence(self):
        """測試空序列處理"""
        input_tokens = []
        model_output_tokens = []

        input_sequence = NoteSequence([], source="input")
        model_output = NoteSequence([], source="model_output")

        # Should not crash
        overlap_corrected = self.processor.overlap_correction(
            model_output,
            input_sequence
        )

        self.assertEqual(len(overlap_corrected), 0)


class TestPipelineWithAPI(unittest.TestCase):
    """測試使用高階 API 的 pipeline"""

    def setUp(self):
        self.processor = FrettingPostProcessor()

    def test_process_tokens_full_pipeline(self):
        """測試使用 API 處理 tokens"""
        input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>',
            'NOTE_ON<62>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<62>'
        ]

        model_output_tokens = [
            'TAB<2,6>',  # Wrong
            'TIME_SHIFT<480>',
            'TAB<2,7>',  # Correct
            'TIME_SHIFT<480>'
        ]

        # Process
        corrected_tokens = self.processor.process_tokens(
            model_output_tokens,
            input_tokens,
            method='neighbor_search'
        )

        # Verify output is tokens
        self.assertIsInstance(corrected_tokens, list)
        self.assertTrue(len(corrected_tokens) > 0)

        # Verify contains TAB tokens
        tab_tokens = [t for t in corrected_tokens if t.startswith('TAB<')]
        self.assertEqual(len(tab_tokens), 2)

    def test_evaluate_full_pipeline(self):
        """測試使用 API 評估"""
        input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>',
            'NOTE_ON<62>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<62>',
            'NOTE_ON<64>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<64>'
        ]

        model_output_tokens = [
            'TAB<2,6>',  # Wrong: 61
            'TIME_SHIFT<480>',
            'TAB<2,7>',  # Correct: 62
            'TIME_SHIFT<480>',
            'TAB<2,9>',  # Correct: 64
            'TIME_SHIFT<480>'
        ]

        # Evaluate
        results = self.processor.evaluate(
            model_output_tokens,
            input_tokens
        )

        # Verify structure
        self.assertIn('raw_model', results)
        self.assertIn('overlap_correction', results)
        self.assertIn('neighbor_search', results)

        # Verify neighbor_search has high accuracy
        neighbor_acc = results['neighbor_search']['pitch_accuracy']
        self.assertGreater(neighbor_acc, 90.0)


class TestEdgeCases(unittest.TestCase):
    """測試邊界情況和特殊案例"""

    def setUp(self):
        self.config = GuitarConfig(tuning=STANDARD_TUNING)
        self.processor = PostProcessor(self.config)

    def test_out_of_range_high_pitch(self):
        """測試超出吉他音域的高音"""
        # Create note with pitch 100 (very high)
        input_sequence = NoteSequence([
            Note(pitch=100, onset_ticks=0, duration_ticks=480, velocity=80)
        ], source="input")

        model_output = NoteSequence([
            Note(pitch=99, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=0, fret=35)  # Invalid fret
        ], source="model")

        # Should handle gracefully
        try:
            corrected = self.processor.overlap_correction(
                model_output,
                input_sequence
            )
            # Should still produce output
            self.assertIsInstance(corrected, NoteSequence)
        except Exception as e:
            self.fail(f"Should handle out of range pitch gracefully, but raised: {e}")

    def test_out_of_range_low_pitch(self):
        """測試超出吉他音域的低音"""
        # Create note with pitch 30 (very low)
        input_sequence = NoteSequence([
            Note(pitch=30, onset_ticks=0, duration_ticks=480, velocity=80)
        ], source="input")

        model_output = NoteSequence([
            Note(pitch=31, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=5, fret=-10)  # Invalid fret
        ], source="model")

        # Should handle gracefully
        try:
            corrected = self.processor.overlap_correction(
                model_output,
                input_sequence
            )
            self.assertIsInstance(corrected, NoteSequence)
        except Exception as e:
            self.fail(f"Should handle out of range pitch gracefully, but raised: {e}")

    def test_very_long_sequence(self):
        """測試長序列（性能測試）"""
        # Create 100 notes
        notes = []
        for i in range(100):
            pitch = 60 + (i % 12)  # C major scale
            onset = i * 240
            notes.append(Note(
                pitch=pitch,
                onset_ticks=onset,
                duration_ticks=240,
                velocity=80
            ))

        input_sequence = NoteSequence(notes, source="input")

        # Model output with some errors
        model_notes = []
        for i, note in enumerate(notes):
            # Every 5th note has wrong pitch
            if i % 5 == 0:
                wrong_pitch = note.pitch + 1
            else:
                wrong_pitch = note.pitch

            # Find string/fret for wrong pitch
            positions = self.config.pitch_to_string_fret(wrong_pitch)
            if positions:
                string, fret = positions[0]
            else:
                string, fret = 0, 0

            model_notes.append(Note(
                pitch=wrong_pitch,
                onset_ticks=note.onset_ticks,
                duration_ticks=note.duration_ticks,
                velocity=note.velocity,
                string=string,
                fret=fret
            ))

        model_output = NoteSequence(model_notes, source="model")

        # Process (should complete in reasonable time)
        import time
        start = time.time()

        corrected = self.processor.overlap_correction(model_output, input_sequence)
        refined = self.processor.neighbor_search(corrected, optimize_for='balanced')

        elapsed = time.time() - start

        # Should complete in < 5 seconds
        self.assertLess(elapsed, 5.0, f"Processing 100 notes took {elapsed:.2f}s")

        # Should produce correct number of notes
        self.assertEqual(len(refined), 100)

    def test_duplicate_notes_same_time(self):
        """測試同一時間的重複音符"""
        input_sequence = NoteSequence([
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80),
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80)  # Duplicate
        ], source="input")

        model_output = NoteSequence([
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5)
        ], source="model")

        # Should handle duplicates
        corrected = self.processor.overlap_correction(model_output, input_sequence)
        self.assertIsInstance(corrected, NoteSequence)


class TestDifferentTunings(unittest.TestCase):
    """測試不同 tuning 的處理"""

    def test_drop_d_tuning(self):
        """測試 Drop-D tuning"""
        from fretting_postprocessor import DROP_D_TUNING

        config = GuitarConfig(tuning=DROP_D_TUNING)
        processor = PostProcessor(config)

        # Low D (pitch 50)
        input_sequence = NoteSequence([
            Note(pitch=50, onset_ticks=0, duration_ticks=480, velocity=80)
        ], source="input")

        # Model output
        model_output = NoteSequence([
            Note(pitch=51, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=5, fret=1)  # Wrong pitch
        ], source="model")

        # Process
        corrected = processor.overlap_correction(model_output, input_sequence)
        refined = processor.neighbor_search(corrected, optimize_for='balanced')

        # Should correct to pitch 50
        self.assertEqual(refined.notes[0].pitch, 50)

        # Should have valid tablature (any valid position is acceptable)
        self.assertTrue(refined.notes[0].has_tablature())

        # Verify the tablature produces correct pitch
        calculated_pitch = config.tuning[refined.notes[0].string] + refined.notes[0].fret
        self.assertEqual(calculated_pitch, 50)

    def test_half_step_down(self):
        """測試降半音 tuning"""
        from fretting_postprocessor import HALF_STEP_DOWN

        config = GuitarConfig(tuning=HALF_STEP_DOWN)
        processor = FrettingPostProcessor(config)

        input_tokens = [
            'NOTE_ON<59>',  # Eb (half step down from standard E)
            'TIME_SHIFT<480>',
            'NOTE_OFF<59>'
        ]

        model_output_tokens = [
            'TAB<5,1>',  # Wrong fret
            'TIME_SHIFT<480>'
        ]

        # Process
        corrected_tokens = processor.process_tokens(
            model_output_tokens,
            input_tokens,
            method='neighbor_search'
        )

        # Should produce valid output
        self.assertIsInstance(corrected_tokens, list)
        self.assertTrue(len(corrected_tokens) > 0)


if __name__ == '__main__':
    unittest.main()
