"""
Tests for api.py - FrettingPostProcessor API
"""

import unittest
from fretting_postprocessor import (
    Note, NoteSequence, GuitarConfig, STANDARD_TUNING,
    FrettingPostProcessor,
    process_tokens_quick, evaluate_quick
)


class TestFrettingPostProcessorInit(unittest.TestCase):
    """測試 FrettingPostProcessor 初始化"""

    def test_init_default_config(self):
        """測試預設配置初始化"""
        processor = FrettingPostProcessor()

        self.assertIsNotNone(processor.config)
        self.assertEqual(processor.config.tuning, STANDARD_TUNING)
        self.assertEqual(processor.config.num_strings, 6)

    def test_init_custom_config(self):
        """測試自訂配置初始化"""
        config = GuitarConfig(tuning=(38, 45, 50, 55, 59, 64))  # Drop-D
        processor = FrettingPostProcessor(config)

        self.assertEqual(processor.config.tuning, (38, 45, 50, 55, 59, 64))


class TestProcessTokens(unittest.TestCase):
    """測試 process_tokens 方法"""

    def setUp(self):
        self.processor = FrettingPostProcessor()

        # 建立測試 tokens
        # Input: C4 (60) quarter note
        self.input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>'
        ]

        # Model output: 錯誤的 pitch (61 instead of 60)
        self.model_output_tokens = [
            'TAB<2,6>',  # String 2, fret 6 = pitch 61 (C#4) - WRONG
            'TIME_SHIFT<480>'
        ]

    def test_process_overlap_method(self):
        """測試 overlap correction 方法"""
        corrected_tokens = self.processor.process_tokens(
            self.model_output_tokens,
            self.input_tokens,
            method='overlap'
        )

        # 應該返回 token list
        self.assertIsInstance(corrected_tokens, list)
        self.assertTrue(len(corrected_tokens) > 0)

        # 應該包含 TAB token
        tab_tokens = [t for t in corrected_tokens if t.startswith('TAB<')]
        self.assertTrue(len(tab_tokens) > 0)

    def test_process_neighbor_search_method(self):
        """測試 neighbor search 方法"""
        corrected_tokens = self.processor.process_tokens(
            self.model_output_tokens,
            self.input_tokens,
            method='neighbor_search'
        )

        # 應該返回 token list
        self.assertIsInstance(corrected_tokens, list)
        self.assertTrue(len(corrected_tokens) > 0)

    def test_process_invalid_method(self):
        """測試無效方法"""
        with self.assertRaises(ValueError):
            self.processor.process_tokens(
                self.model_output_tokens,
                self.input_tokens,
                method='invalid_method'
            )

    def test_process_corrects_pitch_error(self):
        """測試確實修正了 pitch 錯誤"""
        # Parse corrected output
        corrected_tokens = self.processor.process_tokens(
            self.model_output_tokens,
            self.input_tokens,
            method='neighbor_search'
        )

        # 重新 parse 回 NoteSequence 檢查
        input_sequence = self.processor.parser.parse_input_tokens(self.input_tokens)
        corrected_sequence = self.processor.parser.parse_output_tokens(
            corrected_tokens,
            input_sequence,
            self.processor.config
        )

        # 檢查修正後的 pitch
        self.assertEqual(len(corrected_sequence), 1)
        corrected_note = corrected_sequence.notes[0]

        # 應該修正為正確的 pitch (60)
        self.assertEqual(corrected_note.pitch, 60)


class TestEvaluate(unittest.TestCase):
    """測試 evaluate 方法"""

    def setUp(self):
        self.processor = FrettingPostProcessor()

        # 建立測試 tokens
        self.input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>',
            'NOTE_ON<62>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<62>'
        ]

        # Model output with errors
        self.model_output_tokens = [
            'TAB<2,6>',  # Wrong: 61 instead of 60
            'TIME_SHIFT<480>',
            'TAB<2,7>',  # Correct: 62
            'TIME_SHIFT<480>'
        ]

    def test_evaluate_returns_comparison(self):
        """測試 evaluate 返回比較結果"""
        results = self.processor.evaluate(
            self.model_output_tokens,
            self.input_tokens
        )

        # 應該包含三種方法的結果
        self.assertIn('raw_model', results)
        self.assertIn('overlap_correction', results)
        self.assertIn('neighbor_search', results)

        # 每個方法應該有必要的欄位
        for method in ['raw_model', 'overlap_correction', 'neighbor_search']:
            self.assertIn('pitch_accuracy', results[method])
            self.assertIn('tab_accuracy', results[method])
            self.assertIn('total_notes', results[method])

    def test_evaluate_accuracy_improvement(self):
        """測試 accuracy 有提升"""
        results = self.processor.evaluate(
            self.model_output_tokens,
            self.input_tokens
        )

        raw_pitch_acc = results['raw_model']['pitch_accuracy']
        neighbor_pitch_acc = results['neighbor_search']['pitch_accuracy']

        # Neighbor search 的 accuracy 應該 >= raw model
        self.assertGreaterEqual(neighbor_pitch_acc, raw_pitch_acc)

        # Neighbor search 應該達到高 accuracy
        self.assertGreater(neighbor_pitch_acc, 90.0)


class TestProcessAndEvaluate(unittest.TestCase):
    """測試 process_and_evaluate 方法"""

    def setUp(self):
        self.processor = FrettingPostProcessor()

        self.input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>'
        ]

        self.model_output_tokens = [
            'TAB<2,6>',  # Wrong pitch
            'TIME_SHIFT<480>'
        ]

    def test_process_and_evaluate(self):
        """測試同時處理和評估"""
        corrected_tokens, evaluation = self.processor.process_and_evaluate(
            self.model_output_tokens,
            self.input_tokens,
            method='neighbor_search',
            verbose=False  # 不打印
        )

        # 應該返回兩個元素
        self.assertIsInstance(corrected_tokens, list)
        self.assertIsInstance(evaluation, dict)

        # Evaluation 應該包含三種方法
        self.assertIn('raw_model', evaluation)
        self.assertIn('overlap_correction', evaluation)
        self.assertIn('neighbor_search', evaluation)


class TestGetNoteSequences(unittest.TestCase):
    """測試 get_note_sequences 方法"""

    def setUp(self):
        self.processor = FrettingPostProcessor()

        self.input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>'
        ]

        self.model_output_tokens = [
            'TAB<2,6>',
            'TIME_SHIFT<480>'
        ]

    def test_get_sequences_overlap(self):
        """測試取得序列（overlap method）"""
        sequences = self.processor.get_note_sequences(
            self.model_output_tokens,
            self.input_tokens,
            method='overlap'
        )

        # 應該包含三個序列
        self.assertIn('input', sequences)
        self.assertIn('model_output', sequences)
        self.assertIn('overlap_corrected', sequences)

        # 不應該包含 neighbor_refined
        self.assertNotIn('neighbor_refined', sequences)

    def test_get_sequences_neighbor_search(self):
        """測試取得序列（neighbor_search method）"""
        sequences = self.processor.get_note_sequences(
            self.model_output_tokens,
            self.input_tokens,
            method='neighbor_search'
        )

        # 應該包含四個序列
        self.assertIn('input', sequences)
        self.assertIn('model_output', sequences)
        self.assertIn('overlap_corrected', sequences)
        self.assertIn('neighbor_refined', sequences)

    def test_sequences_are_note_sequence_objects(self):
        """測試返回的是 NoteSequence 物件"""
        sequences = self.processor.get_note_sequences(
            self.model_output_tokens,
            self.input_tokens,
            method='neighbor_search'
        )

        for key, seq in sequences.items():
            self.assertIsInstance(seq, NoteSequence, f"{key} should be NoteSequence")


class TestChangeGuitarConfig(unittest.TestCase):
    """測試 change_guitar_config 方法"""

    def test_change_config(self):
        """測試更改配置"""
        processor = FrettingPostProcessor()

        # 初始配置
        self.assertEqual(processor.config.tuning, STANDARD_TUNING)

        # 更改配置
        new_config = GuitarConfig(tuning=(38, 45, 50, 55, 59, 64))  # Drop-D
        processor.change_guitar_config(new_config)

        # 驗證配置已更改
        self.assertEqual(processor.config.tuning, (38, 45, 50, 55, 59, 64))

    def test_get_config(self):
        """測試取得配置"""
        processor = FrettingPostProcessor()
        config = processor.get_config()

        self.assertIsInstance(config, GuitarConfig)
        self.assertEqual(config.tuning, STANDARD_TUNING)


class TestConvenienceFunctions(unittest.TestCase):
    """測試便捷函數"""

    def setUp(self):
        self.input_tokens = [
            'NOTE_ON<60>',
            'TIME_SHIFT<480>',
            'NOTE_OFF<60>'
        ]

        self.model_output_tokens = [
            'TAB<2,6>',
            'TIME_SHIFT<480>'
        ]

    def test_process_tokens_quick(self):
        """測試快速處理函數"""
        corrected_tokens = process_tokens_quick(
            self.model_output_tokens,
            self.input_tokens
        )

        self.assertIsInstance(corrected_tokens, list)
        self.assertTrue(len(corrected_tokens) > 0)

    def test_process_tokens_quick_custom_tuning(self):
        """測試快速處理函數（自訂 tuning）"""
        corrected_tokens = process_tokens_quick(
            self.model_output_tokens,
            self.input_tokens,
            tuning=(38, 45, 50, 55, 59, 64)  # Drop-D
        )

        self.assertIsInstance(corrected_tokens, list)

    def test_evaluate_quick(self):
        """測試快速評估函數"""
        results = evaluate_quick(
            self.model_output_tokens,
            self.input_tokens,
            verbose=False
        )

        self.assertIsInstance(results, dict)
        self.assertIn('raw_model', results)
        self.assertIn('overlap_correction', results)
        self.assertIn('neighbor_search', results)


class TestEndToEndWorkflow(unittest.TestCase):
    """端到端工作流程測試"""

    def test_full_pipeline(self):
        """測試完整 pipeline"""
        # 建立有多個音符的測試案例
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
            'TAB<2,5>',
            'TIME_SHIFT<240>',
            'TAB<2,8>',  # Wrong: should be <2,7>
            'TIME_SHIFT<240>',
            'TAB<2,9>',
            'TIME_SHIFT<480>'
        ]

        # 初始化 processor
        processor = FrettingPostProcessor()

        # 處理
        corrected_tokens = processor.process_tokens(
            model_output_tokens,
            input_tokens,
            method='neighbor_search'
        )

        # 驗證輸出
        self.assertIsInstance(corrected_tokens, list)
        self.assertTrue(len(corrected_tokens) > 0)

        # 評估
        results = processor.evaluate(model_output_tokens, input_tokens)

        # 驗證 neighbor_search 達到高 accuracy
        neighbor_acc = results['neighbor_search']['pitch_accuracy']
        self.assertGreater(neighbor_acc, 90.0)


if __name__ == '__main__':
    unittest.main()
