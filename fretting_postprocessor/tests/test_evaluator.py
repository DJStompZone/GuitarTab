"""
Tests for evaluator.py - PostProcessingEvaluator
"""

import unittest
from fretting_postprocessor import (
    Note, NoteSequence, GuitarConfig, STANDARD_TUNING,
    PostProcessingEvaluator
)


class TestEvaluatePitchAccuracy(unittest.TestCase):
    """測試 evaluate_pitch_accuracy 方法"""

    def setUp(self):
        self.config = GuitarConfig(tuning=STANDARD_TUNING)

    def test_perfect_match(self):
        """測試完美匹配（100% accuracy）"""
        # 建立相同的序列
        notes = [
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5),
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=2, fret=7)
        ]

        predicted = NoteSequence(notes.copy(), source="predicted")
        ground_truth = NoteSequence(notes.copy(), source="ground_truth")

        results = PostProcessingEvaluator.evaluate_pitch_accuracy(
            predicted, ground_truth, self.config
        )

        self.assertEqual(results['pitch_accuracy'], 100.0)
        self.assertEqual(results['tab_accuracy'], 100.0)
        self.assertEqual(results['total_notes'], 2)
        self.assertEqual(results['pitch_correct'], 2)
        self.assertEqual(results['tab_correct'], 2)
        self.assertEqual(len(results['pitch_errors']), 0)
        self.assertEqual(len(results['tab_errors']), 0)

    def test_pitch_errors(self):
        """測試 pitch 錯誤檢測"""
        # Ground truth
        gt_notes = [
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5),
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=2, fret=7)
        ]

        # Predicted (第一個音符 pitch 錯誤)
        pred_notes = [
            Note(pitch=61, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=6),  # Wrong pitch
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=2, fret=7)
        ]

        predicted = NoteSequence(pred_notes, source="predicted")
        ground_truth = NoteSequence(gt_notes, source="ground_truth")

        results = PostProcessingEvaluator.evaluate_pitch_accuracy(
            predicted, ground_truth, self.config
        )

        self.assertEqual(results['pitch_accuracy'], 50.0)  # 1/2 correct
        self.assertEqual(results['pitch_correct'], 1)
        self.assertEqual(len(results['pitch_errors']), 1)
        self.assertEqual(results['pitch_errors'][0]['gt_pitch'], 60)
        self.assertEqual(results['pitch_errors'][0]['pred_pitch'], 61)

    def test_tablature_errors(self):
        """測試 tablature 錯誤檢測（pitch 正確但 string/fret 不同）"""
        # Ground truth
        gt_notes = [
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5)  # D3 string, fret 5
        ]

        # Predicted (同樣 pitch 但不同 string/fret)
        pred_notes = [
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=1, fret=10)  # A2 string, fret 10 (同樣產生 C4)
        ]

        predicted = NoteSequence(pred_notes, source="predicted")
        ground_truth = NoteSequence(gt_notes, source="ground_truth")

        results = PostProcessingEvaluator.evaluate_pitch_accuracy(
            predicted, ground_truth, self.config
        )

        # Pitch 正確，tablature 錯誤
        self.assertEqual(results['pitch_accuracy'], 100.0)
        self.assertEqual(results['tab_accuracy'], 0.0)
        self.assertEqual(len(results['pitch_errors']), 0)
        self.assertEqual(len(results['tab_errors']), 1)

    def test_time_tolerance(self):
        """測試時間容差配對"""
        # Ground truth
        gt_notes = [
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5)
        ]

        # Predicted (時間略有偏移)
        pred_notes = [
            Note(pitch=60, onset_ticks=50, duration_ticks=480, velocity=80,
                 string=2, fret=5)  # 50 ticks 偏移
        ]

        predicted = NoteSequence(pred_notes, source="predicted")
        ground_truth = NoteSequence(gt_notes, source="ground_truth")

        # 使用預設容差（120 ticks）- 應該配對成功
        results = PostProcessingEvaluator.evaluate_pitch_accuracy(
            predicted, ground_truth, self.config, time_tolerance=120
        )

        self.assertEqual(results['pitch_accuracy'], 100.0)

        # 使用較小容差（30 ticks）- 應該配對失敗
        results_strict = PostProcessingEvaluator.evaluate_pitch_accuracy(
            predicted, ground_truth, self.config, time_tolerance=30
        )

        self.assertEqual(results_strict['pitch_accuracy'], 0.0)

    def test_missing_notes(self):
        """測試缺少音符的情況"""
        # Ground truth 有 2 個音符
        gt_notes = [
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5),
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=2, fret=7)
        ]

        # Predicted 只有 1 個音符
        pred_notes = [
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5)
        ]

        predicted = NoteSequence(pred_notes, source="predicted")
        ground_truth = NoteSequence(gt_notes, source="ground_truth")

        results = PostProcessingEvaluator.evaluate_pitch_accuracy(
            predicted, ground_truth, self.config
        )

        self.assertEqual(results['pitch_accuracy'], 50.0)  # 1/2 correct
        self.assertEqual(len(results['pitch_errors']), 1)
        self.assertIsNone(results['pitch_errors'][0]['pred_pitch'])

    def test_empty_sequence(self):
        """測試空序列"""
        predicted = NoteSequence([], source="predicted")
        ground_truth = NoteSequence([], source="ground_truth")

        results = PostProcessingEvaluator.evaluate_pitch_accuracy(
            predicted, ground_truth, self.config
        )

        self.assertEqual(results['pitch_accuracy'], 100.0)
        self.assertEqual(results['total_notes'], 0)


class TestCompareMethods(unittest.TestCase):
    """測試 compare_methods 方法"""

    def setUp(self):
        self.config = GuitarConfig(tuning=STANDARD_TUNING)

        # 建立測試序列
        # Ground truth (完美)
        self.ground_truth = NoteSequence([
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5),
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=2, fret=7),
            Note(pitch=64, onset_ticks=960, duration_ticks=480, velocity=80,
                 string=2, fret=9)
        ], source="ground_truth")

        # Raw model (有錯誤)
        self.raw_model = NoteSequence([
            Note(pitch=61, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=6),  # Pitch 錯誤
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=2, fret=7),
            Note(pitch=64, onset_ticks=960, duration_ticks=480, velocity=80,
                 string=2, fret=9)
        ], source="raw_model")

        # Overlap corrected (修正一些錯誤)
        self.overlap_corrected = NoteSequence([
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5),  # Pitch 修正
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=1, fret=12),  # Pitch 正確但 string 不同
            Note(pitch=64, onset_ticks=960, duration_ticks=480, velocity=80,
                 string=2, fret=9)
        ], source="overlap_corrected")

        # Neighbor refined (完美)
        self.neighbor_refined = NoteSequence([
            Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80,
                 string=2, fret=5),
            Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80,
                 string=2, fret=7),
            Note(pitch=64, onset_ticks=960, duration_ticks=480, velocity=80,
                 string=2, fret=9)
        ], source="neighbor_refined")

    def test_compare_all_methods(self):
        """測試比較三種方法"""
        comparison = PostProcessingEvaluator.compare_methods(
            self.raw_model,
            self.overlap_corrected,
            self.neighbor_refined,
            self.ground_truth,
            self.config
        )

        # 驗證包含三種方法
        self.assertIn('raw_model', comparison)
        self.assertIn('overlap_correction', comparison)
        self.assertIn('neighbor_search', comparison)

        # 驗證 raw model（1/3 pitch 錯誤）
        self.assertAlmostEqual(comparison['raw_model']['pitch_accuracy'], 66.67, places=1)

        # 驗證 overlap correction（pitch 全對，但 tab 有錯）
        self.assertEqual(comparison['overlap_correction']['pitch_accuracy'], 100.0)
        self.assertAlmostEqual(comparison['overlap_correction']['tab_accuracy'], 66.67, places=1)

        # 驗證 neighbor search（完美）
        self.assertEqual(comparison['neighbor_search']['pitch_accuracy'], 100.0)
        self.assertEqual(comparison['neighbor_search']['tab_accuracy'], 100.0)

    def test_comparison_structure(self):
        """測試比較結果的結構"""
        comparison = PostProcessingEvaluator.compare_methods(
            self.raw_model,
            self.overlap_corrected,
            self.neighbor_refined,
            self.ground_truth,
            self.config
        )

        # 檢查每個方法包含必要的欄位
        for method in ['raw_model', 'overlap_correction', 'neighbor_search']:
            self.assertIn('pitch_accuracy', comparison[method])
            self.assertIn('tab_accuracy', comparison[method])
            self.assertIn('total_notes', comparison[method])
            self.assertIn('pitch_correct', comparison[method])
            self.assertIn('tab_correct', comparison[method])

            # 檢查總音符數一致
            self.assertEqual(comparison[method]['total_notes'], 3)


class TestPrintComparisonTable(unittest.TestCase):
    """測試 print_comparison_table 方法"""

    def test_print_table(self):
        """測試表格打印（手動檢查輸出）"""
        comparison = {
            'raw_model': {
                'pitch_accuracy': 97.23,
                'tab_accuracy': 68.56,
                'total_notes': 100,
                'pitch_correct': 97,
                'tab_correct': 68
            },
            'overlap_correction': {
                'pitch_accuracy': 99.92,
                'tab_accuracy': 72.15,
                'total_notes': 100,
                'pitch_correct': 99,
                'tab_correct': 72
            },
            'neighbor_search': {
                'pitch_accuracy': 100.0,
                'tab_accuracy': 72.19,
                'total_notes': 100,
                'pitch_correct': 100,
                'tab_correct': 72
            }
        }

        # 這個測試主要是為了確保不會 crash
        # 實際輸出需要手動檢查
        PostProcessingEvaluator.print_comparison_table(
            comparison,
            title="Test Comparison Table"
        )


class TestAggregateStatistics(unittest.TestCase):
    """測試 calculate_aggregate_statistics 方法"""

    def test_aggregate_statistics(self):
        """測試彙總統計計算"""
        # 建立多個評估結果
        results_list = [
            {
                'raw_model': {'pitch_accuracy': 95.0, 'tab_accuracy': 65.0},
                'overlap_correction': {'pitch_accuracy': 98.0, 'tab_accuracy': 70.0},
                'neighbor_search': {'pitch_accuracy': 100.0, 'tab_accuracy': 72.0}
            },
            {
                'raw_model': {'pitch_accuracy': 97.0, 'tab_accuracy': 68.0},
                'overlap_correction': {'pitch_accuracy': 99.5, 'tab_accuracy': 72.0},
                'neighbor_search': {'pitch_accuracy': 100.0, 'tab_accuracy': 74.0}
            },
            {
                'raw_model': {'pitch_accuracy': 96.0, 'tab_accuracy': 67.0},
                'overlap_correction': {'pitch_accuracy': 99.0, 'tab_accuracy': 71.0},
                'neighbor_search': {'pitch_accuracy': 100.0, 'tab_accuracy': 73.0}
            }
        ]

        aggregate = PostProcessingEvaluator.calculate_aggregate_statistics(results_list)

        # 驗證結構
        self.assertIn('raw_model', aggregate)
        self.assertIn('overlap_correction', aggregate)
        self.assertIn('neighbor_search', aggregate)

        # 驗證 raw_model 平均值
        self.assertAlmostEqual(aggregate['raw_model']['pitch_accuracy_mean'], 96.0, places=1)
        self.assertAlmostEqual(aggregate['raw_model']['tab_accuracy_mean'], 66.67, places=1)

        # 驗證 neighbor_search（全部 100%）
        self.assertEqual(aggregate['neighbor_search']['pitch_accuracy_mean'], 100.0)
        self.assertEqual(aggregate['neighbor_search']['pitch_accuracy_std'], 0.0)

        # 驗證樣本數
        self.assertEqual(aggregate['raw_model']['num_samples'], 3)

    def test_empty_results_list(self):
        """測試空結果列表"""
        aggregate = PostProcessingEvaluator.calculate_aggregate_statistics([])
        self.assertEqual(aggregate, {})


if __name__ == '__main__':
    unittest.main()
