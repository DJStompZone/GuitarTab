"""
Post-Processing Evaluator
==========================

評估 post-processing 效果，複製論文 Table 2 的評估方式。
"""

from typing import Dict, List, Optional, Tuple, Any
from .datatypes import Note
from .sequence import NoteSequence
from .config import GuitarConfig


class PostProcessingEvaluator:
    """
    評估 post-processing 效果的工具類別

    用於比較不同 post-processing 方法的效能，複製論文 Table 2 的結果。
    """

    @staticmethod
    def evaluate_pitch_accuracy(predicted: NoteSequence,
                                ground_truth: NoteSequence,
                                config: GuitarConfig,
                                time_tolerance: int = 120) -> Dict[str, Any]:
        """
        評估 pitch accuracy 和 tablature accuracy

        Args:
            predicted: 預測的音符序列
            ground_truth: Ground truth 音符序列
            config: 吉他配置
            time_tolerance: 時間容差（ticks），用於配對音符

        Returns:
            Dict 包含以下欄位:
                - pitch_accuracy: Pitch 準確度百分比
                - tab_accuracy: Tablature 準確度百分比
                - total_notes: 總音符數
                - pitch_correct: Pitch 正確的音符數
                - tab_correct: Tablature 正確的音符數
                - pitch_errors: Pitch 錯誤列表
                - tab_errors: Tablature 錯誤列表
        """
        total_notes = len(ground_truth)

        if total_notes == 0:
            return {
                'pitch_accuracy': 100.0,
                'tab_accuracy': 100.0,
                'total_notes': 0,
                'pitch_correct': 0,
                'tab_correct': 0,
                'pitch_errors': [],
                'tab_errors': []
            }

        pitch_correct = 0
        tab_correct = 0
        pitch_errors = []
        tab_errors = []

        # 為每個 ground truth 音符找配對
        for gt_note in ground_truth:
            # 在時間容差內找最接近的預測音符
            candidates = [
                pred_note for pred_note in predicted
                if abs(pred_note.onset_ticks - gt_note.onset_ticks) <= time_tolerance
            ]

            if not candidates:
                # 沒有找到配對，記錄為錯誤
                pitch_errors.append({
                    'gt_pitch': gt_note.pitch,
                    'pred_pitch': None,
                    'onset': gt_note.onset_ticks
                })
                tab_errors.append({
                    'gt_string': gt_note.string,
                    'gt_fret': gt_note.fret,
                    'pred_string': None,
                    'pred_fret': None,
                    'onset': gt_note.onset_ticks
                })
                continue

            # 選擇時間最接近的
            pred_note = min(candidates, key=lambda n: abs(n.onset_ticks - gt_note.onset_ticks))

            # 檢查 pitch accuracy
            if pred_note.pitch == gt_note.pitch:
                pitch_correct += 1
            else:
                pitch_errors.append({
                    'gt_pitch': gt_note.pitch,
                    'pred_pitch': pred_note.pitch,
                    'onset': gt_note.onset_ticks
                })

            # 檢查 tablature accuracy
            if (pred_note.has_tablature() and gt_note.has_tablature() and
                pred_note.string == gt_note.string and pred_note.fret == gt_note.fret):
                tab_correct += 1
            else:
                tab_errors.append({
                    'gt_string': gt_note.string,
                    'gt_fret': gt_note.fret,
                    'pred_string': pred_note.string if pred_note.has_tablature() else None,
                    'pred_fret': pred_note.fret if pred_note.has_tablature() else None,
                    'onset': gt_note.onset_ticks
                })

        pitch_accuracy = (pitch_correct / total_notes) * 100.0
        tab_accuracy = (tab_correct / total_notes) * 100.0

        return {
            'pitch_accuracy': pitch_accuracy,
            'tab_accuracy': tab_accuracy,
            'total_notes': total_notes,
            'pitch_correct': pitch_correct,
            'tab_correct': tab_correct,
            'pitch_errors': pitch_errors,
            'tab_errors': tab_errors
        }

    @staticmethod
    def compare_methods(raw_model: NoteSequence,
                       overlap_corrected: NoteSequence,
                       neighbor_refined: NoteSequence,
                       ground_truth: NoteSequence,
                       config: GuitarConfig) -> Dict[str, Dict[str, float]]:
        """
        比較不同 post-processing 方法的效果

        複製論文 Table 2 的評估方式。

        Args:
            raw_model: 原始模型輸出
            overlap_corrected: Overlap correction 後的結果
            neighbor_refined: Neighbor search 後的結果
            ground_truth: Ground truth 序列
            config: 吉他配置

        Returns:
            Dict 包含三種方法的評估結果:
                - 'raw_model': 原始模型的準確度
                - 'overlap_correction': Overlap correction 的準確度
                - 'neighbor_search': Neighbor search 的準確度

            每個方法包含:
                - pitch_accuracy: Pitch 準確度 (%)
                - tab_accuracy: Tablature 準確度 (%)
                - total_notes: 總音符數
                - pitch_correct: Pitch 正確數
                - tab_correct: Tablature 正確數
        """
        evaluator = PostProcessingEvaluator()

        # 評估三種方法
        raw_results = evaluator.evaluate_pitch_accuracy(raw_model, ground_truth, config)
        overlap_results = evaluator.evaluate_pitch_accuracy(overlap_corrected, ground_truth, config)
        neighbor_results = evaluator.evaluate_pitch_accuracy(neighbor_refined, ground_truth, config)

        # 整理結果
        comparison = {
            'raw_model': {
                'pitch_accuracy': raw_results['pitch_accuracy'],
                'tab_accuracy': raw_results['tab_accuracy'],
                'total_notes': raw_results['total_notes'],
                'pitch_correct': raw_results['pitch_correct'],
                'tab_correct': raw_results['tab_correct']
            },
            'overlap_correction': {
                'pitch_accuracy': overlap_results['pitch_accuracy'],
                'tab_accuracy': overlap_results['tab_accuracy'],
                'total_notes': overlap_results['total_notes'],
                'pitch_correct': overlap_results['pitch_correct'],
                'tab_correct': overlap_results['tab_correct']
            },
            'neighbor_search': {
                'pitch_accuracy': neighbor_results['pitch_accuracy'],
                'tab_accuracy': neighbor_results['tab_accuracy'],
                'total_notes': neighbor_results['total_notes'],
                'pitch_correct': neighbor_results['pitch_correct'],
                'tab_correct': neighbor_results['tab_correct']
            }
        }

        return comparison

    @staticmethod
    def print_comparison_table(comparison: Dict[str, Dict[str, float]],
                              title: str = "Post-Processing Comparison"):
        """
        以表格形式打印比較結果（類似論文 Table 2）

        Args:
            comparison: compare_methods() 返回的比較結果
            title: 表格標題
        """
        print(f"\n{title}")
        print("=" * 80)
        print(f"{'Method':<30} {'Pitch Acc (%)':<15} {'Tab Acc (%)':<15} {'Notes':<10}")
        print("-" * 80)

        for method, results in comparison.items():
            method_name = method.replace('_', ' ').title()
            pitch_acc = results['pitch_accuracy']
            tab_acc = results['tab_accuracy']
            total = results['total_notes']

            print(f"{method_name:<30} {pitch_acc:>13.2f} {tab_acc:>13.2f} {total:>8}")

        print("=" * 80)

        # 計算改善
        if 'raw_model' in comparison and 'neighbor_search' in comparison:
            raw_pitch = comparison['raw_model']['pitch_accuracy']
            neighbor_pitch = comparison['neighbor_search']['pitch_accuracy']
            improvement = neighbor_pitch - raw_pitch
            print(f"\nPitch Accuracy Improvement: {improvement:+.2f}%")

            if neighbor_pitch >= 99.9:
                print("✓ Achieved target: ~100% pitch accuracy")

        print()

    @staticmethod
    def calculate_aggregate_statistics(results_list: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        計算多個評估結果的彙總統計

        用於在多個檔案上評估 post-processing 效果。

        Args:
            results_list: compare_methods() 返回的結果列表

        Returns:
            彙總統計，包含平均值和標準差
        """
        if not results_list:
            return {}

        import statistics

        methods = results_list[0].keys()
        aggregate = {}

        for method in methods:
            pitch_accs = [r[method]['pitch_accuracy'] for r in results_list]
            tab_accs = [r[method]['tab_accuracy'] for r in results_list]

            aggregate[method] = {
                'pitch_accuracy_mean': statistics.mean(pitch_accs),
                'pitch_accuracy_std': statistics.stdev(pitch_accs) if len(pitch_accs) > 1 else 0.0,
                'tab_accuracy_mean': statistics.mean(tab_accs),
                'tab_accuracy_std': statistics.stdev(tab_accs) if len(tab_accs) > 1 else 0.0,
                'num_samples': len(results_list)
            }

        return aggregate
