"""
Fretting Post-Processor API
============================

主要 API 介面，提供簡單易用的高階函數來執行 post-processing。
"""

from typing import List, Dict, Optional, Any
from .datatypes import Note
from .sequence import NoteSequence
from .config import GuitarConfig
from .parser import TokenParser
from .serializer import TokenSerializer
from .validator import PitchValidator
from .processor import PostProcessor
from .evaluator import PostProcessingEvaluator


class FrettingPostProcessor:
    """
    Fretting-Transformer Post-Processor 主 API

    提供高階介面來執行 post-processing，包括:
    - Overlap correction (Section 3.5)
    - Neighbor search (Section 4.2)
    - 評估 pitch accuracy

    使用範例:
    ```python
    from fretting_postprocessor import FrettingPostProcessor, GuitarConfig

    # 初始化
    processor = FrettingPostProcessor()

    # 處理 tokens
    corrected_tokens = processor.process_tokens(
        model_output_tokens,
        input_note_tokens,
        method='neighbor_search'
    )

    # 評估效果
    results = processor.evaluate(
        model_output_tokens,
        input_note_tokens
    )
    ```
    """

    def __init__(self, guitar_config: Optional[GuitarConfig] = None):
        """
        初始化 Post-Processor

        Args:
            guitar_config: 吉他配置，如果為 None 則使用預設配置（標準 tuning）
        """
        if guitar_config is None:
            guitar_config = GuitarConfig()

        self.config = guitar_config
        self.parser = TokenParser()
        self.serializer = TokenSerializer()
        self.validator = PitchValidator()
        self.processor = PostProcessor(guitar_config)
        self.evaluator = PostProcessingEvaluator()

    def process_tokens(self,
                      model_output_tokens: List[str],
                      input_note_tokens: List[str],
                      method: str = 'neighbor_search',
                      output_format: str = 'auto') -> List[str]:
        """
        處理 token 序列並返回修正後的輸出

        這是主要的處理函數，執行完整的 post-processing pipeline。

        Args:
            model_output_tokens: 模型輸出 tokens (TAB, NOTE_ON/OFF, 或 MIXED format)
            input_note_tokens: 輸入的 NOTE_ON/OFF tokens (ground truth pitches)
            method: Post-processing 方法:
                - 'overlap': 只執行 overlap correction (~99.92% pitch accuracy)
                - 'neighbor_search': 執行完整 pipeline (~100% pitch accuracy)
            output_format: 輸出格式:
                - 'auto': 自動檢測 (與輸入格式相同)
                - 'tab': 強制 TAB format
                - 'note_on_off': 強制 NOTE_ON/OFF format
                - 'mixed': 強制 MIXED format (NOTE_ON/OFF + TAB)

        Returns:
            修正後的 tokens (指定格式)

        Raises:
            ValueError: 如果 method 或 output_format 無效

        Note:
            支持三種模型輸出格式:
            - TAB format: TAB<string,fret> TIME_SHIFT<ticks>
            - NOTE_ON/OFF format: NOTE_ON<pitch> NOTE_OFF<pitch> TIME_SHIFT<ticks>
            - MIXED format: NOTE_ON<pitch> TAB<string,fret> NOTE_OFF<pitch> TIME_SHIFT<ticks>

            MIXED format 是訓練資料的標準格式，每個音符同時包含 pitch 和 tablature。

            對於 NOTE_ON/OFF-only 輸出，會自動推導 tablature 以便應用
            neighbor_search，然後將優化結果轉回 NOTE_ON/OFF。
        """
        # Step 1: Parse input sequence
        input_sequence = self.parser.parse_input_tokens(input_note_tokens)

        # Step 2: Detect model output format and parse
        model_format = self._detect_output_format(model_output_tokens)

        if model_format == 'TAB':
            # TAB-only format
            model_output = self.parser.parse_output_tokens(
                model_output_tokens,
                input_sequence,
                self.config
            )
        elif model_format == 'NOTE_ON_OFF':
            # NOTE_ON/OFF-only format - infer tablature
            model_output = self.parser.parse_note_on_off_output(
                model_output_tokens,
                input_sequence,
                self.config
            )
        elif model_format == 'MIXED':
            # MIXED format (NOTE_ON/OFF + TAB) - parse both token types
            model_output = self.parser.parse_mixed_format_output(
                model_output_tokens,
                input_sequence,
                self.config
            )
        else:
            raise ValueError(f"Unknown model output format: {model_format}")

        # Step 3: Apply post-processing (保持不變)
        if method == 'overlap':
            # 只執行 overlap correction
            corrected_sequence = self.processor.overlap_correction(
                model_output,
                input_sequence
            )
            result_sequence = corrected_sequence

        elif method == 'neighbor_search':
            # 執行完整 pipeline: overlap correction + neighbor search
            corrected_sequence = self.processor.overlap_correction(
                model_output,
                input_sequence
            )
            result_sequence = self.processor.neighbor_search(
                corrected_sequence,
                optimize_for='balanced'
            )

            # 為 notes 附加 config (用於序列化時重新計算 pitch)
            for note in result_sequence:
                note._guitar_config = self.config

        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Must be 'overlap' or 'neighbor_search'."
            )

        # Step 4: Serialize to requested format
        if output_format == 'auto':
            output_format = model_format.lower()

        if output_format == 'tab':
            output_tokens = self.serializer.serialize_to_output_format(result_sequence)
        elif output_format == 'note_on_off':
            output_tokens = self.serializer.serialize_to_note_on_off(result_sequence)
        elif output_format == 'mixed':
            output_tokens = self.serializer.serialize_to_mixed_format(result_sequence)
        else:
            raise ValueError(
                f"Unknown output format: {output_format}. "
                f"Must be 'auto', 'tab', 'note_on_off', or 'mixed'."
            )

        return output_tokens

    def _detect_output_format(self, tokens: List[str]) -> str:
        """
        自動檢測 token 格式

        Args:
            tokens: Token list to analyze

        Returns:
            'TAB', 'NOTE_ON_OFF', or 'MIXED'

        Note:
            檢查前 100 個非特殊 tokens。
            - MIXED: 同時包含 TAB 和 NOTE_ON tokens (訓練資料格式)
            - TAB: 只有 TAB tokens
            - NOTE_ON_OFF: 只有 NOTE_ON/OFF tokens
        """
        # 檢查前 100 個 tokens (排除 PAD/BOS/EOS)
        sample_tokens = [
            t for t in tokens[:100]
            if t not in ['PAD', 'BOS', 'EOS']
        ]

        tab_count = sum(1 for t in sample_tokens if t.startswith('TAB<'))
        note_count = sum(1 for t in sample_tokens if t.startswith('NOTE_ON<'))

        # 判斷格式
        if tab_count > 0 and note_count > 0:
            # 混合格式：同時有 TAB 和 NOTE_ON
            return 'MIXED'
        elif tab_count > 0:
            # 只有 TAB
            return 'TAB'
        elif note_count > 0:
            # 只有 NOTE_ON/OFF
            return 'NOTE_ON_OFF'
        else:
            # 預設為 NOTE_ON_OFF (向後相容)
            return 'NOTE_ON_OFF'

    def evaluate(self,
                model_output_tokens: List[str],
                input_note_tokens: List[str],
                ground_truth_tab_tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        評估 post-processing 效果（複製論文 Table 2）

        比較三種方法的效能:
        1. Raw model output (no post-processing)
        2. Overlap correction
        3. Neighbor search (full pipeline)

        Args:
            model_output_tokens: 模型輸出的 TAB tokens
            input_note_tokens: 輸入的 NOTE_ON/OFF tokens (ground truth pitches)
            ground_truth_tab_tokens: Ground truth 的 TAB tokens (可選)
                如果提供，會用於評估 tab accuracy
                如果不提供，會使用 input_sequence 推斷的 tablature

        Returns:
            Dict 包含三種方法的評估結果:
                {
                    'raw_model': {
                        'pitch_accuracy': float,
                        'tab_accuracy': float,
                        'total_notes': int,
                        ...
                    },
                    'overlap_correction': {...},
                    'neighbor_search': {...}
                }
        """
        # Step 1: Parse inputs
        input_sequence = self.parser.parse_input_tokens(input_note_tokens)
        model_output = self.parser.parse_output_tokens(
            model_output_tokens,
            input_sequence,
            self.config
        )

        # Parse ground truth if provided
        if ground_truth_tab_tokens is not None:
            ground_truth = self.parser.parse_output_tokens(
                ground_truth_tab_tokens,
                input_sequence,
                self.config
            )
        else:
            # 使用 input_sequence 作為 ground truth
            # (假設 input pitches 是正確的，只評估 pitch accuracy)
            ground_truth = input_sequence

        # Step 2: Apply both post-processing methods
        overlap_corrected = self.processor.overlap_correction(
            model_output,
            input_sequence
        )

        neighbor_refined = self.processor.neighbor_search(
            overlap_corrected,
            optimize_for='balanced'
        )

        # Step 3: Evaluate all three methods
        comparison = self.evaluator.compare_methods(
            model_output,
            overlap_corrected,
            neighbor_refined,
            ground_truth,
            self.config
        )

        return comparison

    def process_and_evaluate(self,
                            model_output_tokens: List[str],
                            input_note_tokens: List[str],
                            method: str = 'neighbor_search',
                            verbose: bool = True) -> tuple[List[str], Dict[str, Any]]:
        """
        處理並評估（一次完成兩個操作）

        這是一個方便的函數，同時執行 process_tokens() 和 evaluate()。

        Args:
            model_output_tokens: 模型輸出的 TAB tokens
            input_note_tokens: 輸入的 NOTE_ON/OFF tokens
            method: Post-processing 方法
            verbose: 是否打印評估結果

        Returns:
            Tuple of (corrected_tokens, evaluation_results)
        """
        # Process
        corrected_tokens = self.process_tokens(
            model_output_tokens,
            input_note_tokens,
            method=method
        )

        # Evaluate
        evaluation = self.evaluate(
            model_output_tokens,
            input_note_tokens
        )

        # Print results if verbose
        if verbose:
            self.evaluator.print_comparison_table(
                evaluation,
                title=f"Post-Processing Results ({method})"
            )

        return corrected_tokens, evaluation

    def get_note_sequences(self,
                          model_output_tokens: List[str],
                          input_note_tokens: List[str],
                          method: str = 'neighbor_search') -> Dict[str, NoteSequence]:
        """
        取得所有中間步驟的 NoteSequence（用於分析和除錯）

        Args:
            model_output_tokens: 模型輸出的 TAB tokens
            input_note_tokens: 輸入的 NOTE_ON/OFF tokens
            method: Post-processing 方法

        Returns:
            Dict 包含:
                - 'input': 輸入序列
                - 'model_output': 原始模型輸出
                - 'overlap_corrected': Overlap correction 後
                - 'neighbor_refined': Neighbor search 後（如果 method='neighbor_search'）
        """
        # Parse
        input_sequence = self.parser.parse_input_tokens(input_note_tokens)
        model_output = self.parser.parse_output_tokens(
            model_output_tokens,
            input_sequence,
            self.config
        )

        # Apply post-processing
        overlap_corrected = self.processor.overlap_correction(
            model_output,
            input_sequence
        )

        sequences = {
            'input': input_sequence,
            'model_output': model_output,
            'overlap_corrected': overlap_corrected
        }

        if method == 'neighbor_search':
            neighbor_refined = self.processor.neighbor_search(
                overlap_corrected,
                optimize_for='balanced'
            )
            sequences['neighbor_refined'] = neighbor_refined

        return sequences

    def change_guitar_config(self, new_config: GuitarConfig):
        """
        更改吉他配置

        這會重新初始化內部的 PostProcessor。

        Args:
            new_config: 新的吉他配置
        """
        self.config = new_config
        self.processor = PostProcessor(new_config)

    def get_config(self) -> GuitarConfig:
        """
        取得當前的吉他配置

        Returns:
            GuitarConfig: 當前配置
        """
        return self.config


# Convenience functions for quick usage
def process_tokens_quick(model_output_tokens: List[str],
                        input_note_tokens: List[str],
                        tuning: tuple = None) -> List[str]:
    """
    快速處理 tokens（使用預設配置）

    這是一個方便的函數，不需要創建 FrettingPostProcessor 物件。

    Args:
        model_output_tokens: 模型輸出的 TAB tokens
        input_note_tokens: 輸入的 NOTE_ON/OFF tokens
        tuning: 吉他 tuning (可選)，預設為標準 tuning

    Returns:
        修正後的 TAB tokens
    """
    if tuning is not None:
        config = GuitarConfig(tuning=tuning)
    else:
        config = GuitarConfig()

    processor = FrettingPostProcessor(config)
    return processor.process_tokens(
        model_output_tokens,
        input_note_tokens,
        method='neighbor_search'
    )


def evaluate_quick(model_output_tokens: List[str],
                  input_note_tokens: List[str],
                  tuning: tuple = None,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    快速評估 post-processing 效果（使用預設配置）

    Args:
        model_output_tokens: 模型輸出的 TAB tokens
        input_note_tokens: 輸入的 NOTE_ON/OFF tokens
        tuning: 吉他 tuning (可選)
        verbose: 是否打印結果表格

    Returns:
        評估結果 dict
    """
    if tuning is not None:
        config = GuitarConfig(tuning=tuning)
    else:
        config = GuitarConfig()

    processor = FrettingPostProcessor(config)
    results = processor.evaluate(model_output_tokens, input_note_tokens)

    if verbose:
        PostProcessingEvaluator.print_comparison_table(results)

    return results
