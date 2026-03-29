# Fretting-Transformer Post-Processing 實作計畫書

> **專案目標**：根據 Fretting-Transformer 論文（Section 3.5 和 4.2）實作 post-processing 模組，將模型生成的 guitar tablature 預測結果從 ~97% pitch accuracy 提升至 100%。

**文件版本**：v1.0
**建立日期**：2025-12-04
**論文參考**：Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription (arXiv:2506.14223v1)

---

## 目錄

1. [專案概述](#1-專案概述)
2. [Input/Output 格式定義](#2-inputoutput-格式定義)
3. [Post-Processing 演算法](#3-post-processing-演算法)
4. [程式碼架構設計](#4-程式碼架構設計)
5. [實作步驟](#5-實作步驟)
6. [Edge Cases 處理](#6-edge-cases-處理)
7. [與現有 Pipeline 整合](#7-與現有-pipeline-整合)
8. [使用範例](#8-使用範例)
9. [預期成果](#9-預期成果)
10. [時間規劃](#10-時間規劃)

---

## 1. 專案概述

### 1.1 背景

Fretting-Transformer 是一個基於 T5 transformer 架構的模型，用於將 MIDI 序列轉換為 guitar tablature。然而，模型輸出的 tablature 可能包含音高錯誤（pitch errors），需要透過 post-processing 來修正。

### 1.2 Post-Processing 的作用

根據論文 Section 3.5：
> "In some cases, the model can generate tabs for a note that results in an incorrect pitch. To address this, errors are corrected in a post-processing step to ensure that the piece of music remains unchanged."

### 1.3 實作目標

實現兩種 post-processing 方法（基於論文 Table 2）：

| Method | Pitch Accuracy | Description |
|--------|----------------|-------------|
| 無 Post-Processing | 97.23% | 模型原始輸出 |
| **Overlap Correction** | 99.92% | 配對輸入音符，修正 pitch errors |
| **Neighbor Search** | **100.00%** | 優化 string-fret 選擇 |

### 1.4 技術規格

- **程式語言**：Python 3.9+
- **Token 格式**：v3 encoding (論文 Table 1)
- **整合方式**：Python module/library
- **支援功能**：Multiple tunings + Capo positions

---

## 2. Input/Output 格式定義

### 2.1 Token 格式 (v3 Encoding)

#### Input Tokens：原始 MIDI 音符序列

**格式**：
```
NOTE_ON<pitch> TIME_SHIFT<ticks> NOTE_OFF<pitch>
```

**範例**：
```
NOTE_ON<55> TIME_SHIFT<120> NOTE_OFF<55> NOTE_ON<57> TIME_SHIFT<240> NOTE_OFF<57>
```

**說明**：
- `NOTE_ON<pitch>`：音符開始，pitch 為 MIDI note number (0-127)
- `TIME_SHIFT<ticks>`：時間推進，單位為 ticks
- `NOTE_OFF<pitch>`：音符結束
- Duration = (NOTE_OFF time) - (NOTE_ON time)

#### Output Tokens：Tablature 序列

**格式**：
```
TAB<string,fret> TIME_SHIFT<ticks>
```

**範例**：
```
TAB<3,0> TIME_SHIFT<120> TAB<3,2> TIME_SHIFT<240>
```

**說明**：
- `TAB<string,fret>`：tablature 位置
  - `string`：弦編號 (0-5 for 6-string guitar)
  - `fret`：品格編號 (0-24)
- `TIME_SHIFT<ticks>`：時間推進

#### Pitch 計算公式

```python
pitch = tuning[string] + fret
```

**範例**（Standard tuning）：
- String 0 (E2): tuning[0] = 40
- Fret 5: fret = 5
- **Pitch = 40 + 5 = 45 (A2)**

### 2.2 內部資料結構

#### Note 類別

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Note:
    """音符的完整表示，整合 pitch 和 tablature 資訊"""

    # 音樂屬性
    pitch: int              # MIDI pitch (0-127)
    onset_ticks: int        # 絕對時間位置 (ticks)
    duration_ticks: int     # 音符長度 (ticks)
    velocity: int           # MIDI velocity (0-127)

    # Tablature 屬性
    string: Optional[int] = None    # 弦編號 (0-5)
    fret: Optional[int] = None      # 品格編號 (0-24)

    # 元資料
    matched: bool = False           # 是否已配對（用於演算法）
    source: str = "input"          # "input", "model", "corrected", "refined"

    def has_tablature(self) -> bool:
        """檢查是否有 tablature 資訊"""
        return self.string is not None and self.fret is not None

    def get_pitch_from_tablature(self, tuning: tuple) -> Optional[int]:
        """從 tablature 計算 pitch"""
        if not self.has_tablature():
            return None
        return tuning[self.string] + self.fret

    def get_offset_ticks(self) -> int:
        """取得音符結束時間"""
        return self.onset_ticks + self.duration_ticks
```

#### GuitarConfig 類別

```python
@dataclass
class GuitarConfig:
    """吉他配置參數"""

    # 預設值：標準 6 弦吉他
    num_strings: int = 6
    tuning: tuple = (40, 45, 50, 55, 59, 64)  # E2, A2, D3, G3, B3, E4
    capo_fret: int = 0                        # Capo 位置
    min_fret: int = 0
    max_fret: int = 24

    def get_effective_tuning(self) -> tuple:
        """取得考慮 capo 的實際 tuning"""
        return tuple(pitch + self.capo_fret for pitch in self.tuning)

    def pitch_to_string_fret(self, pitch: int) -> list:
        """
        找出所有可產生該 pitch 的 (string, fret) 組合

        Returns:
            List[Tuple[int, int]]: [(string_idx, fret_num), ...]
        """
        effective_tuning = self.get_effective_tuning()
        valid_positions = []

        for string_idx, open_pitch in enumerate(effective_tuning):
            fret = pitch - open_pitch
            if self.min_fret <= fret <= self.max_fret:
                valid_positions.append((string_idx, fret))

        return valid_positions
```

#### 常見 Tunings 預設值

```python
# Standard tuning (E standard)
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # E2, A2, D3, G3, B3, E4

# Drop-D tuning
DROP_D_TUNING = (38, 45, 50, 55, 59, 64)    # D2, A2, D3, G3, B3, E4

# Half-step down
HALF_STEP_DOWN = (39, 44, 49, 54, 58, 63)   # Eb2, Ab2, Db3, Gb3, Bb3, Eb4

# Full-step down
FULL_STEP_DOWN = (38, 43, 48, 53, 57, 62)   # D2, G2, C3, F3, A3, D4
```

---

## 3. Post-Processing 演算法

### 3.1 Overlap Correction Algorithm

#### 概念說明

**目標**：比對模型輸出與輸入音符序列，修正 pitch errors

**核心思想**（論文 Section 3.5）：
1. 在 ±5 個音符的 window 內尋找最匹配的輸入音符
2. 評估 pitch 相似度和時間相近度
3. 使用輸入音符的 pitch（ground truth）替換模型預測
4. 如果沒有找到配對，使用 fallback strategy

#### 演算法流程圖

```
┌─────────────────────────────────────────────┐
│ Input: model_output, input_sequence         │
│ window_size = 5                             │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ FOR EACH model_note IN model_output:       │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 1. Calculate predicted_pitch from          │
│    tablature: pitch = fret + tuning[string]│
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 2. Get candidate notes in ±5 window        │
│    candidates = input_sequence.get_notes_  │
│                 in_window(onset, 5)         │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 3. Find best match:                        │
│    FOR EACH candidate:                      │
│      score = (pitch_diff * 1000) +          │
│              (time_diff * 10) +             │
│              duration_diff                  │
│    best_match = argmin(score)               │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │ Match   │      │ No Match │
   │ Found   │      │          │
   └────┬────┘      └────┬─────┘
        │                │
        ▼                ▼
┌──────────────┐  ┌──────────────┐
│ 4a. Use      │  │ 4b. Fallback:│
│ input pitch  │  │ Use first    │
│ (ground      │  │ valid string-│
│ truth)       │  │ fret combo   │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────────────────────┐
│ 5. Validate tablature:                      │
│    IF calculated_pitch != target_pitch:     │
│      correct_note_tablature()               │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Add corrected note to output                │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ END FOR                                     │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Return corrected_sequence                   │
└─────────────────────────────────────────────┘
```

#### 詳細步驟（Pseudocode）

```python
def overlap_correction(model_output, input_sequence, window_size=5):
    """
    Overlap correction algorithm from paper Section 3.5

    Args:
        model_output: NoteSequence - 模型預測（含 tablature）
        input_sequence: NoteSequence - 輸入音符（ground truth pitches）
        window_size: int - 搜尋視窗大小（預設 ±5）

    Returns:
        NoteSequence - 修正後的音符序列
    """
    corrected_notes = []

    for model_note in model_output:
        # Step 1: Calculate predicted pitch from tablature
        if model_note.has_tablature():
            predicted_pitch = model_note.get_pitch_from_tablature(config.tuning)
        else:
            predicted_pitch = model_note.pitch

        # Step 2: Get candidate notes in ±window_size window
        candidates = input_sequence.get_notes_in_window(
            model_note.onset_ticks,
            window_size
        )

        # Step 3: Find best matching note
        best_match = None
        min_score = float('inf')

        for candidate in candidates:
            if candidate.matched:
                continue  # Skip already matched notes

            # Calculate matching score
            pitch_diff = abs(candidate.pitch - predicted_pitch)
            time_diff = abs(candidate.onset_ticks - model_note.onset_ticks)
            duration_diff = abs(candidate.duration_ticks - model_note.duration_ticks)

            # Weighted score (pitch is most important)
            score = (pitch_diff * 1000) + (time_diff * 10) + duration_diff

            if score < min_score:
                min_score = score
                best_match = candidate

        # Step 4: Create corrected note
        if best_match is not None:
            # 4a. Use input pitch (ground truth)
            corrected_note = Note(
                pitch=best_match.pitch,  # Ground truth pitch
                onset_ticks=model_note.onset_ticks,
                duration_ticks=model_note.duration_ticks,
                velocity=best_match.velocity,
                source="corrected"
            )

            # Try to preserve model's string choice if valid
            if model_note.has_tablature():
                corrected_note.string = model_note.string
                corrected_note.fret = best_match.pitch - config.tuning[model_note.string]

                # Step 5: Validate tablature
                if not validator.validate_note(corrected_note):
                    # Fallback: use first valid position
                    validator.correct_note_tablature(corrected_note)
            else:
                # No tablature in model output
                validator.correct_note_tablature(corrected_note)

            best_match.matched = True  # Mark as matched
            corrected_notes.append(corrected_note)
        else:
            # 4b. No match found - use fallback
            fallback_note = create_fallback_note(model_note)
            corrected_notes.append(fallback_note)

    return NoteSequence(corrected_notes, source="corrected")
```

#### 關鍵參數說明

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `window_size` | 5 | 搜尋視窗大小（前後各 N 個音符） |
| `pitch_weight` | 1000 | Pitch 差異的權重（最重要） |
| `time_weight` | 10 | 時間差異的權重 |
| `duration_weight` | 1 | Duration 差異的權重（最不重要） |

#### 預期效果

- **Pitch Accuracy**: 97.23% → **99.92%**
- **主要改善**：修正模型預測錯誤的 pitch
- **保留優點**：維持模型預測的 timing 和 string 選擇（如果有效）

---

### 3.2 Neighbor Search Algorithm

#### 概念說明

**目標**：在確保 pitch 正確的前提下，優化 string-fret 選擇以提升 playability

**核心思想**（論文 Section 4.2）：
1. 對於每個音符，探索所有能產生相同 pitch 的 (string, fret) 組合
2. 根據 context（前後音符）評估每個選擇的分數
3. 選擇最佳位置，考慮：
   - String consistency（弦一致性）
   - Position proximity（位置接近度）
   - Playability（可演奏性）
   - Avoid extremes（避免極端位置）

#### 演算法流程圖

```
┌─────────────────────────────────────────────┐
│ Input: corrected_sequence                   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ FOR EACH note IN corrected_sequence:        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 1. Get all alternative positions for pitch │
│    alternatives = pitch_to_string_fret()    │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   ┌─────────┐      ┌──────────────┐
   │ Only 1  │      │ Multiple     │
   │ position│      │ alternatives │
   └────┬────┘      └──────┬───────┘
        │                  │
        │                  ▼
        │         ┌─────────────────────┐
        │         │ 2. Get context:     │
        │         │ prev_notes = [-3:0] │
        │         │ next_notes = [1:4]  │
        │         └──────┬──────────────┘
        │                │
        │                ▼
        │         ┌─────────────────────────────────┐
        │         │ 3. Score each alternative:      │
        │         │ FOR (string, fret) IN alts:     │
        │         │   score = 0                     │
        │         │   # String consistency          │
        │         │   IF prev uses same string:     │
        │         │     score -= 20                 │
        │         │   # Position proximity          │
        │         │   IF prev on same string:       │
        │         │     score += |fret_diff| * 5    │
        │         │   # Playability                 │
        │         │   score += fret * 0.5           │
        │         │   # Avoid extremes              │
        │         │   IF fret > 15:                 │
        │         │     score += 10                 │
        │         └──────┬──────────────────────────┘
        │                │
        │                ▼
        │         ┌─────────────────────┐
        │         │ 4. Select best:     │
        │         │ best = argmin(score)│
        │         └──────┬──────────────┘
        │                │
        └────────┬───────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Create refined note with optimal position  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Add to output                               │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ END FOR                                     │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Return refined_sequence                     │
└─────────────────────────────────────────────┘
```

#### 詳細步驟（Pseudocode）

```python
def neighbor_search(corrected_sequence, optimize_for="playability"):
    """
    Neighbor search refinement from paper Section 4.2

    Args:
        corrected_sequence: NoteSequence - Overlap correction 後的序列
        optimize_for: str - 優化目標 ("playability" or "position_stability")

    Returns:
        NoteSequence - 優化後的音符序列
    """
    refined_notes = []

    for i, note in enumerate(corrected_sequence):
        if not note.has_tablature():
            refined_notes.append(note)
            continue

        # Step 1: Get all alternative positions for this pitch
        alternatives = guitar_config.pitch_to_string_fret(note.pitch)

        if len(alternatives) <= 1:
            # Only one valid position, keep it
            refined_notes.append(note)
            continue

        # Step 2: Get context (previous and following notes)
        context_window = 3
        prev_notes = list(corrected_sequence)[max(0, i-context_window):i]
        next_notes = list(corrected_sequence)[i+1:min(len(corrected_sequence), i+context_window+1)]

        # Step 3: Score each alternative position
        best_position = None
        min_score = float('inf')

        for string, fret in alternatives:
            score = 0

            # Factor 1: String consistency (prefer same string as neighbors)
            for prev in prev_notes[-2:]:  # Look at last 2 notes
                if prev.has_tablature() and prev.string == string:
                    score -= 20  # Reward for string consistency

            # Factor 2: Position proximity (minimize hand movement)
            if prev_notes and prev_notes[-1].has_tablature():
                prev = prev_notes[-1]
                if prev.string == string:
                    # Same string - penalize large fret jumps
                    fret_distance = abs(fret - prev.fret)
                    score += fret_distance * 5

            # Factor 3: Playability (prefer lower frets for easier playing)
            if optimize_for == "playability":
                score += fret * 0.5  # Slight preference for lower frets

            # Factor 4: Avoid extreme positions
            if fret > 15:
                score += 10  # Penalize high frets

            if score < min_score:
                min_score = score
                best_position = (string, fret)

        # Step 4: Create refined note with optimal position
        refined_note = Note(
            pitch=note.pitch,
            onset_ticks=note.onset_ticks,
            duration_ticks=note.duration_ticks,
            velocity=note.velocity,
            string=best_position[0],
            fret=best_position[1],
            source="refined"
        )
        refined_notes.append(refined_note)

    return NoteSequence(refined_notes, source="refined")
```

#### 評分因子說明

| Factor | Weight | 說明 | 目的 |
|--------|--------|------|------|
| **String Consistency** | -20 | 與前 2 個音符使用相同弦 | 減少弦切換，提升流暢度 |
| **Position Proximity** | +5 per fret | 同弦上的品格距離 | 最小化手部移動 |
| **Playability** | +0.5 per fret | 品格位置（越高越難） | 偏好低把位（easier） |
| **Avoid Extremes** | +10 | 品格 > 15 的懲罰 | 避免不常用的高把位 |

#### 優化目標

1. **`playability`** (預設)：
   - 偏好低品格
   - 適合初學者或簡單樂曲
   - 最小化技術難度

2. **`position_stability`**：
   - 重視位置連續性
   - 適合進階演奏
   - 減少手部移動

#### 預期效果

- **Pitch Accuracy**: 99.92% → **100.00%**
- **主要改善**：
  - 修正剩餘的 pitch errors
  - 優化 string-fret 選擇
  - 提升 playability

---

### 3.3 完整 Pipeline

#### 流程圖

```
┌─────────────────────────────────────────────┐
│ Input:                                      │
│ - model_output_tokens (TAB format)          │
│ - input_note_tokens (NOTE_ON/OFF format)    │
│ - guitar_config (tuning, capo)              │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Step 1: Parse Tokens                        │
│ - TokenParser.parse_input_tokens()          │
│ - TokenParser.parse_output_tokens()         │
│ → input_sequence, model_output (NoteSeq)    │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Step 2: Overlap Correction                  │
│ - PostProcessor.overlap_correction()        │
│ → corrected_sequence                        │
│ Pitch Accuracy: ~97% → ~99.92%              │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Step 3: Neighbor Search (Optional)          │
│ - PostProcessor.neighbor_search()           │
│ → refined_sequence                          │
│ Pitch Accuracy: ~99.92% → 100%              │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Step 4: Serialize Tokens                    │
│ - TokenSerializer.serialize_to_output()     │
│ → corrected_tab_tokens                      │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Output: Corrected TAB tokens                │
└─────────────────────────────────────────────┘
```

#### 程式碼範例

```python
def post_process_pipeline(model_output_tokens, input_note_tokens, guitar_config):
    """完整的 post-processing pipeline"""

    # Step 1: Parse tokens
    parser = TokenParser()
    input_sequence = parser.parse_input_tokens(input_note_tokens)
    model_output = parser.parse_output_tokens(
        model_output_tokens,
        input_sequence,
        guitar_config
    )

    # Step 2: Overlap correction
    post_processor = PostProcessor(guitar_config)
    corrected_sequence = post_processor.overlap_correction(
        model_output,
        input_sequence
    )

    # Step 3: Neighbor search
    refined_sequence = post_processor.neighbor_search(
        corrected_sequence,
        optimize_for="playability"
    )

    # Step 4: Serialize back to tokens
    serializer = TokenSerializer()
    corrected_tokens = serializer.serialize_to_output_format(refined_sequence)

    return corrected_tokens
```

---

## 4. 程式碼架構設計

### 4.1 模組組織

```
fretting_postprocessor/
│
├── __init__.py                 # 套件初始化，匯出主要類別
│   └── from .api import FrettingPostProcessor
│       from .config import GuitarConfig, STANDARD_TUNING, ...
│
├── config.py                   # 吉他配置與預設值
│   ├── class GuitarConfig
│   ├── STANDARD_TUNING
│   ├── DROP_D_TUNING
│   ├── HALF_STEP_DOWN
│   └── FULL_STEP_DOWN
│
├── datatypes.py                # 基礎資料類型
│   ├── enum TokenType
│   ├── class Token
│   └── class Note
│
├── sequence.py                 # 音符序列容器
│   └── class NoteSequence
│       ├── get_notes_at_time()
│       ├── get_notes_in_window()
│       ├── get_notes_in_time_range()
│       └── find_closest_note()
│
├── parser.py                   # Token 解析器
│   └── class TokenParser
│       ├── parse_input_tokens()
│       └── parse_output_tokens()
│
├── validator.py                # Pitch 驗證器
│   └── class PitchValidator
│       ├── validate_note()
│       ├── correct_note_tablature()
│       └── get_alternative_positions()
│
├── processor.py                # 核心 Post-Processing 演算法 ⭐
│   └── class PostProcessor
│       ├── overlap_correction()
│       ├── neighbor_search()
│       ├── process()
│       ├── _find_best_match()
│       ├── _create_fallback_note()
│       ├── _get_context_notes()
│       └── _select_best_position()
│
├── serializer.py               # Token 序列化器
│   └── class TokenSerializer
│       ├── serialize_to_input_format()
│       └── serialize_to_output_format()
│
├── evaluator.py                # 評估工具
│   └── class PostProcessingEvaluator
│       ├── evaluate_pitch_accuracy()
│       └── compare_methods()
│
├── api.py                      # 主要 API 介面 ⭐
│   └── class FrettingPostProcessor
│       ├── __init__()
│       ├── process_tokens()
│       └── evaluate()
│
├── utils.py                    # 輔助函數
│   ├── jams_to_tokens()
│   ├── tokens_to_midi()
│   └── create_guitar_config_from_jams()
│
└── tests/                      # 測試目錄
    ├── test_parser.py
    ├── test_processor.py
    ├── test_neighbor_search.py
    ├── test_integration.py
    └── test_evaluator.py
```

### 4.2 核心類別詳細設計

#### 4.2.1 GuitarConfig (`config.py`)

```python
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class GuitarConfig:
    """吉他配置類別"""

    num_strings: int = 6
    tuning: Tuple[int, ...] = (40, 45, 50, 55, 59, 64)
    capo_fret: int = 0
    min_fret: int = 0
    max_fret: int = 24

    def get_effective_tuning(self) -> Tuple[int, ...]:
        """取得考慮 capo 的實際 tuning"""
        return tuple(pitch + self.capo_fret for pitch in self.tuning)

    def is_valid_string(self, string: int) -> bool:
        """檢查弦編號是否有效"""
        return 0 <= string < self.num_strings

    def is_valid_fret(self, fret: int) -> bool:
        """檢查品格編號是否有效"""
        return self.min_fret <= fret <= self.max_fret

    def pitch_to_string_fret(self, pitch: int) -> List[Tuple[int, int]]:
        """找出所有可產生該 pitch 的 (string, fret) 組合"""
        effective_tuning = self.get_effective_tuning()
        valid_positions = []

        for string_idx, open_pitch in enumerate(effective_tuning):
            fret = pitch - open_pitch
            if self.is_valid_fret(fret):
                valid_positions.append((string_idx, fret))

        return valid_positions

    def get_pitch_range(self) -> Tuple[int, int]:
        """取得可演奏的 pitch 範圍"""
        effective_tuning = self.get_effective_tuning()
        min_pitch = min(effective_tuning) + self.min_fret
        max_pitch = max(effective_tuning) + self.max_fret
        return (min_pitch, max_pitch)

# 預設 tunings
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # E2, A2, D3, G3, B3, E4
DROP_D_TUNING = (38, 45, 50, 55, 59, 64)    # D2, A2, D3, G3, B3, E4
HALF_STEP_DOWN = (39, 44, 49, 54, 58, 63)   # Eb2, Ab2, Db3, Gb3, Bb3, Eb4
FULL_STEP_DOWN = (38, 43, 48, 53, 57, 62)   # D2, G2, C3, F3, A3, D4
```

#### 4.2.2 NoteSequence (`sequence.py`)

```python
from typing import List, Dict, Optional

class NoteSequence:
    """音符序列容器，提供時間索引和查詢功能"""

    def __init__(self, notes: List[Note], source: str = "input"):
        self.notes = sorted(notes, key=lambda n: (n.onset_ticks, n.pitch))
        self.source = source
        self._time_index: Dict[int, List[Note]] = {}
        self._build_time_index()

    def _build_time_index(self):
        """建立時間索引"""
        self._time_index.clear()
        for note in self.notes:
            if note.onset_ticks not in self._time_index:
                self._time_index[note.onset_ticks] = []
            self._time_index[note.onset_ticks].append(note)

    def get_notes_at_time(self, onset_ticks: int) -> List[Note]:
        """取得指定時間點的所有音符"""
        return self._time_index.get(onset_ticks, [])

    def get_notes_in_window(self, onset_ticks: int, window_size: int = 5) -> List[Note]:
        """
        取得 ±window_size 個音符
        用於 overlap correction algorithm
        """
        # 找出目標時間點的音符索引
        start_idx = 0
        for i, note in enumerate(self.notes):
            if note.onset_ticks >= onset_ticks:
                start_idx = i
                break

        # 提取 window
        window_start = max(0, start_idx - window_size)
        window_end = min(len(self.notes), start_idx + window_size + 1)

        return self.notes[window_start:window_end]

    def get_notes_in_time_range(self, start_ticks: int, end_ticks: int) -> List[Note]:
        """取得時間範圍內的所有音符"""
        return [n for n in self.notes if start_ticks <= n.onset_ticks <= end_ticks]

    def find_closest_note(self, target_pitch: int, onset_ticks: int,
                          max_time_diff: int = 480) -> Optional[Note]:
        """找出最接近的音符（用於配對）"""
        candidates = self.get_notes_in_time_range(
            onset_ticks - max_time_diff,
            onset_ticks + max_time_diff
        )

        if not candidates:
            return None

        # 評分：pitch 差異（主要）+ 時間差異（次要）
        def score_note(note: Note) -> Tuple[int, int]:
            pitch_diff = abs(note.pitch - target_pitch)
            time_diff = abs(note.onset_ticks - onset_ticks)
            return (pitch_diff, time_diff)

        return min(candidates, key=score_note)

    def __len__(self):
        return len(self.notes)

    def __iter__(self):
        return iter(self.notes)
```

#### 4.2.3 PostProcessor (`processor.py`) - 核心演算法

```python
class PostProcessor:
    """Post-Processing 核心演算法"""

    def __init__(self, guitar_config: GuitarConfig, window_size: int = 5):
        self.config = guitar_config
        self.validator = PitchValidator(guitar_config)
        self.window_size = window_size

    def overlap_correction(self,
                          model_output: NoteSequence,
                          input_sequence: NoteSequence) -> NoteSequence:
        """Overlap correction algorithm (Section 3.5)"""
        # 詳見 Section 3.1 的 pseudocode
        pass

    def neighbor_search(self,
                       corrected_sequence: NoteSequence,
                       optimize_for: str = "playability") -> NoteSequence:
        """Neighbor search algorithm (Section 4.2)"""
        # 詳見 Section 3.2 的 pseudocode
        pass

    def process(self,
                model_output: NoteSequence,
                input_sequence: NoteSequence,
                apply_neighbor_search: bool = True) -> NoteSequence:
        """完整 post-processing pipeline"""
        # Step 1: Overlap correction
        corrected = self.overlap_correction(model_output, input_sequence)

        # Step 2: Neighbor search (optional)
        if apply_neighbor_search:
            refined = self.neighbor_search(corrected)
            return refined

        return corrected

    # Private helper methods
    def _find_best_match(self, model_note, candidate_notes):
        """找出最佳配對音符"""
        pass

    def _create_fallback_note(self, model_note):
        """建立 fallback 音符"""
        pass

    def _get_context_notes(self, sequence, current_idx, context_window=3):
        """取得 context 音符"""
        pass

    def _select_best_position(self, note, alternatives, context, optimize_for):
        """選擇最佳 string-fret 位置"""
        pass
```

#### 4.2.4 FrettingPostProcessor (`api.py`) - 主 API

```python
class FrettingPostProcessor:
    """
    Fretting Post-Processor 主 API

    使用範例:
        processor = FrettingPostProcessor(guitar_config)
        corrected_tokens = processor.process_tokens(
            model_output_tokens,
            input_note_tokens,
            method='neighbor_search'
        )
    """

    def __init__(self, guitar_config: Optional[GuitarConfig] = None):
        if guitar_config is None:
            guitar_config = GuitarConfig()

        self.config = guitar_config
        self.parser = TokenParser()
        self.post_processor = PostProcessor(guitar_config)
        self.serializer = TokenSerializer()
        self.evaluator = PostProcessingEvaluator(guitar_config)

    def process_tokens(self,
                      model_output_tokens: List[str],
                      input_note_tokens: List[str],
                      method: str = 'neighbor_search') -> List[str]:
        """
        處理 token 序列並返回修正後的輸出

        Args:
            model_output_tokens: 模型預測 (TAB format)
            input_note_tokens: Ground truth notes (NOTE_ON/OFF format)
            method: 'overlap' 或 'neighbor_search'

        Returns:
            修正後的 TAB tokens
        """
        # Step 1: Parse tokens
        input_sequence = self.parser.parse_input_tokens(input_note_tokens)
        model_output = self.parser.parse_output_tokens(
            model_output_tokens,
            input_sequence,
            self.config
        )

        # Step 2: Apply post-processing
        if method == 'overlap':
            corrected = self.post_processor.overlap_correction(
                model_output,
                input_sequence
            )
        elif method == 'neighbor_search':
            corrected = self.post_processor.process(
                model_output,
                input_sequence,
                apply_neighbor_search=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Step 3: Serialize back to tokens
        output_tokens = self.serializer.serialize_to_output_format(corrected)

        return output_tokens

    def evaluate(self,
                model_output_tokens: List[str],
                input_note_tokens: List[str],
                ground_truth_tab_tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        評估 post-processing 效果（複製 Table 2）

        Returns:
            {
                'raw_model': {'pitch_accuracy': 97.23, 'tab_accuracy': 68.56},
                'overlap_correction': {'pitch_accuracy': 99.92, ...},
                'neighbor_search': {'pitch_accuracy': 100.00, ...}
            }
        """
        # Parse inputs
        input_sequence = self.parser.parse_input_tokens(input_note_tokens)
        model_output = self.parser.parse_output_tokens(
            model_output_tokens,
            input_sequence,
            self.config
        )

        # Apply both methods
        overlap_corrected = self.post_processor.overlap_correction(
            model_output,
            input_sequence
        )

        neighbor_refined = self.post_processor.neighbor_search(overlap_corrected)

        # Evaluate
        return self.evaluator.compare_methods(
            model_output,
            overlap_corrected,
            neighbor_refined,
            input_sequence
        )
```

---

## 5. 實作步驟

### Phase 1: 基礎資料結構 (1-2 days)

**檔案**: `datatypes.py`, `config.py`

#### 任務清單

- [ ] 實作 `TokenType` enum
  ```python
  class TokenType(Enum):
      NOTE_ON = "NOTE_ON"
      NOTE_OFF = "NOTE_OFF"
      TIME_SHIFT = "TIME_SHIFT"
      TAB = "TAB"
  ```

- [ ] 實作 `Token` dataclass
  ```python
  @dataclass
  class Token:
      token_type: TokenType
      value: int
      string_fret: Optional[Tuple[int, int]] = None
      position: int = 0
  ```

- [ ] 實作 `Note` dataclass
  - 參考 `gp2jams/guitarpro_utils.py:37-78`
  - 新增方法：
    - `has_tablature() -> bool`
    - `get_pitch_from_tablature(tuning) -> Optional[int]`
    - `get_offset_ticks() -> int`

- [ ] 實作 `GuitarConfig` class
  - 預設 tuning: `(40, 45, 50, 55, 59, 64)`
  - 方法：
    - `get_effective_tuning() -> Tuple[int, ...]`
    - `pitch_to_string_fret(pitch) -> List[Tuple[int, int]]` ⭐ 核心功能
    - `is_valid_string(string) -> bool`
    - `is_valid_fret(fret) -> bool`
    - `get_pitch_range() -> Tuple[int, int]`

- [ ] 定義常見 tunings
  - `STANDARD_TUNING`
  - `DROP_D_TUNING`
  - `HALF_STEP_DOWN`
  - `FULL_STEP_DOWN`

#### 測試重點

- [ ] `pitch_to_string_fret()` 正確性
  - 測試 E2 (pitch=40) → [(0, 0)]
  - 測試 A2 (pitch=45) → [(0, 5), (1, 0)]
  - 測試超出範圍的 pitch → []

- [ ] Capo 功能
  - Capo 2: E2 (40) 變成 F#2 (42)

---

### Phase 2: Token 解析與序列化 (2-3 days)

**檔案**: `parser.py`, `serializer.py`, `sequence.py`

#### 任務清單

**`parser.py` - TokenParser**

- [ ] 實作 `parse_input_tokens()`
  - [ ] 使用 regex 解析 `NOTE_ON<pitch>`
  - [ ] 使用 regex 解析 `NOTE_OFF<pitch>`
  - [ ] 使用 regex 解析 `TIME_SHIFT<ticks>`
  - [ ] 追蹤 active notes dictionary: `{pitch: onset_time}`
  - [ ] 計算 duration: `note_off_time - note_on_time`
  - [ ] 建立 `Note` 物件並返回 `NoteSequence`

- [ ] 實作 `parse_output_tokens()`
  - [ ] 解析 `TAB<string,fret>` tokens
  - [ ] 解析 `TIME_SHIFT<ticks>` tokens
  - [ ] 從 tablature 計算 pitch: `fret + tuning[string]`
  - [ ] 配對 input sequence 推斷 duration/velocity
  - [ ] 處理 buffer（同時發生的音符）

**`sequence.py` - NoteSequence**

- [ ] 實作 `__init__()`
  - [ ] 按 (onset_ticks, pitch) 排序
  - [ ] 建立時間索引 `_time_index`

- [ ] 實作查詢方法
  - [ ] `get_notes_at_time(onset_ticks)`
  - [ ] `get_notes_in_window(onset_ticks, window_size)` ⭐ 核心
  - [ ] `get_notes_in_time_range(start, end)`
  - [ ] `find_closest_note(pitch, onset)`

**`serializer.py` - TokenSerializer**

- [ ] 實作 `serialize_to_input_format()`
  - [ ] 建立 note-on/note-off events
  - [ ] 按時間排序
  - [ ] 生成 TIME_SHIFT tokens
  - [ ] 處理同時發生的 events

- [ ] 實作 `serialize_to_output_format()`
  - [ ] 按 onset 分組音符
  - [ ] 生成 TAB tokens
  - [ ] 生成 TIME_SHIFT tokens

#### 測試重點

- [ ] Token parsing 正確性
  ```python
  tokens = ["NOTE_ON<60>", "TIME_SHIFT<480>", "NOTE_OFF<60>"]
  sequence = parser.parse_input_tokens(tokens)
  assert len(sequence) == 1
  assert sequence.notes[0].pitch == 60
  assert sequence.notes[0].duration_ticks == 480
  ```

- [ ] Window query 正確性
  ```python
  notes = sequence.get_notes_in_window(onset=1000, window_size=5)
  # 應該返回 onset 在 [前5個音符, 後5個音符] 範圍內的 notes
  ```

- [ ] Round-trip 測試
  ```python
  original_tokens = [...]
  sequence = parser.parse_input_tokens(original_tokens)
  reconstructed_tokens = serializer.serialize_to_input_format(sequence)
  assert original_tokens == reconstructed_tokens
  ```

---

### Phase 3: Pitch Validation (1-2 days)

**檔案**: `validator.py`

#### 任務清單

- [ ] 實作 `PitchValidator.__init__()`
  - [ ] 儲存 `guitar_config`

- [ ] 實作 `validate_note(note)`
  - [ ] 檢查 `string` 是否在有效範圍
  - [ ] 檢查 `fret` 是否在有效範圍
  - [ ] 驗證 `fret + tuning[string] == pitch`

- [ ] 實作 `correct_note_tablature(note, preferred_string)`
  - [ ] 使用 `pitch_to_string_fret()` 找出有效位置
  - [ ] 如果有 `preferred_string`，優先使用
  - [ ] Fallback：使用第一個有效位置（最低弦）
  - [ ] 更新 note 的 string/fret
  - [ ] 返回 success/failure

- [ ] 實作 `get_alternative_positions(note, exclude_current)`
  - [ ] 取得所有 (string, fret) 組合
  - [ ] 如果 `exclude_current=True`，排除當前位置
  - [ ] 返回 list

#### 測試重點

- [ ] Validation 正確性
  ```python
  note = Note(pitch=45, string=1, fret=0, ...)  # A2 on string 1
  assert validator.validate_note(note) == True

  note.fret = 5  # Wrong fret
  assert validator.validate_note(note) == False
  ```

- [ ] Correction 功能
  ```python
  note = Note(pitch=45, string=None, fret=None, ...)
  validator.correct_note_tablature(note)
  assert note.string == 0 and note.fret == 5  # or (1, 0)
  ```

- [ ] Alternative positions
  ```python
  note = Note(pitch=50, ...)  # D3
  alts = validator.get_alternative_positions(note)
  # Expected: [(0, 10), (1, 5), (2, 0)]
  ```

---

### Phase 4: Overlap Correction Algorithm (3-4 days)

**檔案**: `processor.py`

#### 任務清單

**Helper Methods**

- [ ] 實作 `_find_best_match(model_note, candidate_notes)`
  - [ ] 計算 predicted_pitch from tablature
  - [ ] FOR EACH candidate:
    - [ ] 計算 pitch_diff
    - [ ] 計算 time_diff
    - [ ] 計算 duration_diff
    - [ ] 計算 score: `(pitch_diff * 1000) + (time_diff * 10) + duration_diff`
  - [ ] 選擇最低 score 的 candidate
  - [ ] 標記為 matched
  - [ ] 返回 best_match

- [ ] 實作 `_create_fallback_note(model_note)`
  - [ ] 取得 pitch from tablature or note.pitch
  - [ ] 建立新 Note 物件
  - [ ] 使用 `validator.correct_note_tablature()` 生成有效 tablature
  - [ ] 返回 fallback_note

**Main Algorithm**

- [ ] 實作 `overlap_correction(model_output, input_sequence)`
  - [ ] FOR EACH model_note IN model_output:
    - [ ] Step 1: Get predicted_pitch
    - [ ] Step 2: Get candidates in ±5 window
    - [ ] Step 3: Find best_match
    - [ ] Step 4a: If match found:
      - [ ] Create corrected_note with input pitch
      - [ ] Try to preserve model's string
      - [ ] Validate tablature
      - [ ] Use fallback if invalid
    - [ ] Step 4b: If no match:
      - [ ] Create fallback_note
    - [ ] Add to corrected_notes
  - [ ] Return NoteSequence(corrected_notes)

#### 測試重點

- [ ] **單元測試**: `_find_best_match()`
  - 測試完全匹配（pitch + time 相同）
  - 測試 pitch 匹配但 time 不同
  - 測試沒有候選音符的情況

- [ ] **單元測試**: `_create_fallback_note()`
  - 測試生成有效的 tablature
  - 測試超出範圍的 pitch

- [ ] **整合測試**: `overlap_correction()`
  - 測試完整序列處理
  - 測試 pitch accuracy 提升（應接近 99.92%）
  - 測試 edge cases（和弦、missing notes）

#### 預期效果驗證

```python
# 測試資料
model_output_tokens = [...]  # 含 pitch errors
input_note_tokens = [...]    # Ground truth

# 處理
corrected = processor.overlap_correction(model_output, input_sequence)

# 評估
evaluator = PostProcessingEvaluator(config)
results = evaluator.evaluate_pitch_accuracy(corrected, input_sequence)

# 驗證
assert results['pitch_accuracy'] >= 99.0  # 應接近 99.92%
```

---

### Phase 5: Neighbor Search Algorithm (3-4 days)

**檔案**: `processor.py`

#### 任務清單

**Helper Methods**

- [ ] 實作 `_get_context_notes(sequence, current_idx, context_window=3)`
  - [ ] 計算 prev_start = max(0, idx - window)
  - [ ] 計算 next_end = min(len, idx + window + 1)
  - [ ] 提取 previous_notes
  - [ ] 提取 following_notes
  - [ ] 返回 (previous, following)

- [ ] 實作 `_select_best_position(note, alternatives, context, optimize_for)`
  - [ ] FOR EACH (string, fret) IN alternatives:
    - [ ] 初始化 score = 0
    - [ ] **Factor 1**: String consistency
      - [ ] Check previous 2 notes
      - [ ] If same string: score -= 20
    - [ ] **Factor 2**: Position proximity
      - [ ] If prev on same string: score += |fret_diff| * 5
    - [ ] **Factor 3**: Playability
      - [ ] If optimize_for == "playability": score += fret * 0.5
    - [ ] **Factor 4**: Avoid extremes
      - [ ] If fret > 15: score += 10
  - [ ] Return position with min_score

**Main Algorithm**

- [ ] 實作 `neighbor_search(corrected_sequence, optimize_for)`
  - [ ] FOR EACH note IN corrected_sequence AT index i:
    - [ ] Step 1: Get alternative_positions
    - [ ] If only 1 position: keep it
    - [ ] Step 2: Get context_notes
    - [ ] Step 3: Score each alternative
    - [ ] Step 4: Select best_position
    - [ ] Create refined_note with optimal position
    - [ ] Add to refined_notes
  - [ ] Return NoteSequence(refined_notes)

#### 測試重點

- [ ] **單元測試**: `_get_context_notes()`
  - 測試 middle position（正常 window）
  - 測試 start position（window 被截斷）
  - 測試 end position（window 被截斷）

- [ ] **單元測試**: `_select_best_position()`
  - 測試 string consistency factor
    ```python
    # prev_notes 都在 string 2
    # 應該偏好 string 2 的替代位置
    ```
  - 測試 position proximity factor
    ```python
    # prev fret = 5, alternatives = [(2, 3), (2, 15)]
    # 應該選擇 (2, 3)（距離較近）
    ```
  - 測試 playability factor
    ```python
    # alternatives = [(1, 10), (2, 2)]
    # 應該偏好 (2, 2)（低品格）
    ```

- [ ] **整合測試**: `neighbor_search()`
  - 測試完整序列優化
  - 測試不同 optimize_for 參數
  - 驗證 pitch 保持不變（100% accuracy）

#### 預期效果驗證

```python
# 處理
overlap_corrected = processor.overlap_correction(model_output, input_sequence)
neighbor_refined = processor.neighbor_search(overlap_corrected)

# 評估
results = evaluator.evaluate_pitch_accuracy(neighbor_refined, input_sequence)

# 驗證
assert results['pitch_accuracy'] == 100.0  # 應達到 100%
```

---

### Phase 6: 高階 API 與評估 (2-3 days)

**檔案**: `api.py`, `evaluator.py`

#### 任務清單

**`evaluator.py` - PostProcessingEvaluator**

- [ ] 實作 `evaluate_pitch_accuracy(predicted, ground_truth)`
  - [ ] FOR EACH gt_note IN ground_truth:
    - [ ] Find matching pred_note
    - [ ] Check pitch accuracy: `pred.pitch == gt.pitch`
    - [ ] Check tab accuracy: `(pred.string, pred.fret) == (gt.string, gt.fret)`
  - [ ] 計算統計資料
  - [ ] 返回 dict:
    ```python
    {
        'pitch_accuracy': 99.92,
        'tablature_accuracy': 72.15,
        'total_notes': 150,
        'pitch_correct': 149,
        'tablature_correct': 108
    }
    ```

- [ ] 實作 `compare_methods(model_output, overlap_corrected, neighbor_refined, gt)`
  - [ ] 評估 raw model
  - [ ] 評估 overlap correction
  - [ ] 評估 neighbor search
  - [ ] 返回比較結果（Table 2 格式）

**`api.py` - FrettingPostProcessor**

- [ ] 實作 `__init__(guitar_config)`
  - [ ] 初始化所有元件：
    - [ ] TokenParser
    - [ ] PostProcessor
    - [ ] TokenSerializer
    - [ ] PostProcessingEvaluator

- [ ] 實作 `process_tokens(model_output_tokens, input_note_tokens, method)`
  - [ ] Step 1: Parse tokens
  - [ ] Step 2: Apply post-processing
    - [ ] If method == 'overlap': overlap_correction()
    - [ ] If method == 'neighbor_search': process(apply_neighbor_search=True)
  - [ ] Step 3: Serialize to tokens
  - [ ] Return corrected_tokens

- [ ] 實作 `evaluate(model_output_tokens, input_note_tokens, gt_tab_tokens)`
  - [ ] Parse inputs
  - [ ] Apply both methods
  - [ ] Evaluate using evaluator
  - [ ] Return comparison results

#### 測試重點

- [ ] **API 測試**: 完整 workflow
  ```python
  processor = FrettingPostProcessor()
  corrected = processor.process_tokens(model_output, input_notes, method='neighbor_search')
  assert isinstance(corrected, list)
  assert all(isinstance(t, str) for t in corrected)
  ```

- [ ] **評估測試**: Table 2 複製
  ```python
  results = processor.evaluate(model_output, input_notes)
  assert 'raw_model' in results
  assert 'overlap_correction' in results
  assert 'neighbor_search' in results
  assert results['neighbor_search']['pitch_accuracy'] >= 99.0
  ```

---

### Phase 7: 與現有 Pipeline 整合 (2-3 days)

**檔案**: `utils.py` + integration scripts

#### 任務清單

**JAMS to Tokens 轉換**

- [ ] 實作 `jams_to_tokens(jams_path)`
  - [ ] 讀取 JAMS file（參考 `jams2midi.py:6-16`）
  - [ ] 提取 annotations
  - [ ] 提取 guitar configuration:
    - [ ] tuning from `sandbox.open_tuning`
    - [ ] fret_count from `sandbox.fret_count`
    - [ ] instrument from `sandbox.instrument`
  - [ ] 轉換 notes to input tokens:
    - [ ] 建立 note-on/note-off events
    - [ ] 生成 TIME_SHIFT tokens
  - [ ] 轉換 notes to output tokens (if available):
    - [ ] 生成 TAB tokens from string/fret
  - [ ] 返回 (input_tokens, output_tokens, guitar_config)

**Tokens to MIDI 轉換**

- [ ] 實作 `tokens_to_midi(tokens, output_path, guitar_config)`
  - [ ] Parse tokens to NoteSequence
  - [ ] 建立 MIDI file（參考 `jams2midi.py:73-104`）
  - [ ] 設定 ticks_per_beat = 960
  - [ ] 建立 events:
    - [ ] note-on/note-off with channel = string
    - [ ] tempo changes (if needed)
  - [ ] 排序 events by time
  - [ ] 寫入 MIDI messages
  - [ ] 儲存檔案

**完整處理 Script**

- [ ] 建立 `process_jams_with_postprocessing.py`
  ```python
  def process_jams_file(jams_path, model_output_tokens, output_midi_path):
      # 1. Load JAMS and extract config
      input_tokens, _, config = jams_to_tokens(jams_path)

      # 2. Apply post-processing
      processor = FrettingPostProcessor(config)
      corrected_tokens = processor.process_tokens(
          model_output_tokens,
          input_tokens,
          method='neighbor_search'
      )

      # 3. Convert to MIDI
      tokens_to_midi(corrected_tokens, output_midi_path, config)
  ```

#### 測試重點

- [ ] **JAMS 讀取測試**
  ```python
  input_tokens, output_tokens, config = jams_to_tokens('test.jams')
  assert isinstance(config.tuning, tuple)
  assert len(input_tokens) > 0
  ```

- [ ] **MIDI 生成測試**
  ```python
  tokens_to_midi(corrected_tokens, 'output.mid', config)
  assert os.path.exists('output.mid')
  midi = mido.MidiFile('output.mid')
  assert midi.ticks_per_beat == 960
  ```

- [ ] **端到端測試**
  ```python
  # Input: JAMS file + model predictions
  # Output: Corrected MIDI file
  process_jams_file('input.jams', model_output_tokens, 'output.mid')
  # Verify MIDI file correctness
  ```

---

### Phase 8: 測試與驗證 (2-3 days)

#### 任務清單

**單元測試**

- [ ] `test_config.py`
  - [ ] Test `pitch_to_string_fret()`
  - [ ] Test capo functionality
  - [ ] Test different tunings

- [ ] `test_parser.py`
  - [ ] Test input token parsing
  - [ ] Test output token parsing
  - [ ] Test malformed tokens

- [ ] `test_sequence.py`
  - [ ] Test time indexing
  - [ ] Test window queries
  - [ ] Test closest note finding

- [ ] `test_validator.py`
  - [ ] Test note validation
  - [ ] Test tablature correction
  - [ ] Test alternative positions

- [ ] `test_processor.py`
  - [ ] Test overlap correction
  - [ ] Test neighbor search
  - [ ] Test helper methods

**整合測試**

- [ ] `test_integration.py`
  - [ ] Test complete pipeline
  - [ ] Test with real JAMS files
  - [ ] Test edge cases:
    - [ ] Chords（同時多音）
    - [ ] Out of range notes
    - [ ] Empty sequences
    - [ ] Missing tokens

**效能驗證**

- [ ] 複製 Table 2 結果
  - [ ] 在 GuitarToday dataset 測試
  - [ ] 在 DadaGP dataset 測試
  - [ ] 在 Leduc dataset 測試
  - [ ] 驗證 accuracy 指標

- [ ] 建立 benchmark script
  ```python
  # benchmark.py
  def benchmark_on_dataset(dataset_name, jams_files):
      results = []
      for jams_path in jams_files:
          # Process and evaluate
          ...

      # Compute average accuracy
      avg_pitch_acc = mean([r['pitch_accuracy'] for r in results])

      print(f"{dataset_name}:")
      print(f"  Raw model: {raw_avg:.2f}%")
      print(f"  Overlap: {overlap_avg:.2f}%")
      print(f"  Neighbor: {neighbor_avg:.2f}%")
  ```

---

## 6. Edge Cases 處理

### 6.1 Chords（同時發生的音符）

**問題**：多個音符在同一時間，每個 string 只能彈一個音

**範例**：
```python
# C major chord: C (48), E (52), G (55)
# 同時在 onset=0
```

**解決方案**：

```python
def handle_chord(notes_at_time: List[Note], guitar_config: GuitarConfig) -> List[Note]:
    """
    處理和弦，確保沒有 string 衝突
    """
    # 按 pitch 排序（低到高）
    sorted_notes = sorted(notes_at_time, key=lambda n: n.pitch)

    assigned_strings = set()
    valid_notes = []

    for note in sorted_notes:
        # 找出尚未分配的 string
        positions = guitar_config.pitch_to_string_fret(note.pitch)
        available = [(s, f) for s, f in positions if s not in assigned_strings]

        if available:
            note.string, note.fret = available[0]
            assigned_strings.add(note.string)
            valid_notes.append(note)
        else:
            # 無法分配，跳過此音符（或記錄警告）
            logging.warning(f"Cannot assign string for pitch {note.pitch} in chord")

    return valid_notes
```

**整合至 overlap_correction**：

```python
def overlap_correction(self, model_output, input_sequence):
    corrected_notes = []

    # Group notes by onset time
    notes_by_time = {}
    for note in corrected_notes:
        if note.onset_ticks not in notes_by_time:
            notes_by_time[note.onset_ticks] = []
        notes_by_time[note.onset_ticks].append(note)

    # Process each time point
    final_notes = []
    for onset_time in sorted(notes_by_time.keys()):
        notes_at_time = notes_by_time[onset_time]

        if len(notes_at_time) > 1:
            # Handle chord
            valid_notes = handle_chord(notes_at_time, self.config)
            final_notes.extend(valid_notes)
        else:
            final_notes.append(notes_at_time[0])

    return NoteSequence(final_notes)
```

---

### 6.2 Notes Outside Guitar Range

**問題**：音符超出吉他可演奏範圍

**範例**：
```python
# E1 (pitch=28) - 低於最低弦 E2 (40)
# C6 (pitch=84) - 高於最高音
```

**解決方案**：

```python
def handle_out_of_range(note: Note, guitar_config: GuitarConfig) -> Optional[Note]:
    """
    處理超出範圍的音符

    策略:
    1. 嘗試 transpose ±12 semitones (octave)
    2. 如果仍超出範圍，返回 None（捨棄）
    """
    min_pitch, max_pitch = guitar_config.get_pitch_range()

    # 太低 - 嘗試 transpose up
    if note.pitch < min_pitch:
        note.pitch += 12
        if note.pitch > max_pitch:
            note.pitch -= 24  # Try down instead

    # 太高 - 嘗試 transpose down
    elif note.pitch > max_pitch:
        note.pitch -= 12
        if note.pitch < min_pitch:
            note.pitch += 24  # Try up instead

    # 驗證最終結果
    if not (min_pitch <= note.pitch <= max_pitch):
        logging.warning(f"Pitch {note.pitch} cannot be played on this guitar")
        return None

    return note
```

**整合至 _create_fallback_note**：

```python
def _create_fallback_note(self, model_note):
    """建立 fallback 音符，處理超出範圍"""
    fallback = Note(
        pitch=model_note.pitch,
        onset_ticks=model_note.onset_ticks,
        duration_ticks=model_note.duration_ticks,
        velocity=model_note.velocity,
        source="fallback"
    )

    # Check if in range
    fallback = handle_out_of_range(fallback, self.config)
    if fallback is None:
        # Cannot be played - use closest valid note
        min_pitch, max_pitch = self.config.get_pitch_range()
        fallback.pitch = max(min_pitch, min(max_pitch, fallback.pitch))

    # Generate valid tablature
    self.validator.correct_note_tablature(fallback)

    return fallback
```

---

### 6.3 Invalid String-Fret Combinations

**問題**：模型輸出無效的 (string, fret) 組合

**範例**：
```python
# string=0 (E2), fret=30 (超出 max_fret=24)
# string=7 (不存在於 6 弦吉他)
# fret=-1 (負數品格)
```

**解決方案**：

```python
def validate_and_fix_tablature(note: Note,
                               guitar_config: GuitarConfig,
                               validator: PitchValidator) -> bool:
    """
    驗證並修正 tablature

    Returns:
        True if valid or successfully corrected, False otherwise
    """
    # Check basic validity
    if not guitar_config.is_valid_string(note.string):
        logging.warning(f"Invalid string: {note.string}")
        return validator.correct_note_tablature(note)

    if not guitar_config.is_valid_fret(note.fret):
        logging.warning(f"Invalid fret: {note.fret}")
        return validator.correct_note_tablature(note)

    # Check pitch correspondence
    if not validator.validate_note(note):
        logging.warning(f"Tablature doesn't match pitch for note {note}")
        return validator.correct_note_tablature(note)

    return True
```

**自動修正策略**：

1. **String 無效**：使用 `pitch_to_string_fret()` 找第一個有效位置
2. **Fret 無效**：
   - 如果 fret < 0：設為 0
   - 如果 fret > max_fret：找替代 string
3. **Pitch 不符**：重新計算 fret = pitch - tuning[string]

---

### 6.4 Missing or Malformed Tokens

**問題**：Token 格式錯誤或遺失

**範例**：
```python
# 格式錯誤
"NOTE_ON<abc>"  # 非數字
"TAB<1>"        # 缺少 fret
"INVALID<55>"   # 未知 token type

# 遺失
"NOTE_ON<60> TIME_SHIFT<240>"  # 缺少 NOTE_OFF
```

**解決方案**：

```python
def robust_token_parsing(token_str: str) -> Optional[Token]:
    """
    Robust token parsing with error handling
    """
    import re
    import warnings

    # Try all token patterns
    patterns = [
        (r'NOTE_ON<(\d+)>', TokenType.NOTE_ON),
        (r'NOTE_OFF<(\d+)>', TokenType.NOTE_OFF),
        (r'TIME_SHIFT<(\d+)>', TokenType.TIME_SHIFT),
        (r'TAB<(\d+),(\d+)>', TokenType.TAB),
    ]

    for pattern, token_type in patterns:
        match = re.match(pattern, token_str)
        if match:
            if token_type == TokenType.TAB:
                string = int(match.group(1))
                fret = int(match.group(2))
                return Token(token_type, value=0, string_fret=(string, fret))
            else:
                value = int(match.group(1))
                return Token(token_type, value)

    # No match - log warning
    warnings.warn(f"Malformed token: {token_str}")
    return None

# 在 TokenParser 中使用
def parse_input_tokens(self, tokens: List[str]):
    parsed_tokens = []
    for token_str in tokens:
        token = robust_token_parsing(token_str)
        if token is not None:
            parsed_tokens.append(token)

    # Process parsed tokens...
```

**處理遺失 NOTE_OFF**：

```python
def parse_input_tokens(self, tokens):
    """Parse with handling for missing NOTE_OFF"""
    notes = []
    active_notes = {}  # pitch -> onset_time
    current_time = 0

    for token in tokens:
        if token.token_type == TokenType.NOTE_ON:
            if token.value in active_notes:
                # Missing NOTE_OFF for previous note with same pitch
                # Terminate it now
                onset = active_notes.pop(token.value)
                duration = current_time - onset
                notes.append(Note(pitch=token.value, onset_ticks=onset,
                                 duration_ticks=duration, velocity=64))

            active_notes[token.value] = current_time

        elif token.token_type == TokenType.NOTE_OFF:
            if token.value in active_notes:
                onset = active_notes.pop(token.value)
                duration = current_time - onset
                notes.append(Note(pitch=token.value, onset_ticks=onset,
                                 duration_ticks=duration, velocity=64))

        elif token.token_type == TokenType.TIME_SHIFT:
            current_time += token.value

    # Handle any remaining active notes (missing NOTE_OFF at end)
    for pitch, onset in active_notes.items():
        duration = current_time - onset
        notes.append(Note(pitch=pitch, onset_ticks=onset,
                         duration_ticks=duration, velocity=64))

    return NoteSequence(notes)
```

---

### 6.5 Multiple Tunings and Capo Support

**問題**：支援不同 tuning 和 capo 位置

**解決方案已在 `GuitarConfig` 中實作**：

```python
# 使用範例
config = GuitarConfig(
    tuning=DROP_D_TUNING,  # (38, 45, 50, 55, 59, 64)
    capo_fret=2
)

# Effective tuning (考慮 capo)
effective = config.get_effective_tuning()
# Result: (40, 47, 52, 57, 61, 66)
#          D2+2, A2+2, D3+2, G3+2, B3+2, E4+2

# Pitch to string-fret (自動考慮 capo)
positions = config.pitch_to_string_fret(pitch=40)
# With capo 2: D2 (38) + 2 frets = E2 (40)
# Result: [(0, 0)]  # Open string 0 with capo at fret 2
```

**整合至 JAMS 讀取**：

```python
def create_guitar_config_from_jams(jams_data):
    """從 JAMS 檔案建立 GuitarConfig"""
    annotations = jams_data['annotations']

    # Extract tuning
    tuning = []
    for ann in annotations:
        if ann['namespace'] == 'note_tab':
            tuning.append(ann['sandbox']['open_tuning'])

    # Extract capo (if available)
    capo = jams_data.get('sandbox', {}).get('capo_fret', 0)

    # Extract fret count
    fret_count = jams_data.get('sandbox', {}).get('fret_count', 24)

    return GuitarConfig(
        num_strings=len(tuning),
        tuning=tuple(tuning),
        capo_fret=capo,
        max_fret=fret_count
    )
```

---

## 7. 與現有 Pipeline 整合

### 7.1 整合架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                    Existing Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GuitarPro files (.gp3, .gp4, .gp5)                        │
│           ↓                                                 │
│  [gp2jams/process_guitarpro.py]                            │
│           ↓                                                 │
│  JAMS files (note_tab + tempo annotations)                 │
│           ↓                                                 │
│  [jams2midi/jams2midi.py]                                  │
│           ↓                                                 │
│  MIDI files (multi-channel, per-string)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ Integration Point
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              New Post-Processing Module                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  JAMS file + Model Output Tokens                           │
│           ↓                                                 │
│  [fretting_postprocessor/utils.py]                         │
│  - jams_to_tokens()                                        │
│           ↓                                                 │
│  Input Tokens (NOTE_ON/OFF) + Output Tokens (TAB)          │
│           ↓                                                 │
│  [fretting_postprocessor/api.py]                           │
│  - FrettingPostProcessor.process_tokens()                  │
│           ↓                                                 │
│  Corrected TAB Tokens (100% pitch accuracy)                │
│           ↓                                                 │
│  [fretting_postprocessor/utils.py]                         │
│  - tokens_to_midi()                                        │
│           ↓                                                 │
│  Corrected MIDI file                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 JAMS to Tokens 轉換

**參考檔案**: `jams2midi/jams2midi.py:6-60`

**實作** (`utils.py`):

```python
import json
from typing import Tuple, List

def jams_to_tokens(jams_path: str) -> Tuple[List[str], List[str], GuitarConfig]:
    """
    將 JAMS 檔案轉換為 token 格式

    Args:
        jams_path: JAMS 檔案路徑

    Returns:
        (input_tokens, output_tokens, guitar_config)
    """
    # Load JAMS file
    with open(jams_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']

    # 1. Extract guitar configuration
    tuning = []
    fret_count = 24

    for ann in annotations:
        if ann['namespace'] == 'note_tab':
            tuning.append(ann['sandbox']['open_tuning'])

    if 'sandbox' in data:
        fret_count = data['sandbox'].get('fret_count', 24)
        capo = data['sandbox'].get('capo_fret', 0)
    else:
        capo = 0

    guitar_config = GuitarConfig(
        num_strings=len(tuning),
        tuning=tuple(tuning),
        capo_fret=capo,
        max_fret=fret_count
    )

    # 2. Collect all notes across strings
    all_notes = []

    for ann in annotations:
        if ann['namespace'] != 'note_tab':
            continue

        string_index = ann['sandbox']['string_index'] - 1  # Convert to 0-indexed
        open_tuning = ann['sandbox']['open_tuning']

        for note_data in ann['data']:
            pitch = note_data['value']['fret'] + open_tuning
            onset = int(note_data['time'])
            duration = int(note_data['duration'])
            velocity = note_data['value'].get('velocity', 64)
            fret = note_data['value']['fret']

            note = Note(
                pitch=pitch,
                onset_ticks=onset,
                duration_ticks=duration,
                velocity=velocity,
                string=string_index,
                fret=fret,
                source="jams"
            )
            all_notes.append(note)

    # Sort by onset time
    all_notes.sort(key=lambda n: (n.onset_ticks, n.pitch))

    # 3. Generate input tokens (NOTE_ON/OFF format)
    input_tokens = []
    events = []

    for note in all_notes:
        events.append((note.onset_ticks, 'on', note.pitch))
        events.append((note.onset_ticks + note.duration_ticks, 'off', note.pitch))

    events.sort(key=lambda e: (e[0], e[1] == 'off'))  # Sort by time, offs before ons

    current_time = 0
    for time, event_type, pitch in events:
        if time > current_time:
            input_tokens.append(f"TIME_SHIFT<{time - current_time}>")
            current_time = time

        if event_type == 'on':
            input_tokens.append(f"NOTE_ON<{pitch}>")
        else:
            input_tokens.append(f"NOTE_OFF<{pitch}>")

    # 4. Generate output tokens (TAB format)
    output_tokens = []
    notes_by_time = {}

    for note in all_notes:
        if note.onset_ticks not in notes_by_time:
            notes_by_time[note.onset_ticks] = []
        notes_by_time[note.onset_ticks].append(note)

    current_time = 0
    for onset_time in sorted(notes_by_time.keys()):
        if onset_time > current_time:
            output_tokens.append(f"TIME_SHIFT<{onset_time - current_time}>")
            current_time = onset_time

        for note in sorted(notes_by_time[onset_time], key=lambda n: n.string):
            output_tokens.append(f"TAB<{note.string},{note.fret}>")

    return (input_tokens, output_tokens, guitar_config)
```

### 7.3 Tokens to MIDI 轉換

**參考檔案**: `jams2midi/jams2midi.py:73-104`

**實作** (`utils.py`):

```python
import mido

def tokens_to_midi(tokens: List[str],
                  output_path: str,
                  guitar_config: GuitarConfig,
                  ticks_per_beat: int = 960):
    """
    將 corrected tokens 轉換為 MIDI 檔案

    Args:
        tokens: TAB/TIME_SHIFT format tokens
        output_path: 輸出 MIDI 檔案路徑
        guitar_config: 吉他配置
        ticks_per_beat: MIDI ticks per beat (預設 960)
    """
    # Parse tokens to notes
    parser = TokenParser()
    # Create dummy input sequence (for parsing reference)
    dummy_input = NoteSequence([], source="dummy")
    note_sequence = parser.parse_output_tokens(tokens, dummy_input, guitar_config)

    # Create MIDI file
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Add tempo (default 120 BPM)
    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    # Collect events
    events = []
    for note in note_sequence:
        if not note.has_tablature():
            continue

        channel = note.string  # Each string on separate channel (0-5)
        pitch = note.pitch
        velocity = note.velocity
        onset = note.onset_ticks
        offset = note.get_offset_ticks()

        events.append((onset, 'on', pitch, velocity, channel))
        events.append((offset, 'off', pitch, velocity, channel))

    # Sort events by time
    events.sort(key=lambda e: (e[0], e[1] == 'off'))

    # Generate MIDI messages
    current_time = 0
    for time, event_type, pitch, velocity, channel in events:
        delta = time - current_time

        if event_type == 'on':
            msg = mido.Message('note_on', note=pitch, velocity=velocity,
                             time=delta, channel=channel)
        else:
            msg = mido.Message('note_off', note=pitch, velocity=0,
                             time=delta, channel=channel)

        track.append(msg)
        current_time = time

    # Save MIDI file
    midi.save(output_path)
```

### 7.4 完整處理 Script

**新檔案**: `process_with_postprocessing.py`

```python
#!/usr/bin/env python3
"""
完整的 JAMS + Model Output → Corrected MIDI pipeline
"""

import argparse
from fretting_postprocessor import FrettingPostProcessor
from fretting_postprocessor.utils import jams_to_tokens, tokens_to_midi

def main():
    parser = argparse.ArgumentParser(
        description='Post-process model tablature predictions'
    )
    parser.add_argument('--jams', required=True,
                       help='Input JAMS file (ground truth)')
    parser.add_argument('--model-output', required=True,
                       help='Model output tokens file (TAB format)')
    parser.add_argument('--output-midi', required=True,
                       help='Output corrected MIDI file')
    parser.add_argument('--method', default='neighbor_search',
                       choices=['overlap', 'neighbor_search'],
                       help='Post-processing method')
    parser.add_argument('--evaluate', action='store_true',
                       help='Print evaluation metrics')

    args = parser.parse_args()

    # Step 1: Load JAMS and extract tokens + config
    print(f"Loading JAMS file: {args.jams}")
    input_tokens, gt_output_tokens, guitar_config = jams_to_tokens(args.jams)

    # Step 2: Load model output tokens
    print(f"Loading model output: {args.model_output}")
    with open(args.model_output, 'r') as f:
        model_output_tokens = [line.strip() for line in f if line.strip()]

    # Step 3: Apply post-processing
    print(f"Applying post-processing ({args.method})...")
    processor = FrettingPostProcessor(guitar_config)

    corrected_tokens = processor.process_tokens(
        model_output_tokens,
        input_tokens,
        method=args.method
    )

    # Step 4: Convert to MIDI
    print(f"Generating corrected MIDI: {args.output_midi}")
    tokens_to_midi(corrected_tokens, args.output_midi, guitar_config)

    # Step 5: Evaluate (optional)
    if args.evaluate:
        print("\nEvaluating post-processing effectiveness...")
        results = processor.evaluate(
            model_output_tokens,
            input_tokens,
            gt_output_tokens
        )

        print("\nResults:")
        print(f"  Raw Model:")
        print(f"    Pitch Accuracy: {results['raw_model']['pitch_accuracy']:.2f}%")
        print(f"    Tab Accuracy:   {results['raw_model']['tablature_accuracy']:.2f}%")

        print(f"\n  After Overlap Correction:")
        print(f"    Pitch Accuracy: {results['overlap_correction']['pitch_accuracy']:.2f}%")
        print(f"    Tab Accuracy:   {results['overlap_correction']['tablature_accuracy']:.2f}%")

        print(f"\n  After Neighbor Search:")
        print(f"    Pitch Accuracy: {results['neighbor_search']['pitch_accuracy']:.2f}%")
        print(f"    Tab Accuracy:   {results['neighbor_search']['tablature_accuracy']:.2f}%")

    print("\nDone!")

if __name__ == '__main__':
    main()
```

**使用範例**:

```bash
# Basic usage
python process_with_postprocessing.py \
    --jams Dataset/song.jams \
    --model-output model_predictions.txt \
    --output-midi corrected_output.mid

# With evaluation
python process_with_postprocessing.py \
    --jams Dataset/song.jams \
    --model-output model_predictions.txt \
    --output-midi corrected_output.mid \
    --method neighbor_search \
    --evaluate
```

---

## 8. 使用範例

### 8.1 Basic Usage

```python
from fretting_postprocessor import FrettingPostProcessor

# Initialize processor with default config (standard tuning)
processor = FrettingPostProcessor()

# Model output tokens (with potential pitch errors)
model_output = [
    "TAB<2,5>", "TIME_SHIFT<240>",
    "TAB<1,7>", "TIME_SHIFT<240>",
    "TAB<2,3>", "TIME_SHIFT<480>"
]

# Input note tokens (ground truth pitches)
input_notes = [
    "NOTE_ON<62>", "TIME_SHIFT<240>", "NOTE_OFF<62>",
    "NOTE_ON<64>", "TIME_SHIFT<240>", "NOTE_OFF<64>",
    "NOTE_ON<60>", "TIME_SHIFT<480>", "NOTE_OFF<60>"
]

# Apply post-processing
corrected_tokens = processor.process_tokens(
    model_output,
    input_notes,
    method='neighbor_search'  # or 'overlap'
)

print("Corrected tokens:", corrected_tokens)
# Expected: TAB tokens with 100% pitch accuracy
```

### 8.2 Custom Guitar Configuration

```python
from fretting_postprocessor import FrettingPostProcessor, GuitarConfig
from fretting_postprocessor import DROP_D_TUNING

# Drop-D tuning with capo on 2nd fret
config = GuitarConfig(
    tuning=DROP_D_TUNING,  # (38, 45, 50, 55, 59, 64)
    capo_fret=2,
    max_fret=22
)

processor = FrettingPostProcessor(guitar_config=config)

corrected_tokens = processor.process_tokens(
    model_output,
    input_notes,
    method='neighbor_search'
)
```

### 8.3 Evaluation

```python
# Evaluate post-processing effectiveness
results = processor.evaluate(
    model_output_tokens=model_output,
    input_note_tokens=input_notes,
    ground_truth_tab_tokens=ground_truth  # optional
)

# Print results (Table 2 format)
print(f"Raw model pitch accuracy: {results['raw_model']['pitch_accuracy']:.2f}%")
print(f"After overlap correction: {results['overlap_correction']['pitch_accuracy']:.2f}%")
print(f"After neighbor search: {results['neighbor_search']['pitch_accuracy']:.2f}%")

# Expected output (similar to paper Table 2):
# Raw model pitch accuracy: 97.23%
# After overlap correction: 99.92%
# After neighbor search: 100.00%
```

### 8.4 Processing JAMS Files

```python
from fretting_postprocessor.utils import jams_to_tokens, tokens_to_midi

# Load JAMS file
input_tokens, gt_tokens, config = jams_to_tokens('Dataset/song.jams')

# Load model predictions (from file)
with open('model_predictions.txt', 'r') as f:
    model_output = [line.strip() for line in f]

# Process
processor = FrettingPostProcessor(config)
corrected = processor.process_tokens(model_output, input_tokens)

# Save to MIDI
tokens_to_midi(corrected, 'corrected_output.mid', config)
```

### 8.5 Batch Processing

```python
import glob
from pathlib import Path

def batch_process_dataset(dataset_dir, model_outputs_dir, output_dir):
    """Batch process all JAMS files in a dataset"""

    jams_files = glob.glob(f"{dataset_dir}/**/*.jams", recursive=True)

    for jams_path in jams_files:
        # Load JAMS
        input_tokens, gt_tokens, config = jams_to_tokens(jams_path)

        # Find corresponding model output
        stem = Path(jams_path).stem
        model_output_path = f"{model_outputs_dir}/{stem}_predictions.txt"

        if not os.path.exists(model_output_path):
            continue

        # Load model output
        with open(model_output_path, 'r') as f:
            model_output = [line.strip() for line in f]

        # Process
        processor = FrettingPostProcessor(config)
        corrected = processor.process_tokens(model_output, input_tokens)

        # Save
        output_path = f"{output_dir}/{stem}_corrected.mid"
        tokens_to_midi(corrected, output_path, config)

        print(f"Processed: {jams_path} -> {output_path}")

# Run batch processing
batch_process_dataset(
    dataset_dir='Dataset',
    model_outputs_dir='model_predictions',
    output_dir='corrected_output'
)
```

---

## 9. 預期成果

### 9.1 功能性成果

✅ **完整的 Post-Processing Library**
- Python module: `fretting_postprocessor`
- 8 個核心模組，職責分明
- 清晰的 API 介面

✅ **v3 Token Format 支援**
- Input: `NOTE_ON<pitch> TIME_SHIFT<ticks> NOTE_OFF<pitch>`
- Output: `TAB<string,fret> TIME_SHIFT<ticks>`
- Robust parsing with error handling

✅ **兩種 Post-Processing 演算法**
- Overlap Correction (Section 3.5)
- Neighbor Search (Section 4.2)
- 完整實作並經過測試

✅ **多種 Tunings + Capo 支援**
- Standard, Drop-D, Half-step, Full-step
- Capo positions (0-12)
- 自動處理 effective tuning

✅ **完整評估框架**
- Pitch accuracy calculation
- Tab accuracy calculation
- Method comparison (複製 Table 2)

✅ **與現有 Pipeline 整合**
- JAMS to Tokens 轉換
- Tokens to MIDI 轉換
- Batch processing scripts

### 9.2 效能指標（基於論文 Table 2）

#### 在 Leduc Dataset 上的預期結果

| Method | Pitch Accuracy | Tab Accuracy |
|--------|----------------|--------------|
| **No Post-Processing** | 97.23% | 68.56% |
| **Overlap Correction** | 99.92% | 72.15% |
| **Overlap + Neighbor Search** | **100.00%** | 72.19% |

#### 在其他 Datasets 上的預期結果

**GuitarToday** (easier, beginner-friendly):
- Raw: ~98%
- Overlap: ~99.9%
- Neighbor: **100%**

**DadaGP** (diverse genres, variable quality):
- Raw: ~81%
- Overlap: ~99%
- Neighbor: **100%**

### 9.3 程式碼品質

✅ **模組化設計**
- 單一職責原則
- 低耦合、高內聚
- 易於測試和維護

✅ **型別提示**
- 完整的 type hints
- 使用 `dataclasses` 和 `typing`
- IDE 友善

✅ **文件齊全**
- Docstrings for all public methods
- Usage examples
- Inline comments for complex logic

✅ **測試覆蓋**
- 單元測試（每個模組）
- 整合測試（完整 pipeline）
- Edge cases 測試

✅ **易於整合**
- 清晰的 API 介面
- 與現有 codebase 兼容
- 獨立的 package 結構

---

## 10. 時間規劃

### 總時程：16-24 days

| Phase | 工作內容 | 預估時間 | 關鍵里程碑 |
|-------|---------|---------|-----------|
| **Phase 1** | 基礎資料結構 | 1-2 days | `Note`, `GuitarConfig` 完成並測試 |
| **Phase 2** | Token 解析與序列化 | 2-3 days | Parser, Serializer, NoteSequence 完成 |
| **Phase 3** | Pitch Validation | 1-2 days | PitchValidator 完成，所有驗證邏輯測試通過 |
| **Phase 4** | Overlap Correction | 3-4 days | ⭐ 核心演算法完成，達到 ~99.92% accuracy |
| **Phase 5** | Neighbor Search | 3-4 days | ⭐ 優化演算法完成，達到 100% accuracy |
| **Phase 6** | 高階 API 與評估 | 2-3 days | FrettingPostProcessor API 完成 |
| **Phase 7** | Pipeline 整合 | 2-3 days | JAMS/MIDI 轉換完成，可處理完整 pipeline |
| **Phase 8** | 測試與驗證 | 2-3 days | 所有測試通過，Table 2 結果複製成功 |

### 關鍵檢查點

**Week 1 End**:
- [ ] 基礎資料結構完成（Phase 1-2）
- [ ] Token parsing 功能驗證

**Week 2 End**:
- [ ] Overlap correction 完成並達到目標 accuracy
- [ ] 中期 demo：展示 pitch correction 效果

**Week 3 End**:
- [ ] Neighbor search 完成
- [ ] 完整 API 可用

**Week 4 End** (如需要):
- [ ] 所有測試通過
- [ ] 文件完成
- [ ] 與現有 pipeline 完全整合

---

## 11. 附錄

### A. 相關論文章節對照

| 本計畫章節 | 論文章節 | 頁碼 | 說明 |
|-----------|---------|------|------|
| 2.1 Token 格式 | Table 1 (v3 encoding) | p.5 | Input/Output token 定義 |
| 3.1 Overlap Correction | Section 3.5 | p.4 | 主要演算法說明 |
| 3.2 Neighbor Search | Section 4.2 + Table 2 | p.5 | 優化演算法與效果 |
| 9.2 效能指標 | Table 2 | p.5 | Leduc dataset 結果 |

### B. 關鍵檔案對照表

| 現有檔案 | 參考內容 | 新模組對應 |
|---------|---------|-----------|
| `gp2jams/guitarpro_utils.py:37-78` | Note class 設計 | `datatypes.py` - Note |
| `gp2jams/guitarpro_utils.py:181-233` | NoteTracker, tuning | `config.py` - GuitarConfig |
| `jams2midi/jams2midi.py:6-60` | JAMS 讀取, pitch 計算 | `utils.py` - jams_to_tokens |
| `jams2midi/jams2midi.py:73-104` | MIDI 生成 | `utils.py` - tokens_to_midi |

### C. 依賴套件

```
# requirement.txt 新增項目
dataclasses>=0.6  # Python 3.7+ (built-in in 3.7+)
typing>=3.7       # Python 3.5+ (built-in in 3.5+)
mido>=1.2.10      # MIDI processing (already in requirement.txt)
```

### D. 參考資源

1. **論文**: Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription
   - arXiv:2506.14223v1 [cs.SD] 17 Jun 2025
   - `/work/b10502010/GuitarTab/Fretting-Transformer.pdf`

2. **現有 Codebase**:
   - `/work/b10502010/GuitarTab/gp2jams/`
   - `/work/b10502010/GuitarTab/jams2midi/`

3. **Datasets** (用於測試):
   - GuitarToday: 363 files (easy fingerstyle)
   - DadaGP: 2,301 acoustic guitar tracks
   - Leduc: 232 jazz guitar tablatures

---

## 總結

本實作計畫提供了完整的 Fretting-Transformer Post-Processing 模組設計與實作指南，涵蓋：

1. ✅ **清晰的 Input/Output 定義**：基於論文 v3 token format
2. ✅ **兩種核心演算法**：Overlap Correction + Neighbor Search，附完整 pseudocode
3. ✅ **模組化架構**：8 個核心模組，職責分明，易於測試
4. ✅ **詳細實作步驟**：8 個 phases，共 16-24 天，含測試計畫
5. ✅ **Edge Cases 處理**：和弦、超出範圍、無效組合、錯誤 tokens
6. ✅ **Pipeline 整合**：與現有 JAMS/MIDI infrastructure 無縫整合
7. ✅ **使用範例**：從 basic usage 到 batch processing
8. ✅ **預期成果**：複製論文 Table 2，達到 100% pitch accuracy

**核心優勢**：
- 系統化的設計方法
- 完整的測試策略
- 與現有 codebase 高度整合
- 可擴展的架構（未來支援更多樂器、效果）

實作完成後，將能夠：
- 將模型預測的 tablature pitch accuracy 從 ~97% 提升至 **100%**
- 同時優化 playability（減少手部移動、弦切換）
- 提供易用的 API 供研究和生產環境使用

---

**文件版本記錄**：
- v1.0 (2025-12-04): 初版完成
