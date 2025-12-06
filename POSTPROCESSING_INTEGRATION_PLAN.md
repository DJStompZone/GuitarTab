# Post-Processing 整合到 Inference Pipeline 實作計劃

## 目標

將 `fretting_postprocessor/` 模組整合到 `inference.py` 推論流程中，讓使用者能透過簡單的配置啟用後處理功能，以提升音高準確度至 ~100%。

## 核心挑戰

1. **Token 格式轉換**：現有 vocab 使用 `NOTE_ON_60`（底線格式），但 fretting_postprocessor 期望 `NOTE_ON<60>`（角括號格式）
2. **批次處理**：需逐序列處理 tensor，因後處理器需完整上下文
3. **評估比較**：需計算並比較原始預測 vs. 後處理預測的指標

## 實作步驟

### 步驟 1：建立 Post-Processing 配置檔

**新增檔案：`configs/postprocessing.yaml`**

```yaml
# Post-processing 配置
enabled: false  # 預設關閉（保持向後相容）

# 後處理方法
method: "neighbor_search"  # 選項: "overlap", "neighbor_search"

# 吉他配置
guitar:
  num_strings: 6
  tuning: "standard"  # 選項: "standard", "drop_d", "half_step_down", "full_step_down"
  capo_fret: 0
  min_fret: 0
  max_fret: 24

# 評估選項
save_intermediate: true   # 儲存原始預測（後處理前）
verbose: true            # 顯示比較表格
```

**修改檔案：`configs/inference.yaml`**

在 `defaults:` 區塊加入：
```yaml
defaults:
  - config
  - postprocessing: postprocessing  # 載入 post-processing 配置
  - _self_
```

### 步驟 2：建立 Token 格式轉換橋接模組

**新增檔案：`src/postprocessing_bridge.py`**

核心功能：
1. **格式轉換**：`NOTE_ON_60` ↔ `NOTE_ON<60>`
2. **批次處理**：逐序列處理 tensor → tokens → post-process → tensor
3. **錯誤處理**：處理格式不符或序列長度變化

關鍵方法：
- `ids_to_token_strings(ids, vocab)`: Tensor IDs → Token 字串列表（後處理器格式）
- `token_strings_to_ids(tokens, vocab)`: Token 字串列表 → Tensor IDs
- `process_batch(input_ids, predictions, method)`: 批次後處理主函數

Token 格式對應：
```python
# Dataset 格式 → Post-processor 格式
"NOTE_ON_60"      → "NOTE_ON<60>"
"NOTE_OFF_64"     → "NOTE_OFF<64>"
"TIME_SHIFT_480"  → "TIME_SHIFT<480>"
"TAB_3_5"         → "TAB<3,5>"
```

**可重用現有程式碼**：
- 參考 `notebooks/demo_clean copy.ipynb` 中的 `ids_to_events()` 函數進行 token 解析
- Token 字串解析邏輯類似，但需要額外的格式轉換步驟

實作考量：
- 使用正則表達式解析和轉換 token 字串
- 處理 PAD/BOS/EOS 等特殊 tokens
- 維持批次中的序列長度一致（padding）
- 可選：將後處理結果轉換為 Event 物件以便視覺化（使用現有的 `render_as_tablature()`）

### 步驟 3：擴充 Metrics 模組

**修改檔案：`src/metrics.py`**

新增 dataclass：
```python
@dataclass
class PostProcessingMetrics:
    """後處理比較指標"""
    # 原始模型指標
    raw_token_accuracy: float
    raw_pitch_accuracy: float
    raw_tab_accuracy: float

    # 後處理指標
    post_token_accuracy: float
    post_pitch_accuracy: float
    post_tab_accuracy: float

    # 改善幅度
    token_improvement: float
    pitch_improvement: float
    tab_improvement: float

    # 統計
    total_tokens: int
    total_notes: int
    method: str
```

新增函數：
```python
def compute_postprocessing_metrics(
    raw_predictions: torch.Tensor,
    postprocessed_predictions: torch.Tensor,
    targets: torch.Tensor,
    output_vocab,
    method: str,
    pad_id: int = 0
) -> PostProcessingMetrics
```

### 步驟 4：整合到 Inference Pipeline

**修改檔案：`inference.py`**

#### 4.1 新增 imports
```python
from src.postprocessing_bridge import PostProcessingBridge
from src.metrics import compute_postprocessing_metrics, PostProcessingMetrics
from fretting_postprocessor import GuitarConfig
from fretting_postprocessor.config import STANDARD_TUNING, DROP_D_TUNING, ...
```

#### 4.2 新增 helper 函數
```python
def create_guitar_config(cfg) -> GuitarConfig:
    """從 Hydra 配置建立 GuitarConfig"""
    tuning_map = {
        'standard': STANDARD_TUNING,
        'drop_d': DROP_D_TUNING,
        'half_step_down': HALF_STEP_DOWN,
        'full_step_down': FULL_STEP_DOWN,
    }
    tuning = tuning_map.get(cfg.postprocessing.guitar.tuning, STANDARD_TUNING)

    return GuitarConfig(
        num_strings=cfg.postprocessing.guitar.num_strings,
        tuning=tuning,
        capo_fret=cfg.postprocessing.guitar.capo_fret,
        min_fret=cfg.postprocessing.guitar.min_fret,
        max_fret=cfg.postprocessing.guitar.max_fret
    )
```

#### 4.3 修改 main() 函數

在 `generate_and_compute_accuracy()` 之後新增：

```python
# 執行推論（現有代碼）
metrics, (input_ids, targets, predictions) = generate_and_compute_accuracy(...)

# === 新增：後處理整合 ===
if cfg.postprocessing.enabled:
    print("\n" + "=" * 80)
    print(f"Applying Post-Processing ({cfg.postprocessing.method})")
    print("=" * 80)

    # 初始化橋接器
    guitar_config = create_guitar_config(cfg)
    bridge = PostProcessingBridge(
        dataset.input_vocab,
        dataset.output_vocab,
        guitar_config
    )

    # 批次後處理
    postprocessed_predictions = bridge.process_batch(
        input_ids=input_ids,
        predictions=predictions,
        method=cfg.postprocessing.method
    )

    # 計算比較指標
    pp_metrics = compute_postprocessing_metrics(
        raw_predictions=predictions,
        postprocessed_predictions=postprocessed_predictions,
        targets=targets,
        output_vocab=dataset.output_vocab,
        method=cfg.postprocessing.method
    )

    # 顯示結果
    if cfg.postprocessing.verbose:
        print(f"\n後處理結果:")
        print(f"  原始模型 - Pitch: {pp_metrics.raw_pitch_accuracy:.2%}, Tab: {pp_metrics.raw_tab_accuracy:.2%}")
        print(f"  後處理後 - Pitch: {pp_metrics.post_pitch_accuracy:.2%}, Tab: {pp_metrics.post_tab_accuracy:.2%}")
        print(f"  改善幅度 - Pitch: {pp_metrics.pitch_improvement:+.2%}, Tab: {pp_metrics.tab_improvement:+.2%}")

    # 儲存結果
    if cfg.postprocessing.save_intermediate:
        torch.save(predictions, cfg.output_dir / "raw_predictions.pt")

    torch.save(postprocessed_predictions, cfg.output_dir / "postprocessed_predictions.pt")

    import json
    with open(cfg.output_dir / "postprocessing_comparison.json", 'w') as f:
        json.dump({
            'method': pp_metrics.method,
            'raw_metrics': {
                'token_accuracy': pp_metrics.raw_token_accuracy,
                'pitch_accuracy': pp_metrics.raw_pitch_accuracy,
                'tab_accuracy': pp_metrics.raw_tab_accuracy,
            },
            'postprocessed_metrics': {
                'token_accuracy': pp_metrics.post_token_accuracy,
                'pitch_accuracy': pp_metrics.post_pitch_accuracy,
                'tab_accuracy': pp_metrics.post_tab_accuracy,
            },
            'improvements': {
                'token': pp_metrics.token_improvement,
                'pitch': pp_metrics.pitch_improvement,
                'tab': pp_metrics.tab_improvement,
            }
        }, f, indent=2)

    # 更新最終的 predictions 為後處理版本
    predictions = postprocessed_predictions

# 儲存最終預測（現有代碼，但現在可能是後處理後的結果）
torch.save(predictions, cfg.output_dir / "predictions.pt")
```

### 步驟 5：實作 PostProcessingBridge 詳細邏輯

**`src/postprocessing_bridge.py` 的核心實作**

```python
import re
import torch
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
from fretting_postprocessor import FrettingPostProcessor, GuitarConfig

class PostProcessingBridge:
    def __init__(self, input_vocab, output_vocab, guitar_config: GuitarConfig):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.processor = FrettingPostProcessor(guitar_config)

    def ids_to_token_strings(self, ids: torch.Tensor, vocab) -> List[str]:
        """轉換 tensor IDs 為 token 字串（後處理器格式）"""
        tokens = []
        for idx in ids:
            if idx == vocab.pad_id:
                continue  # 跳過 padding

            token_str = vocab.id_to_token[idx.item()]

            # 轉換格式：NOTE_ON_60 → NOTE_ON<60>
            if token_str.startswith(("NOTE_ON_", "NOTE_OFF_")):
                prefix, pitch = token_str.rsplit("_", 1)
                tokens.append(f"{prefix}<{pitch}>")
            elif token_str.startswith("TIME_SHIFT_"):
                _, shift = token_str.rsplit("_", 1)
                tokens.append(f"TIME_SHIFT<{shift}>")
            elif token_str.startswith("TAB_"):
                _, string, fret = token_str.split("_")
                tokens.append(f"TAB<{string},{fret}>")
            else:
                tokens.append(token_str)  # PAD, BOS, EOS, UNK

        return tokens

    def token_strings_to_ids(self, tokens: List[str], vocab) -> torch.Tensor:
        """轉換 token 字串為 tensor IDs"""
        ids = []
        for token in tokens:
            # 轉換格式：NOTE_ON<60> → NOTE_ON_60
            if match := re.match(r'(NOTE_ON|NOTE_OFF)<(\d+)>', token):
                token_str = f"{match.group(1)}_{match.group(2)}"
            elif match := re.match(r'TIME_SHIFT<(\d+)>', token):
                token_str = f"TIME_SHIFT_{match.group(1)}"
            elif match := re.match(r'TAB<(\d+),(\d+)>', token):
                token_str = f"TAB_{match.group(1)}_{match.group(2)}"
            else:
                token_str = token

            token_id = vocab.token_to_id.get(token_str, vocab.unk_id)
            ids.append(token_id)

        return torch.tensor(ids, dtype=torch.long)

    def process_batch(self, input_ids: torch.Tensor, predictions: torch.Tensor,
                     method: str = 'neighbor_search') -> torch.Tensor:
        """批次後處理"""
        B, L = predictions.shape
        postprocessed_batch = []

        for i in range(B):
            # 轉換為 token 字串
            input_tokens = self.ids_to_token_strings(input_ids[i], self.input_vocab)
            pred_tokens = self.ids_to_token_strings(predictions[i], self.output_vocab)

            # 後處理
            try:
                corrected_tokens = self.processor.process_tokens(
                    model_output_tokens=pred_tokens,
                    input_note_tokens=input_tokens,
                    method=method
                )
            except Exception as e:
                print(f"Warning: Post-processing failed for sequence {i}: {e}")
                corrected_tokens = pred_tokens  # Fallback

            # 轉回 IDs
            corrected_ids = self.token_strings_to_ids(corrected_tokens, self.output_vocab)
            postprocessed_batch.append(corrected_ids)

        # Pad to uniform length
        postprocessed_predictions = pad_sequence(
            postprocessed_batch,
            batch_first=True,
            padding_value=self.output_vocab.pad_id
        )

        # 確保與原始 predictions 相同形狀
        if postprocessed_predictions.shape[1] < L:
            padding = torch.full(
                (B, L - postprocessed_predictions.shape[1]),
                self.output_vocab.pad_id,
                dtype=postprocessed_predictions.dtype
            )
            postprocessed_predictions = torch.cat([postprocessed_predictions, padding], dim=1)
        elif postprocessed_predictions.shape[1] > L:
            postprocessed_predictions = postprocessed_predictions[:, :L]

        return postprocessed_predictions
```

### 步驟 6：輸出檔案結構

啟用後處理後，輸出目錄將包含：

```
outputs/{experiment_name}/
├── input_ids.pt                      # 模型輸入
├── targets.pt                        # Ground truth
├── predictions.pt                    # 最終預測（已後處理）
├── raw_predictions.pt                # 原始模型輸出（如果 save_intermediate=true）
├── postprocessed_predictions.pt      # 後處理輸出
├── postprocessing_comparison.json    # 比較指標
└── config.yaml                       # Hydra 配置快照
```

### 步驟 7：使用方式

```bash
# 基本用法：啟用後處理
python inference.py postprocessing.enabled=true

# 使用 overlap correction 方法
python inference.py postprocessing.enabled=true postprocessing.method=overlap

# 自訂吉他配置
python inference.py postprocessing.enabled=true \
    postprocessing.guitar.tuning=drop_d \
    postprocessing.guitar.capo_fret=2

# 向後相容：預設行為不變
python inference.py  # postprocessing.enabled=false
```

## 關鍵檔案

1. **`configs/postprocessing.yaml`** (新增) - 後處理配置
2. **`configs/inference.yaml`** (修改) - 加入 postprocessing defaults
3. **`src/postprocessing_bridge.py`** (新增) - Token 格式轉換與批次處理
4. **`src/metrics.py`** (擴充) - 新增 PostProcessingMetrics
5. **`inference.py`** (修改) - 整合後處理流程

## 測試計劃

1. **單元測試**：測試 token 格式轉換正確性
2. **整合測試**：在小型資料集上測試完整流程
3. **端到端測試**：驗證音高準確度提升至 ~100%
4. **向後相容測試**：確認預設行為不變

## 預期成果

- ✅ 使用者可透過 `postprocessing.enabled=true` 輕鬆啟用後處理
- ✅ 音高準確度從 ~97% 提升至 ~100%
- ✅ 保持向後相容性（預設關閉）
- ✅ 清晰的比較指標輸出
- ✅ 易於維護和擴充

## 可選擴充功能

### 視覺化後處理結果
參考 `notebooks/demo_clean copy.ipynb` 和 `notebooks/demo_full_pipeline.ipynb` 中的視覺化功能，可以在 `inference.py` 或新的 notebook 中加入：

1. **轉換函數**（已存在於 notebooks）：
   - `ids_to_events()`: 將 token IDs 轉為 Event 物件

2. **視覺化函數**（已存在於 `src/visualization.py`）：
   - `render_as_tablature()`: 渲染為吉他指法譜
   - `render_as_notes()`: 渲染為音符記譜法

3. **整合方式**：
   - 在 `inference.py` 中加入視覺化選項
   - 或建立新的 `notebooks/demo_with_postprocessing.ipynb` 示範後處理效果
   - 可視覺化比較：原始預測 vs. 後處理預測 vs. ground truth

這可以幫助使用者直觀地看到後處理對指法準確度的改善效果。
