# Post-Processing Integration Guide

## 概述

本指南說明如何使用 Fretting-Transformer 的後處理功能來提升吉他指法轉錄的音高準確度至 ~100%。

後處理模組基於 Fretting-Transformer 論文（arXiv:2506.14223v1）實作，提供兩種後處理方法：
- **Overlap Correction**: 透過時間窗口匹配修正音高錯誤（~99.92% 準確度）
- **Neighbor Search**: 完整流程，優化指法位置（~100% 準確度）

## 快速開始

### 1. 基本使用

啟用後處理只需在推論時加入一個參數：

```bash
python inference.py postprocessing.enabled=true
```

這將使用預設設定（neighbor_search 方法 + 標準調音）進行後處理。

### 2. 選擇後處理方法

```bash
# 使用 Overlap Correction（較快，~99.92% 準確度）
python inference.py postprocessing.enabled=true postprocessing.method=overlap

# 使用 Neighbor Search（完整流程，~100% 準確度）
python inference.py postprocessing.enabled=true postprocessing.method=neighbor_search
```

### 3. 自訂吉他配置

```bash
# Drop D 調音
python inference.py postprocessing.enabled=true \
    postprocessing.guitar.tuning=drop_d

# 使用 Capo 在第 2 格
python inference.py postprocessing.enabled=true \
    postprocessing.guitar.capo_fret=2

# 組合使用
python inference.py postprocessing.enabled=true \
    postprocessing.guitar.tuning=drop_d \
    postprocessing.guitar.capo_fret=2 \
    postprocessing.method=neighbor_search
```

## 配置選項

### 完整配置檔：`configs/postprocessing.yaml`

```yaml
# 啟用/停用後處理（預設：false 保持向後相容）
enabled: false

# 後處理方法
# 選項：
#   - "overlap": 僅 Overlap correction (~99.92% 音高準確度)
#   - "neighbor_search": 完整流程 (~100% 音高準確度)
method: "neighbor_search"

# 吉他配置
guitar:
  num_strings: 6

  # 吉他調音預設值
  # 選項: "standard", "drop_d", "half_step_down", "full_step_down"
  tuning: "standard"

  # Capo 位置（格數，0 = 無 capo）
  capo_fret: 0

  # 品格範圍
  min_fret: 0
  max_fret: 24

# 輸出和評估選項
save_intermediate: true  # 儲存後處理前的原始預測
verbose: true            # 顯示比較表格
```

### 透過命令列覆寫配置

所有配置選項都可以透過 Hydra 的命令列參數覆寫：

```bash
python inference.py \
    postprocessing.enabled=true \
    postprocessing.method=overlap \
    postprocessing.guitar.num_strings=6 \
    postprocessing.guitar.tuning=standard \
    postprocessing.guitar.capo_fret=0 \
    postprocessing.save_intermediate=true \
    postprocessing.verbose=true
```

## 輸出檔案

啟用後處理後，輸出目錄（`outputs/{experiment_name}/`）將包含：

```
outputs/{experiment_name}/
├── input_ids.pt                      # 模型輸入
├── targets.pt                        # Ground truth
├── predictions.pt                    # 最終預測（已後處理）
├── raw_predictions.pt                # 原始模型輸出（如果 save_intermediate=true）
├── postprocessed_predictions.pt      # 後處理輸出
├── postprocessing_comparison.json    # 比較指標（詳細說明見下）
└── config.yaml                       # Hydra 配置快照
```

### 比較指標檔案格式

`postprocessing_comparison.json` 包含詳細的比較指標：

```json
{
  "method": "neighbor_search",
  "raw_metrics": {
    "token_accuracy": 0.8523,
    "pitch_accuracy": 0.9723,
    "tab_accuracy": 0.6856
  },
  "postprocessed_metrics": {
    "token_accuracy": 0.8634,
    "pitch_accuracy": 1.0000,
    "tab_accuracy": 0.7219
  },
  "improvements": {
    "token": 0.0111,
    "pitch": 0.0277,
    "tab": 0.0363
  },
  "counts": {
    "total_tokens": 15234,
    "total_notes": 3456
  }
}
```

## 運作原理

### Token 格式轉換

後處理模組使用不同的 token 格式，因此需要進行轉換：

**Dataset 格式** → **Post-processor 格式**
- `NOTE_ON_60` → `NOTE_ON<60>`
- `NOTE_OFF_64` → `NOTE_OFF<64>`
- `TIME_SHIFT_480` → `TIME_SHIFT<480>`
- `TAB_3_5` → `TAB<3,5>`

這個轉換由 `PostProcessingBridge` 類別自動處理。

### 處理流程

1. **模型推論**：產生原始預測
2. **格式轉換**：將 tensor IDs 轉換為 token 字串
3. **後處理**：
   - Overlap Correction: 修正音高錯誤
   - Neighbor Search: 優化指法位置
4. **格式轉回**：將處理後的 tokens 轉回 tensor IDs
5. **評估比較**：計算原始 vs. 後處理的指標
6. **儲存結果**：儲存比較指標和預測結果

### 批次處理

後處理會逐序列處理整個批次：
- 每個序列獨立處理（保留完整上下文）
- 處理後自動進行 padding 以保持批次形狀一致
- 支援錯誤處理和 fallback 機制

## 支援的吉他調音

### 預設調音

- `standard`: 標準調音 (E-A-D-G-B-E)
- `drop_d`: Drop D (D-A-D-G-B-E)
- `half_step_down`: 降半音 (Eb-Ab-Db-Gb-Bb-Eb)
- `full_step_down`: 降全音 (D-G-C-F-A-D)

### 自訂調音

如需使用自訂調音，可以修改 `configs/postprocessing.yaml` 並加入 `custom_tuning` 選項（未來功能）。

## 進階使用

### 在程式碼中使用

您也可以在自己的 Python 程式碼中直接使用後處理功能：

```python
from src.postprocessing_bridge import PostProcessingBridge
from fretting_postprocessor import GuitarConfig
from fretting_postprocessor.config import STANDARD_TUNING

# 建立吉他配置
guitar_config = GuitarConfig(
    num_strings=6,
    tuning=STANDARD_TUNING,
    capo_fret=0,
    min_fret=0,
    max_fret=24
)

# 初始化橋接器
bridge = PostProcessingBridge(
    input_vocab=dataset.input_vocab,
    output_vocab=dataset.output_vocab,
    guitar_config=guitar_config
)

# 批次後處理
postprocessed_predictions = bridge.process_batch(
    input_ids=input_ids,
    predictions=predictions,
    method='neighbor_search'
)
```

### 計算比較指標

```python
from src.metrics import compute_postprocessing_metrics

pp_metrics = compute_postprocessing_metrics(
    raw_predictions=raw_predictions,
    postprocessed_predictions=postprocessed_predictions,
    targets=targets,
    output_vocab=output_vocab,
    method='neighbor_search'
)

print(f"Pitch improvement: {pp_metrics.pitch_improvement:+.2%}")
print(f"Tab improvement: {pp_metrics.tab_improvement:+.2%}")
```

## 效能考量

### 計算成本

- **Overlap Correction**: 輕量級，在時間窗口內匹配音符
- **Neighbor Search**: 較重，需要探索多個指法位置

### 建議

- 對於快速評估，使用 `method=overlap`
- 對於最佳準確度，使用 `method=neighbor_search`
- 大型資料集上可能需要較長處理時間（視序列長度而定）

### 記憶體使用

後處理會產生額外的中間結果：
- 設定 `save_intermediate=false` 可減少磁碟使用
- 批次大小不影響記憶體（逐序列處理）

## 疑難排解

### 常見問題

**Q: 後處理失敗並顯示警告訊息**
```
Warning: Post-processing failed for sequence X: ...
```

A: 這通常是因為 token 格式轉換問題或無效的音符序列。系統會自動 fallback 到原始預測，不會中斷推論流程。

**Q: 音高準確度沒有達到 100%**

A: 確認：
1. 使用 `method=neighbor_search`（而非 `overlap`）
2. 吉他配置（tuning、capo）符合資料集
3. 模型預測品質（原始音高準確度應 >90%）

**Q: 向後相容性**

A: 預設 `enabled=false`，不會影響現有流程。只有明確啟用時才執行後處理。

### Debug 模式

啟用詳細輸出以查看處理細節：

```bash
python inference.py \
    postprocessing.enabled=true \
    postprocessing.verbose=true
```

這會顯示：
- 原始模型指標
- 後處理後指標
- 改善幅度

## 參考資料

- **論文**: Fretting-Transformer (arXiv:2506.14223v1)
- **後處理模組**: `fretting_postprocessor/` 目錄
- **實作計劃**: `POSTPROCESSING_INTEGRATION_PLAN.md`
- **原始碼**:
  - `src/postprocessing_bridge.py` - 格式轉換與批次處理
  - `src/metrics.py` - 評估指標
  - `inference.py` - 整合流程

## 版本歷史

- **v1.0** (2025-12-06): 初始整合
  - 支援 overlap correction 和 neighbor search
  - 批次處理功能
  - 完整的比較指標
  - 向後相容設計

## 授權

本功能整合到 GuitarTab 專案中，遵循專案原有授權。
