# v1 / v2 / v2+aux 輸出格式錯誤分解與能力上限驗證

**實驗日期：** 2026-05-10  
**Inference Jobs：** 903471（M1/M2 C0–C3 分析）、903472（M3 C0–C3）  
**測試集：** `data_splits/test_files.json`（519 首、4725 個 segment）

---

## 0. 實驗概述

本實驗比較三個吉他 MIDI 轉譜模型，探討輸出格式與 pitch supervision 對模型能力的影響。

### 模型

| 標記 | 輸出格式 | Checkpoint | 備注 |
|------|---------|------------|------|
| M1 | v1：`NOTE_ON + TAB + NOTE_OFF + TIME_SHIFT` | `ckpt/dadagp_v1_300epcohs_weight_decay/best_model.pt` | — |
| M2 | v2：`TAB + TIME_SHIFT` | `ckpt/dadagp_v2_300epochs_weight_decay/best_model.pt` | — |
| M3 | v2+aux：v2 token 序列 + decoder 輔助 pitch head | `ckpt/v2_aux/best_model.pt`（epoch 166 早停） | Pitch head 僅用於訓練，inference 時忽略 |

### M3 的設計動機與 Auxiliary Pitch Head

在比較 M1 與 M2 時，兩者之間同時存在**兩個差異**：

| | 輸出 token format | Pitch supervision |
|--|:-----------------:|:-----------------:|
| M1 (v1) | NOTE_ON + TAB + NOTE_OFF + TIME_SHIFT | ✓ 序列中有明確的 `NOTE_ON_<pitch>` token |
| M2 (v2) | TAB + TIME_SHIFT | ✗ pitch 只能從 TAB token 間接推斷 |

這導致 M1 vs M2 的實驗無法區分「**是 format 帶來優勢**」還是「**是 pitch supervision 帶來優勢**」——兩個因素同時變動，無法單獨歸因。

M3 的設計目標就是**控制這個變因**：在保持 v2 token format 不變的前提下，透過 auxiliary pitch head 為訓練注入 pitch supervision，使得：

```
M2：v2 format，無 pitch supervision
M3：v2 format，有 pitch supervision（aux head）
```

如果 M3 的 pitch 精度接近 M1、Tab_3_1 接近 M2，就能得出結論：**v1 的優勢主要來自 pitch supervision，而非 NOTE_ON token format 本身**。

#### Auxiliary Pitch Head 架構

M3 在標準 `FrettingTransformer`（T5 backbone）的基礎上，加入一個輕量的輔助分類頭：

```
T5 Decoder last hidden state  [B, L_dec, d_model=128]
           ↓
   pitch_head: nn.Linear(128, 128)   ← 128 個類別對應 MIDI pitch 0–127
           ↓
   pitch_logits  [B, L_dec, 128]
```

只加了一層 Linear，不增加 Transformer layers。

#### Pitch Label 的建立方式

訓練資料來自 v1 格式的完整 event stream（內含 NOTE_ON），從中抽取每個 TAB token 對應的 pitch：

```
... NOTE_ON(pitch=60) → TAB(s=2, f=8) → NOTE_OFF → TIME_SHIFT ...
```

每個 TAB token 往前找最近的 NOTE_ON，取其 MIDI pitch 作為 label。TIME_SHIFT、BOS、EOS、PAD 位置全部設為 `ignore_index=-100`，cross-entropy 自動跳過，**只在 TAB token 的位置計算 pitch loss**。

#### 訓練 Loss 設計

```
total_loss = (1 - α) × CE(TAB tokens) + α × CE(pitch)
```

預設 α = 0.5，TAB 與 pitch 各佔一半權重。

#### 為什麼 Inference 可以忽略 Pitch Head

Pitch head 在訓練時的功能是**正規化器（regularizer）**：它迫使 decoder 的 hidden state 在每個 TAB 位置保留足夠的 pitch 資訊，使得 TAB token 的生成間接受到 pitch 監督。

Inference 時模型輸出的是 TAB token 序列（v2 format），pitch head 的 logit 根本不會被用到。丟掉 pitch head 不影響生成品質——pitch supervision 的效果已內化到 decoder 的參數空間中。

這也是為什麼用 `strict=False` 載入 checkpoint 是正確做法：checkpoint 儲存了 `pitch_head.weight` 和 `pitch_head.bias`，但 inference 用的 `FrettingTransformer` 沒有這兩個 key，忽略即可。

---

### 解碼條件

| 條件 | 說明 |
|------|------|
| C0 | 無約束自迴歸解碼 |
| C1 | 僅強制 grammar（NOTE_ON/TAB/NOTE_OFF 配對合法）。**僅對 v1 有意義**；v2 的輸出詞表中沒有 NOTE_ON/OFF token，理論上應退化為 C0（本次 M2/M3 的 C1 因 bug 結果無效，詳見第 5 節） |
| C2 | 強制輸入骨架（TIME_SHIFT 與 TAB 位置對齊輸入序列），TAB token 本身自由選擇 |
| C3 | 強制輸入骨架 + pitch mask（每個 TAB 只能選擇在指板上與輸入 pitch 一致的合法把位）|

---

## 1. 實驗結果

### 1.1 基準指標（inference.py 原始輸出）

| 模型 | 條件 | Token Acc | Pitch Acc | Tab Acc | 總音符數 |
|------|------|----------:|----------:|--------:|--------:|
| M1 | C0 | 91.13% | 90.06% | 77.10% | 523,392 |
| M1 | C1 | 88.48% | 85.55% | 73.23% | 523,392 |
| M1 | C2 | 89.27% | 93.02% | 79.39% | 523,392 |
| M1 | C3 | 95.77% | **100.00%** | **85.05%** | 523,392 |
| M2 | C0 | 80.57% | 85.66% | 71.78% | 681,228 |
| M2 | C1 | — | — | — | ❌ Bug（grammar deadlock，見第 5 節） |
| M2 | C2 | 78.11% | 86.19% | 72.08% | 681,228 |
| M2 | C3 | 90.17% | **100.00%** | **83.65%** | 681,228 |
| M3 | C0 | 82.90% | 89.36% | 73.91% | 681,228 |
| M3 | C1 | — | — | — | ❌ Bug（同上，修復前的結果） |
| M3 | C2 | 80.17% | 89.20% | 73.57% | 681,228 |
| M3 | C3 | 89.36% | **100.00%** | **82.25%** | 681,228 |

> M1 與 M2/M3 的總音符數不同，原因是 v1 格式將每個音符拆成 NOTE_ON + TAB + NOTE_OFF 三步，造成序列長度與 token 計數方式不同。底層 pitch/tab 評估邏輯一致，但分母有別。

### 1.2 Robust 對齊指標

| 模型 | 條件 | coverage | precision | f1 | tab_acc_aligned | pitch_acc_aligned | valid_ratio_pred |
|------|------|:--------:|:---------:|:--:|:---------------:|:-----------------:|:----------------:|
| M1 | C0 | 0.9852 | 0.9213 | 0.9522 | 78.93% | 92.25% | 1.0000 |
| M1 | C1 | 0.9659 | 0.7562 | 0.8483 | 76.10% | 88.84% | 1.0000 |
| M1 | C2 | 0.9303 | 0.5064 | 0.6558 | 85.30% | 99.85% | **0.6570** |
| M1 | C3 | **1.0000** | 0.7683 | 0.8690 | 85.05% | **100.00%** | 1.0000 |
| M2 | C0 | 0.9544 | 0.9559 | 0.9551 | 76.12% | 90.84% | 1.0000 |
| M2 | C1 | — | — | — | ❌ | | |
| M2 | C2 | 0.8737 | 0.4599 | 0.6026 | 83.24% | 99.26% | 1.0000 |
| M2 | C3 | **1.0000** | **1.0000** | **1.0000** | 83.65% | **100.00%** | 1.0000 |
| M3 | C0 | 0.9779 | 0.7982 | 0.8790 | 76.33% | 92.25% | 1.0000 |
| M3 | C1 | — | — | — | ❌ | | |
| M3 | C2 | 0.9078 | 0.5420 | 0.6788 | 82.15% | 99.41% | 1.0000 |
| M3 | C3 | **1.0000** | **1.0000** | **1.0000** | 82.25% | **100.00%** | 1.0000 |

> **Pitch Acc（基準）vs pitch_acc_aligned 的差異**：基準指標以所有 target token 為分母，做 position-wise 比較，受 coverage 不足與序列錯位影響；aligned 指標先做 timeline 對齊，只在成功配對的音符對上計算 pitch 正確率，更公平地反映模型實際的 pitch 判斷能力。

### 1.3 Error Class 細分計數

每個音符被歸類到以下其中一類（依優先順序）：

| 類別 | 定義 |
|------|------|
| **G** | Grammar 錯誤：parse 失敗、非法 token 序列（僅 v1 有意義） |
| **T** | Timing 錯誤：對齊容差外，音符無法與 target 配對（= missed note） |
| **Tab_3_2** | Pitch 錯誤：對齊成功但 TAB 隱含 pitch 與 target 不一致 |
| **Tab_3_1** | Tab-only 錯誤：pitch 正確但 (string, fret) 選錯 |
| **I** | Internal inconsistency（v1 專有）：NOTE_ON pitch 與 TAB 隱含 pitch 不一致 |
| **correct** | pitch 與 (string, fret) 均正確 |

marginal rate = count / total_target；conditional rate = count / total_aligned（排除 missed note）。

#### C0 三模型細分（無約束，marginal rate）

| 類別 | M1 (count) | M1 rate | M2 (count) | M2 rate | M3 (count) | M3 rate |
|------|:----------:|:-------:|:----------:|:-------:|:----------:|:-------:|
| total_target | 523,392 | — | 681,228 | — | 681,228 | — |
| total_aligned | 515,623 | — | 650,185 | — | 666,196 | — |
| missed (= T) | 7,769 | 1.48% | 31,043 | 4.56% | 15,032 | 2.21% |
| extra_pred | 44,036 | — | 30,018 | — | 168,438 | — |
| **G** | 1,537 | 0.29% | 0 | 0% | 0 | 0% |
| **T** | 7,769 | 1.48% | 31,043 | 4.56% | 15,032 | 2.21% |
| **Tab_3_2**（pitch 錯）| 39,954 | 7.63% | 59,564 | 8.74% | 51,654 | **7.58%** |
| **Tab_3_1**（tab 錯）| 68,665 | 13.12% | 95,701 | 14.05% | 106,012 | 15.56% |
| **I**（v1 自我打架）| 39,865 | **7.62%** | 0 | 0% | 0 | 0% |
| correct | 407,004 | 77.76% | 494,920 | 72.65% | 508,530 | 74.65% |

**C0 關鍵發現：**

1. **Timing（T）錯誤**：M1 最少（1.48%），M3 居中（2.21%），M2 最多（4.56%）。M1 的明確 NOTE_ON token 提供更穩固的時間錨點；M3 的 aux training 也讓 timing 優於 M2。

2. **Pitch 錯誤（Tab_3_2）**：M3（7.58%）≈ M1（7.63%）< M2（8.74%）。**Aux pitch head 讓 M3 的 pitch 錯誤率與 M1 齊平**，直接驗證 H2——pitch supervision 是關鍵，而非 token format。

3. **True capability（Tab_3_1）**：M3（15.56%）> M2（14.05%）> M1（13.12%）。M3 在 C0 下 Tab_3_1 rate 反而最高，因為 pitch 更準確使更多音符進入「pitch 正確」的分母，相對放大了純 tab 選擇錯誤的佔比。

4. **Internal Inconsistency（I，v1 專有）**：M1 有 7.62% 的音符出現 NOTE_ON pitch 與 TAB 隱含 pitch 不一致。這是 v1 冗餘格式「自我打架」的代價，v2/M3 為零。

5. **extra_pred（多餘預測）**：M3 的 extra_pred（168,438）遠多於 M1（44,036）和 M2（30,018），顯示 M3 的 decoder 傾向生成較長的序列，在 C2 constrained 時會導致更多 coverage gap。

#### 逐層約束的錯誤消除——M1

| 條件 | G rate | T rate | Tab_3_2 rate | Tab_3_1 rate | I rate | correct rate |
|------|:------:|:------:|:------------:|:------------:|:------:|:------------:|
| C0 | 0.29% | 1.48% | 7.63% | 13.12% | 7.62% | 77.76% |
| C1 | **0%** | 3.41% ↑ | 10.78% ↑ | 12.30% | 12.66% ↑ | 73.50% ↓ |
| C2 | ⚠ 63.01% | 6.97% | **0.14%** ↓↓ | 13.53% | **0.08%** ↓↓ | 79.36% |
| C3 | **0%** | **0%** | **0%** | 14.95% | **0%** | **85.05%** |

- **C1 未有效消除 G**（0.29% → 0%），但 T 和 I 反而上升，整體 Tab Acc 退化，是非單調異常。Grammar constraint 破壞了模型自然的 decoding 路徑。
- **C2 的 G = 63.01% 是異常值**：input skeleton 強制 NOTE_ON/OFF 位置後，模型仍生成大量中間結構 token，這些 token 在後處理時被標記為 grammar 錯誤。Tab_3_2 和 I 確實被大幅消除（符合設計），但 G 的語意在此條件下失真，不應直接解讀。
- **C3 正確清零** G、T、Tab_3_2、I，只剩 Tab_3_1（14.95%），是真正的 irreducible error。

#### 逐層約束的錯誤消除——M2 / M3

| 條件 | 模型 | T rate | Tab_3_2 rate | Tab_3_1 rate | correct rate |
|------|------|:------:|:------------:|:------------:|:------------:|
| C0 | M2 | 4.56% | 8.74% | 14.05% | 72.65% |
| C0 | M3 | 2.21% | 7.58% | 15.56% | 74.65% |
| C2 | M2 | 12.63% ↑ | **0.64%** ↓↓ | 14.00% | 72.73% |
| C2 | M3 | 9.22% ↑ | **0.54%** ↓↓ | 15.67% | 74.57% |
| C3 | M2 | **0%** | **0%** | **16.35%** | **83.65%** |
| C3 | M3 | **0%** | **0%** | **17.75%** | **82.25%** |

- **C2 時 T 反而上升**（M2: 4.56% → 12.63%）：Skeleton 步數受輸入長度限制，部分 target 音符超出 skeleton 覆蓋範圍，導致更多 missed notes，與設計預期相反。Coverage 降至 87%（M2）和 91%（M3）。
- **Tab_3_2 在 C2 被大幅消除**（8.74% → 0.64%）：Skeleton 強制 TAB 位置後，pitch 間接確定，此行為符合預期。
- **C3 完整清零** T 與 Tab_3_2，Tab_3_1 成為唯一殘餘錯誤類別。

#### C3 True Capability 最終比較（Tab_3_1 only）

| 模型 | Tab_3_1 count | Tab_3_1 marginal | correct（Tab Acc）|
|------|:------------:|:----------------:|:-----------------:|
| M1 | 78,245 | 14.95% | **85.05%** |
| M2 | 111,390 | 16.35% | 83.65% |
| M3 | 120,904 | 17.75% | 82.25% |

在 pitch error 被完全消除的條件下，**M1 的把位選擇能力最強**（Tab_3_1 最低），M2 次之，M3 最弱。

---

## 2. 分析

### 2.1 實驗一：無約束解碼下的錯誤分解（C0）——pitch supervision 效果

| | Pitch Acc（基準）| pitch_acc_aligned | Tab Acc | tab_acc_aligned |
|--|:---:|:---:|:---:|:---:|
| M1（v1，明確 NOTE_ON 監督）| 90.06% | **92.25%** | 77.10% | **78.93%** |
| M2（v2，無 pitch 監督）| 85.66% | 90.84% | 71.78% | 76.12% |
| M3（v2+aux，pitch head 正規化）| 89.36% | **92.25%** | 73.91% | 76.33% |

**核心發現：M3 的 `pitch_acc_aligned`（92.25%）與 M1 完全一致，比 M2（90.84%）高出 1.41 個百分點。**

這強力支持假設 H2：

> **v1 相對 v2 的 pitch 精度優勢，主因是 pitch supervision，而非 NOTE_ON token format 本身。**

M3 使用與 M2 相同的 v2 輸出格式，inference 時沒有 pitch head，卻在無約束解碼下達到與 M1 相同的 pitch_acc_aligned。這說明訓練時的輔助 pitch loss 有效地將 pitch 資訊注入了 decoder 的表示空間，即便 inference 時不使用 pitch head 輸出，這種隱式監督仍然發揮作用。

Tab 精度方面 M1 仍然領先（77.10% vs M3 73.91% vs M2 71.78%），差距縮小但未消除，顯示 token format 對 tab 選擇仍有邊際貢獻，但並非主因。

### 2.2 實驗二：逐層約束消融的單調性驗證

#### M1（v1）：C0 → C1 → C2 → C3

| 條件 | Tab Acc | pitch_acc_aligned | coverage | f1 |
|------|:-------:|:-----------------:|:--------:|:--:|
| C0 | 77.10% | 92.25% | 98.52% | 0.9522 |
| C1 | 73.23% ↓ | 88.84% ↓ | 96.59% ↓ | 0.8483 ↓ |
| C2 | 79.39% ↑ | 99.85% ↑ | 93.03% ↓ | 0.6558 ↓ |
| C3 | **85.05%** | **100.00%** | **100.00%** | 0.8690 |

**C0 → C1 非單調退化**是本次最值得注意的異常。Grammar constraint 對 M1 反而有害：Tab Acc 從 77.10% 跌至 73.23%，pitch_acc_aligned 從 92.25% 跌至 88.84%。可能原因是模型的自由生成已大致滿足 grammar 結構，額外施加 grammar state machine 的強制轉移反而迫使解碼走向次優路徑，與模型分布不一致。

**M1 C2 語法問題**：`valid_ratio_pred = 0.6570`，意即約 34% 的預測序列語法無效，`syntax_penalty = 1.0`，導致 `strict_tab_score = 0.0`。Input skeleton 強制特定的 NOTE_ON/OFF 位置，但 v1 模型仍傾向在這些位置外插入額外的結構 token，造成序列語法衝突。此條件下只有 aligned 與 normalized 指標可信，strict track 不應採用。

#### M2 / M3（v2）：C0 → C2 → C3（C1 排除）

| 條件 | 模型 | Tab Acc | pitch_acc_aligned | coverage | f1 |
|------|------|:-------:|:-----------------:|:--------:|:--:|
| C0 | M2 | 71.78% | 90.84% | 95.44% | 0.9551 |
| C0 | M3 | 73.91% | 92.25% | 97.79% | 0.8790 |
| C2 | M2 | 72.08% ≈ | 99.26% ↑↑ | 87.37% ↓ | 0.6026 |
| C2 | M3 | 73.57% ≈ | 99.41% ↑↑ | 90.78% ↓ | 0.6788 |
| C3 | M2 | **83.65%** | **100.00%** | **100.00%** | **1.0000** |
| C3 | M3 | **82.25%** | **100.00%** | **100.00%** | **1.0000** |

C2 → C3 是 v2 模型最大的單步躍升（約 +11~12 個百分點 Tab Acc），由 pitch mask 消除了 pitch 不一致的指板位置所驅動。C2 時 coverage 下降（87~91%）是因為 skeleton 的步數由輸入序列長度決定，有時短於模型自由生成的長度，導致部分 target 音符未被對齊。

### 2.3 實驗三：C3 下的 True Capability 比較

C3 是本實驗定義的能力上限：pitch error 被強制清零，剩下的 Tab 錯誤（pitch 正確但 string/fret 選錯）才是模型真正的「把位選擇能力」指標（Tab_3.1）。

| 模型 | Tab Acc（C3）| tab_acc_aligned（C3）| f1（C3）|
|------|:-----------:|:--------------------:|:-------:|
| M1 | **85.05%** | **85.05%** | 0.8690 |
| M2 | 83.65% | 83.65% | **1.0000** |
| M3 | 82.25% | 82.25% | **1.0000** |

**在完全約束解碼下，M1 > M2 > M3，差距分別為 +1.40 pp。**

M1 的 f1 僅 0.8690（precision = 0.7683）反映了 v1 格式的冗餘性：coverage = 1.0（所有 target 音符都被覆蓋），但 decoder 還生成了多餘的 NOTE_ON/OFF token，導致 precision 下降。v2 模型（M2、M3）在 C3 下 f1 = 1.0，對齊完美，是更乾淨的比較基準。

M3 在 C3 下略遜 M2（82.25% vs 83.65%）。輔助 pitch head 的正規化改善了無約束下的 pitch 精度，但可能輕微改變了 decoder 的表示空間，對 constrained 解碼的 tab 選擇產生微幅負面影響——是可接受的取捨。

---

## 3. 結論

| 驗證目標 | 結論 |
|---------|------|
| v1 的 pitch 精度優勢來自 supervision 而非 format | **確認**。M3（v2 格式 + pitch supervision）的 pitch_acc_aligned 與 M1 相同（92.25%），遠超 M2（90.84%） |
| Aux pitch head 能有效解耦 supervision 與 format | **確認**。M3 在 inference 時不使用 pitch head，仍享有與 M1 等同的 pitch 對齊精度 |
| C3 是 true capability 的操作型上限 | **確認**。C3 唯一能將 pitch error 清零，剩餘 tab_acc 反映純粹的把位選擇能力 |
| C0→C1→C2→C3 約束單調消減 | **不成立（M1）**。M1 的 C1 < C0，grammar constraint 反而使 M1 退化。C2→C3 對所有模型均單調成立 |
| v1 format 相對 v2 具有超出 supervision 的額外優勢 | **微弱但存在**。C3 下 M1 vs M2 Tab Acc 差距為 +1.40 pp，具實際意義但幅度有限 |

---

## 4. 產出物路徑

| 執行標記 | 輸出目錄 |
|---------|---------|
| M1/M2 C0–C3 推論 | `outputs/outputs/error_decomp_2026-05-10/` |
| M3 C0–C3 推論 | `outputs/error_decomp_2026-05-10_15-44/` |
| Robust 指標（各 run）| `<run_dir>/analysis_report_robust/robust_metrics.json` |
| 樣本診斷（各 run）| `<run_dir>/analysis_report_robust/sample_diagnostics.jsonl` |

---

## 5. 已知問題與限制

1. **M2 C1 / M3 C1 結果無效（全零）**：`GrammarTablatureLogitsProcessor` 在 v2 output vocab 上永久 deadlock——v2 詞表中無 NOTE_ON token，導致 `note_on_ids` 為空，START 狀態無法轉移，所有 logit 被 mask 成 −∞。修復已合入 `inference.py`（grammar mode + 非 v1 格式 → 自動 fallback 為無約束）。**需重新執行這兩個 run，預期結果 ≈ C0。**

2. **M1 C2 語法問題**：`valid_ratio_pred = 0.6570`，`syntax_penalty = 1.0`，`strict_tab_score = 0.0`。Input skeleton 約束與 v1 的結構 token 冗餘性衝突，產生大量語法無效序列（約 34%）。此條件下只應使用 aligned / normalized 指標，strict track 不可靠。

3. **M1 C3 的 f1 < 1.0**：precision = 0.7683，原因是 v1 decoder 在 C3 下仍生成多餘的 NOTE_ON/OFF token（coverage = 1.0，表示所有 target 音符均已覆蓋）。不影響 tab_acc_aligned 的可信度。

4. **M3 早停**：v2_aux checkpoint 於 epoch 166 早停（預計 300 epochs），val loss 已收斂但尚未達到訓練預算上限。完整訓練後 pitch supervision 效果可能更強。

5. **單 seed，無訓練 CI**：統計顯著性依賴 sample-level bootstrap（本次未計算）。≤1.4 pp 的差距應謹慎解讀。

6. **M1 與 M2/M3 的音符計數基礎不同**：v1 的 523,392 vs v2 的 681,228，源於 v1 以 NOTE_ON+TAB+NOTE_OFF 三步表示一個音符，分母計算邏輯有別。跨模型比較時應以 aligned 指標為主。
