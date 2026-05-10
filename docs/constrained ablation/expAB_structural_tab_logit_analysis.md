# Exp A+B：Structural Token 信心分析 & Constrained vs Unconstrained TAB 比較

**實驗日期**：2026-05-09  
**Log 來源**：`logs_inference_final/slurm-902797-inf-struct-logit.out`  
**Checkpoint**：`ckpt/combine_v1_token_200_epochs/best_model.pt`  
**格式**：v1（`NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT`）  
**Constrained mode**：`input_skeleton`

---

## 實驗設計

| Condition | 說明 |
|-----------|------|
| **Constrained** | input_skeleton 模式；NOTE_ON/NOTE_OFF/TIME_SHIFT 強制跟隨 ground truth，TAB token 做 pitch mask |
| **Unconstrained (stats_only)** | 不施加任何 mask，但 logits processor 持續追蹤步驟位置並收集 TAB token 的信心統計 |

- **Exp A**：在 constrained 條件下，量測 structural token（NOTE_ON/OFF/TIME_SHIFT）在 pre-mask logits 上的預測信心。
- **Exp B**：比較 constrained vs unconstrained 條件下，TAB token 的 logits 信心差異，驗證「正確結構上下文是否有助提升 TAB 預測信心」。

---

## Accuracy 比較

| 指標 | Constrained | Unconstrained | Delta |
|------|-------------|---------------|-------|
| Token Accuracy | **95.08%** | 89.78% | +5.30 pp |
| Pitch Accuracy | **100.00%** | 89.41% | +10.59 pp |
| Pitch Acc (NOTE_ON) | **100.00%** | 95.50% | +4.50 pp |
| Tab Accuracy | **82.49%** | 74.15% | +8.34 pp |
| Total Notes | 523,392 | 523,392 | — |

> Constrained 的 Pitch Accuracy = 100% 為預期行為，因為 NOTE_ON token 被強制替換為 ground truth，pitch 必然正確。Tab Accuracy 提升 **8.34 pp** 為直接效益。

---

## Exp B：TAB Token Logit 信心比較

### 整體統計

| 指標 | Constrained (n=681,228) | Unconstrained (n=677,301) | 差異 |
|------|-------------------------|---------------------------|------|
| P(valid TABs) | **95.49%** ± 12.81% | 79.54% ± 39.06% | **+15.95 pp** |
| Entropy within valid | **0.0600** ± 0.1581 nats | 0.1080 ± 0.2308 nats | −0.048（更集中） |
| Normalized entropy | **0.0552** ± 0.1416 | 0.0946 ± 0.1892 | −0.039 |
| KL from Uniform | **1.1026** ± 0.5189 nats | 1.0551 ± 0.5214 nats | +0.047 |
| Top-1/Top-2 Margin | **7.3891** ± 4.3654 | 6.7675 ± 4.5347 | +0.62 |
| Free argmax is valid | **97.48%** ± 15.69% | 80.67% ± 39.48% | **+16.81 pp** |

**關鍵觀察**：Unconstrained 條件下 P(valid TABs) 的標準差高達 **39.06%**（constrained 僅 12.81%），顯示無約束時分佈呈雙峰（bimodal）——要麼完全正確，要麼完全偏移，整體行為不穩定。

### 按 Pitch Ambiguity 分層

**Constrained**：

| Bucket | Count | Entropy | KL | Margin | P_valid | FreeValid |
|--------|-------|---------|----|--------|---------|-----------|
| ambi_1 | 68,409 | 0.0000 | 0.0000 | 0.0000 | 92.45% | 95.0% |
| ambi_2 | 115,410 | 0.0595 | 0.6337 | 7.9825 | 94.68% | 97.1% |
| ambi_3_4 | 258,380 | 0.0593 | 1.1786 | 8.6080 | 95.62% | 97.5% |
| ambi_5+ | 239,029 | 0.0782 | 1.5623 | 7.8997 | **96.60%** | **98.3%** |

**Unconstrained**：

| Bucket | Count | Entropy | KL | Margin | P_valid | FreeValid |
|--------|-------|---------|----|--------|---------|-----------|
| ambi_1 | 67,855 | 0.0000 | 0.0000 | 0.0000 | 83.2% | 83.2% |
| ambi_2 | 114,648 | 0.0757 | 0.6175 | 7.6306 | 82.54% | 84.0% |
| ambi_3_4 | 256,900 | 0.1104 | 1.1275 | 7.8925 | 80.22% | 81.3% |
| ambi_5+ | 237,898 | 0.1517 | 1.4888 | 7.0671 | **76.91%** | **77.7%** |

### 分層結果解讀

兩個條件下 ambiguity 與 P_valid 的趨勢**完全相反**：

- **Constrained**：ambiguity 越高 → P_valid 越高。合理，valid set 越大，機率質量自然更容易落在 valid 範圍內；而模型有正確結構上下文，能準確辨識 pitch。
- **Unconstrained**：ambiguity 越高 → P_valid **越低**（76.91%，比 ambi_1 的 83.2% 還低）。代表模型在缺少結構上下文時，高 ambiguity 的 pitch 更難正確預測，錯誤傾向隨複雜度增加而惡化。

這個趨勢反轉是 **Exp B 最強的實驗證據**，直接支持「正確結構上下文（即 NOTE_ON/TIME_SHIFT constraint）有助模型在 TAB token 預測時做出更準確的判斷」。

---

## Exp A：Structural Token 信心分析（Constrained 條件）

### 整體與分類統計（n = 1,852,973 步）

| Token 類型 | n | Prob on correct | Free argmax correct | Margin | Correct rank | Entropy |
|-----------|---|-----------------|---------------------|--------|-------------|---------|
| **Overall** | 1,852,973 | 94.93% ± 12.14% | **98.31%** ± 12.89% | 9.0195 ± 5.14 | 0.0174 | 0.1185 nats |
| NOTE_ON | 681,228 | 94.51% ± 13.54% | 97.46% ± 15.75% | 8.8036 ± 5.03 | 0.0265 | 0.1205 nats |
| NOTE_OFF | 681,228 | 95.06% ± 11.13% | **98.90%** ± 10.43% | 9.2910 ± 5.37 | 0.0111 | 0.1212 nats |
| TIME_SHIFT | 490,517 | 95.32% ± 11.38% | 98.68% ± 11.43% | 8.9424 ± 4.93 | 0.0136 | 0.1120 nats |

### 解讀

1. **模型對 structural token 已有極高信心**：`free_argmax_is_correct` 整體 98.31%，即使不施加任何強制，模型也能自行預測正確的結構 token。Full-vocab entropy 僅 ~0.12 nats（uniform = ln(910) ≈ 6.81 nats），極度集中在正確 token 上。

2. **NOTE_ON 最難預測**（free argmax correct = 97.46%）：NOTE_ON 需要決定 pitch 與音符時間位置，資訊量最大，錯誤率最高（约 2.54%）。

3. **NOTE_OFF 最容易**（98.90%）：NOTE_OFF 有 NOTE_ON 的配對結構支撐，模型幾乎能無誤地判斷何時關閉音符。

4. **Constraint 對 structural token 的作用**：主要是防止 2–3% 的罕見錯誤（correct rank ≈ 0.017 → 正確 token 幾乎永遠是 top-1）。在 unconstrained 條件下這些錯誤會累積，導致序列上下文偏移，進而影響後續 TAB token 的預測（如 Exp B 所示）。

---

## 總結

### 與 Exp 5（v2 format）的對比

| | Exp 5（v2, pitch-only constraint） | Exp A+B（v1, structural constraint） |
|---|---|---|
| 格式 | v2：decoder 只輸出 TAB/TIME_SHIFT | v1：decoder 輸出 NOTE_ON/TAB/NOTE_OFF/TIME_SHIFT |
| 約束對象 | TAB token 的 pitch mask | NOTE_ON/OFF/TIME_SHIFT 強制 + TAB pitch mask |
| Unconstrained P_valid | ~99.77%（model 已內化） | **79.54%**（明顯下降） |
| Free argmax = constrained | ~99.80% | **80.67%** |
| 結論 | model 已透過訓練內化 pitch 約束；constraint 為 safety net | **正確結構上下文對 TAB 預測有顯著影響** |

### 核心 Claim 的支持程度

本實驗強力支持：「**constrained decoding 透過提供正確的上下文（pitch + time + structural tokens）確實能幫助模型在預測 TAB 時表現更好。**」

- Constrained vs Unconstrained 的 P(valid TABs) 差距達 **+15.95 pp**（95.49% vs 79.54%）
- Free argmax valid rate 差距 **+16.81 pp**
- High-ambiguity pitch 的趨勢反轉（constrained 下 ambi 越高越好，unconstrained 下 ambi 越高越差）提供了機制層面的解釋
- Structural token confidence 高（98.31% free argmax correct）但非完美，2–3% 的錯誤在累積效應下即足以造成 TAB 預測崩潰

### 輸出文件

| 類型 | 路徑 |
|------|------|
| Exp A 圖表 | `outputs/2026-05-09_01-52-struct-logit-v1-constrained/structural_logit_analysis/` |
| Exp B 比較圖 | `outputs/2026-05-09_01-52-struct-logit-v1-expB-comparison/` |
| TAB stats (constrained) | `outputs/2026-05-09_01-52-struct-logit-v1-constrained/logit_stats.pt` |
| Structural stats | `outputs/2026-05-09_01-52-struct-logit-v1-constrained/structural_logit_stats.pt` |
| TAB stats (unconstrained) | `outputs/2026-05-09_01-52-struct-logit-v1-unconstrained/logit_stats.pt` |
