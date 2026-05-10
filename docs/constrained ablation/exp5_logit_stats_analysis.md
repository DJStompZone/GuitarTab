# Experiment 5：Logit Statistics 結果分析與 Claim 不符之原因推論

**Run**: `2026-05-08_20-15-inf-logit-stats-cmb_v2`  
**Checkpoint**: `ckpt/combine_v2_token_200_epochs/best_model.pt` (epoch 85)  
**Mode**: `input_skeleton` constrained decoding, v2 output format  
**樣本數**: 681,228 TAB 決策步驟（來自 4,725 個 test segments）

---

## 1. 核心數字

| 指標 | 值 |
|---|---|
| Prob mass on valid TABs | **99.77%** ± 4.19% |
| Normalized entropy (0=conf, 1=unif) | **0.0287** ± 0.1093 |
| KL from Uniform | **1.1323** ± 0.5195 nats |
| Top-1/Top-2 margin | **11.55** ± 6.47 logit |
| Free argmax is valid TAB | **99.80%** ± 4.49% |
| Free == Constrained choice | **99.80%** ± 4.49% |

### Ambiguity-stratified breakdown

| Bucket | Count | Entropy | KL | Margin | P\_valid | FreeValid |
|---|---|---|---|---|---|---|
| ambi\_1 (唯一解) | 68,409 | 0.000 | 0.000 | 0.000 | 99.79% | 99.8% |
| ambi\_2 | 115,410 | 0.036 | 0.657 | 12.19 | 99.92% | 99.9% |
| ambi\_3\_4 | 258,380 | 0.029 | 1.209 | 13.20 | 99.78% | 99.8% |
| ambi\_5plus | 239,029 | 0.037 | 1.603 | 12.76 | 99.67% | 99.7% |

### 最終 Accuracy

| 指標 | 值 |
|---|---|
| Token Accuracy | 84.07% |
| Pitch Accuracy | 100.00% |
| Tab Accuracy | 73.08% |

---

## 2. 原始 Claim 與實驗預期

**原始 claim**：  
> Constrained decoding 能提供正確的上下文（time、pitch information），讓模型在預測 tablature 時表現得更好。

對應的 Exp 5 假設應是：
- **若 claim 成立**：模型在 unconstrained 狀態下對 valid TAB tokens 的 probability mass **偏低**，entropy **偏高**（模型不確定）；constrained decoding 透過排除 invalid tokens，迫使模型集中注意力，從而提高 confidence。
- **若 claim 不成立**：即使不施加 constraint，模型自己就已幾乎只預測 valid tokens，entropy 極低，constraint 幾乎不做任何事。

---

## 3. 實際結果與 Claim 的衝突

實驗結果顯示**後者**：

1. **99.77% prob mass 已在 valid TABs 上**：模型 unconstrained 輸出幾乎全部集中在 valid token。Constraint 實際上只在 0.23% 的情況下真正「強制修正」模型的 distribution。

2. **99.80% Free == Constrained choice**：模型的 unconstrained argmax 有 99.8% 和 constrained argmax 完全一致。也就是說，就算拿掉 constraint，模型幾乎永遠會做出一樣的選擇。

3. **極低的 normalized entropy（2.87%）**：模型在 valid TAB 集合內幾乎永遠是「all-in 一個選項」，幾乎沒有猶豫。這意味著模型並非因為「constraint 縮小了搜尋空間才變得 confident」，而是本身就極度 confident。

4. **ambi_1（唯一解）bucket 有 0 entropy 且 99.8% FreeValid**：這些本來就沒有選擇，但值得注意的是 ambi_5plus 的 FreeValid 仍高達 99.7%，說明即使有 5+ 個合法選項，模型也幾乎不會「跑偏」。

**結論**：Constrained decoding 的作用並非「給模型提供 context 讓它更好地選擇」，而比較接近一個「安全網（safety net）」——在極少數情況下防止模型出錯，而不是積極地引導模型利用上下文。

---

## 4. 可能的根本原因

### 4.1 訓練時已學會利用 context（最主要原因）

T5 encoder-decoder 架構中，decoder 在每一個 TAB token 預測時，都透過 **cross-attention** 對齊到 encoder 的輸出。Encoder 看到的是包含正確 `NOTE_ON_{pitch}` 的 input skeleton，因此 decoder 在訓練時就已學會：

> 當 encoder 傳來 pitch=X 的 context 時，對應的 TAB 只能在特定的 (string, fret) 組合中。

換言之，**pitch constraint 已被模型內化（internalized）**，外掛的 logit mask 是多餘的（但不是有害的）。

### 4.2 Teacher Forcing 的影響

Training 使用 teacher forcing，decoder 的輸入永遠是 ground truth tokens。這直接讓模型學到「在看到正確 input skeleton 的情況下，接下來應該輸出哪個 TAB」——這個知識已包含 pitch-to-tab 的 constraint。

### 4.3 Experiment 5 測量的是錯誤問題

Exp 5 設計的假設是「constraint 幫助 model 在 valid 集合內做更好的選擇」，但實驗只能測量：
- 「model 本身有多少 prob mass 在 valid 集合內」
- 「free argmax 和 constrained argmax 的一致性」

這些指標**無法區分**以下兩種情況：
- (A) 模型靠 constrained decoding 提供的 context 才知道要選 valid token
- (B) 模型靠 training 就已學會不需要 constraint 也能選正確

實驗結果明確支持 (B)，但這並不能完全否定 claim——claim 的核心或許是在訓練時 constraint 就已 implicitly 提供了 context。

### 4.4 Input Skeleton Mode 的特殊性

`input_skeleton` 模式中，所有的 `NOTE_ON`、`TIME_SHIFT` token 都直接從 input 複製，decoder 只需要插入 `TAB` tokens。Encoder 的完整 skeleton 信息對 decoder 完全可見，這讓 cross-attention 極其容易對齊——pitch 信息不需要 constraint 就能傳達。

---

## 5. 對 Claim 的修正建議

原 claim 需要被更精確地重新表述：

**原版（過強）**：
> Constrained decoding 透過提供正確 context 讓模型在 TAB 預測時「主動」表現更好。

**修正版（更準確）**：
> Constrained decoding 確保即使在模型少數出錯的情況下（≈0.2%），仍能輸出 structurally valid 的 tablature；而模型本身已透過 encoder-decoder cross-attention 內化了 pitch constraint，這正是高 pitch accuracy（100%）和高 token accuracy（84.07%）的根本原因。

或者換個角度：
> Constrained decoding 的主要貢獻在於**序列層級的結構正確性**（e.g., 每個 NOTE_ON 後面一定有 TAB，NOTE_OFF 必須關閉對應音符），而非 token 層級的 probability 重新分配。

---

## 6. 後續實驗建議

| 方向 | 目的 |
|---|---|
| **Exp 1（Random Valid Baseline）** | 如果隨機選 valid token 的 accuracy 遠低於模型的 73.08%，則說明模型不只是「知道 valid 集合」，而是真的在做有意義的選擇——間接支持 context 利用 |
| **Exp 2（Progressive Context Ablation）** | 移除或 shuffle encoder 輸入的 pitch token，若 acc 大幅下降，則說明模型確實在利用 pitch context，只是已內化而非依賴 hard constraint |
| **Exp 3（Context Perturbation）** | 給錯誤的 pitch context，若模型仍傾向預測「錯誤但 pitch-consistent」的 TAB，則說明 context 確實在驅動預測 |
| **比較 unconstrained 的 Tab Accuracy** | 若 unconstrained 的 tab accuracy ≈ constrained（73.08%），則 constraint 的作用真的很小；若差距大，則說明那 0.2% 的 override 集中在困難案例 |

---

## 7. 小結

Exp 5 的結果**不支持**「constrained decoding 透過提供 context 主動引導模型」的強版本 claim，但**支持**一個修正後的敘述：**模型已透過訓練充分學習 pitch-to-tab context，constrained decoding 扮演的是 structural safety net 而非 context provider**。這本身也是一個值得在論文中呈現的有趣發現——它說明模型的 generalization 能力已足夠強，不需要依賴外部 hard constraint 就能在絕大多數情況下做出 valid 的選擇。
