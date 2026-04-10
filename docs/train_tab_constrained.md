# TAB 導向訓練（`train_tab_constrained.py`）

## 目的

對齊 **input_skeleton** constrained decoding：推論時 `NOTE_ON` / `NOTE_OFF` / `TIME_SHIFT` 由輸入決定，模型主要學 **TAB**。本腳本在 **teacher forcing** 下可只對 TAB（可選 pitch 合法子集）計算 CE，或與全序列 CE 做 **composite**。

一般訓練仍使用 **`train.py`**（不改行為）。

## 執行

```bash
# 需先載入環境（見專案 .cursor/rules）
python train_tab_constrained.py
```

### 從既有 checkpoint finetune

- **`training.init_checkpoint`**：只載入 `model_state_dict`，**epoch 從 1 重來**，optimizer 新建（適合換 TAB loss / 換學習率）。  
  例：`python train_tab_constrained.py training.init_checkpoint=outputs/foo/best_model.pt training.optimizer.lr=3e-4`
- **`training.resume_checkpoint`**：載入 model + optimizer，**下一個 epoch = 存檔裡的 epoch + 1**（同一個 run 斷點續跑）。兩者**不要同時**設。

Slurm：`scripts/slurm/run_train_tab_constrained.sh` 裡設環境變數 `INIT_CKPT` 或 `RESUME_CKPT`。

Hydra 入口：`configs/train_tab_constrained.yaml`（override `training: tab_focused`）。

## 設定（`configs/training/tab_focused.yaml`）

| 欄位 | 說明 |
|------|------|
| `loss_mode` | `full`（等同原版）、`tab_only`、`tab_restricted`、`composite` |
| `composite_lambda` | `composite` 時：`loss_full + λ * loss_tab_restricted` |
| `tab_loss_num_frets` | 預設 `${data.num_frets}`，須與 constrained decoding 一致 |

覆寫範例：

```bash
python train_tab_constrained.py training.loss_mode=tab_restricted
python train_tab_constrained.py training.loss_mode=composite training.composite_lambda=0.3
```

## 實驗建議

1. **對照**：相同 epoch / data 下比較 `full` vs `tab_only` vs `tab_restricted` / `composite`。
2. **推論**：與部署一致時，用 `inference.py` 開 `constrained_decoding: true`、`constrained_decoding_mode: input_skeleton`，對照 **tab_accuracy / pitch_accuracy**（見 `src/metrics.py`）。
3. **日誌**：teacher-forcing val loss 在 `tab_only` 下數值尺度會變（僅 TAB token 平均），建議與 **AR + constrained** 指標一併看；必要時另 log `loss_mode=full` 僅供監控（需改腳本或另跑 eval）。
4. **v2 / v3**：`tab_restricted` / pitch 對齊目前以 **v1**（target 內 `NOTE_ON` 緊鄰 TAB）為準；若 `output_format` 為 v2，需改 `src/training_loss.py` 從 **input** 對齊 pitch（比照 metrics v2 邏輯）。

## 實作位置

- Loss：`src/training_loss.py`（`compute_tab_focused_loss`、`mask_labels_tab_only`）
- 訓練迴圈：`train_tab_constrained.py`（**勿改** `train.py`）
- 測試：`tests/test_training_loss.py`
