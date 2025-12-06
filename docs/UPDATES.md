# Updates

## 12/6 - Post-Processing Integration

### New Features
- **Post-processing module integration**: 整合 `fretting_postprocessor/` 到推論流程
- **Pitch accuracy improvement**: 音高準確度可提升至 ~100%
- **Two processing methods**:
  - Overlap Correction: ~99.92% 音高準確度
  - Neighbor Search: ~100% 音高準確度（推薦）

### Usage
```bash
# 啟用後處理（使用預設 neighbor_search 方法）
python inference.py postprocessing.enabled=true

# 使用 overlap correction 方法（較快）
python inference.py postprocessing.enabled=true postprocessing.method=overlap

# 自訂吉他配置
python inference.py postprocessing.enabled=true \
    postprocessing.guitar.tuning=drop_d \
    postprocessing.guitar.capo_fret=2
```

### New Files
- `configs/postprocessing.yaml` - 後處理配置
- `src/postprocessing_bridge.py` - Token 格式轉換與批次處理
- `docs/POST_PROCESSING_GUIDE.md` - 完整使用指南
- `POSTPROCESSING_INTEGRATION_PLAN.md` - 實作計劃文檔

### Modified Files
- `configs/inference.yaml` - 加入 postprocessing defaults
- `src/metrics.py` - 新增 PostProcessingMetrics 和比較功能
- `inference.py` - 整合後處理流程

### Documentation
詳見 `docs/POST_PROCESSING_GUIDE.md` 獲取完整使用說明。

---

## 12/5 Updates

## Quick Start
```bash
# 1. Copy / symlink DadaGP-v1.1 to ./DadaGP-v1.1
ln -s ...

# 2. Train
# only 1000 files
python train.py data=selected data.max_files=1000

# unlimited (all selected files)
python train.py data=selected

# 3. Test and evalutation
python inference.py +checkpoint_path=./outputs/<exp_dir>/best_model.pt
```

## Data and tokenization
- TODO
