# 12/5 Updates

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
