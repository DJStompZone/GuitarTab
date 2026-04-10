#!/bin/bash
#SBATCH --job-name=ft-rst-best
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_train/slurm-%j-%x.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

export TQDM_DISABLE=1

RUN_TAG="$(date +%Y-%m-%d_%H-%M)-ft-constrained-rst-best"

# Finetune：設成既有 best_model.pt / checkpoint_epoch_*.pt 路徑（相對專案根或絕對路徑）
# 留空 = 從頭訓練。有設 INIT_CKPT 時會自動加：training.optimizer.lr、training.loss_mode
#   FINETUNE_LR（預設 3e-4）、FINETUNE_LOSS_MODE（預設 tab_restricted；可改 tab_only / composite）
INIT_CKPT="outputs/2026-04-01_03-10-v1/best_model.pt"
# 若要「完整接續同一個 run」（含 optimizer state、epoch），改用 RESUME_CKPT，且不要同時設 INIT_CKPT
RESUME_CKPT=""

if [[ -n "$INIT_CKPT" && -n "$RESUME_CKPT" ]]; then
  echo "Error: set only one of INIT_CKPT or RESUME_CKPT" >&2
  exit 1
fi

HYDRA_OVERRIDES=(
  data=dadagp
  data.output_format=v1
  experiment_name="$RUN_TAG"
)
if [[ -n "$INIT_CKPT" ]]; then
  HYDRA_OVERRIDES+=(training.init_checkpoint="$INIT_CKPT")
  # Finetune 預設：較低 lr + tab_restricted（與 constrained 推論更對齊）；可用環境變數覆寫
  HYDRA_OVERRIDES+=(training.optimizer.lr="${FINETUNE_LR:-3e-4}")
  HYDRA_OVERRIDES+=(training.loss_mode="tab_restricted")
fi
if [[ -n "$RESUME_CKPT" ]]; then
  HYDRA_OVERRIDES+=(training.resume_checkpoint="$RESUME_CKPT")
fi

# 預設 loss_mode=tab_only；可改：training.loss_mode=tab_restricted / composite
# Finetune 常會降 lr，例如：training.optimizer.lr=3e-4
python train_tab_constrained.py "${HYDRA_OVERRIDES[@]}"
