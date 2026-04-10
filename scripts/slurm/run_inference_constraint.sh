#!/bin/bash
## SBATCH --job-name=inf-const-ft-rst-best-20
#SBATCH --job-name=inf-const-ft-tb_only-best-200
##SBATCH --job-name=inf-const-30-da_KV
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_inference/slurm-%j-%x.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

# CHECKPOINT_PATH="/home/b10502010/work/GuitarTab/outputs/2026-04-01_03-10-v1/checkpoint_epoch_100.pt"
# CHECKPOINT_PATH="outputs/2026-04-08_01-10-ft-constrained-tab_only-50/checkpoint_epoch_200.pt"
CHECKPOINT_PATH="/home/b10502010/work/GuitarTab/outputs/2026-04-08_01-08-ft-constrained-tab_only-best/checkpoint_epoch_200.pt"
# CHECKPOINT_PATH="/home/b10502010/work/GuitarTab/outputs/2026-04-08_01-17-ft-constrained-rst-best/checkpoint_epoch_20.pt"
RUN_TAG="$(date +%Y-%m-%d_%H-%M)-inference-constrained-ft-tb_only-best-200"

time python inference.py \
    data.output_format=v1 \
    data.selected_files_json=data_splits/test_files.json \
    checkpoint_path="$CHECKPOINT_PATH" \
    experiment_name="$RUN_TAG" \
    constrained_decoding=true \
    constrained_decoding_mode=input_skeleton \
    disable_kv_cache=true