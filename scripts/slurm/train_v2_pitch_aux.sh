#!/bin/bash
#SBATCH --job-name=tr-v2-aux
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp4d
#SBATCH --output=logs_train/slurm-%j-train-v2-aux.out

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

export TQDM_DISABLE=1

RUN_TAG="$(date +%Y-%m-%d_%H-%M)-v2-aux"
python train_v2_pitch_aux.py \
  experiment_name="$RUN_TAG" \
  pitch_loss_weight=0.5

echo "End time: $(date)"
