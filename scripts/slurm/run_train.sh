#!/bin/bash
#SBATCH --job-name=tr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_train/slurm-%j-train-v1.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

export TQDM_DISABLE=1

RUN_TAG="$(date +%Y-%m-%d_%H-%M)-v1"
python train.py \
  data=dadagp \
  data.output_format=v1 \
  experiment_name="$RUN_TAG"