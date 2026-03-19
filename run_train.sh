#!/bin/bash
#SBATCH --job-name=tr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=normal
#SBATCH --output=slurm-%j-train-v1.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

RUN_TAG="$(date +%Y-%m-%d_%H-%M)-v1"
python train.py \
  data=train_split \
  data.output_format=v1 \
  experiment_name="$RUN_TAG"