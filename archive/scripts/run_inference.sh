#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_inference/slurm-%j-inf-v1.out

echo "Running on node: $(hostname)"

module purge
# ml nvhpc-hpcx-cuda12/24.7
module load miniconda3 cuda
conda activate MusicFinal

time python inference.py \
    data=test_split \
    
    # constrained_decoding=true