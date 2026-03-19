#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=normal
#SBATCH --output=slurm-%j-inf-constrained.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

time python inference.py \
    data=test_split \
    
    # constrained_decoding=true