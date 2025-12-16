#!/bin/bash
#SBATCH --job-name=MIR-anl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=normal

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

python analyze_output.py \
    out-v1-500epoch \
    --post \
    --timeline-tolerance 0