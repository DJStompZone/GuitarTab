#!/bin/bash
#SBATCH --job-name=MIR-inf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=normal
#SBATCH --output=slurm-%j-inference-v2.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

time python inference.py \
    output_dir=out-v2-500epoch \
    +checkpoint_path=outputs/2025-12-10_16-05-v2/checkpoint_epoch_500.pt \
    ++data.output_format=v2