#!/bin/bash
#SBATCH --job-name=process_gp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=normal

module purge
module load miniconda3
conda activate MusicFinal

python process_guitarpro.py