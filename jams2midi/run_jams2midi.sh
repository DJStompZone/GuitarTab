#!/bin/bash
#SBATCH --job-name=jams2midi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=normal

module purge
module load miniconda3
conda activate MusicFinal

python jams2midi.py "../Dataset/" "../Dataset_midi/" "keyswitch_config.json"