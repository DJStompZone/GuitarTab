#!/bin/bash
#SBATCH --job-name=inf-const-cmb_v2
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

CHECKPOINT_PATH="/home/b10502010/work/GuitarTab/ckpt/combine_v2_token_200_epochs/best_model.pt"
RUN_TAG="$(date +%Y-%m-%d_%H-%M)-inference-constrained-cmb_v2"

time python inference.py \
    data.output_format=v2 \
    data.selected_files_json=data_splits/test_files.json \
    checkpoint_path="$CHECKPOINT_PATH" \
    experiment_name="$RUN_TAG" \
    constrained_decoding=true \
    constrained_decoding_mode=input_skeleton 
    # disable_kv_cache=true