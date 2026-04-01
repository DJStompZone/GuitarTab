#!/bin/bash
#SBATCH --job-name=inf-v1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --time=14:00:00
#SBATCH --output=logs_inference/slurm-%j-inf-v1.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

# Optional: pass checkpoint path as first argument.
# Example:
#   sbatch run_inference_v2.sh outputs/xxx/best_model.pt
CHECKPOINT_PATH="/home/b10502010/work/GuitarTab/outputs/2026-04-01_03-10-v1/best_model.pt"
RUN_TAG="$(date +%Y-%m-%d_%H-%M)-inference-v1"

export TQDM_DISABLE=1

time python inference.py \
  data.output_format=v1 \
  data.selected_files_json=data_splits/test_files.json \
  checkpoint_path="$CHECKPOINT_PATH" \
  experiment_name="$RUN_TAG"
