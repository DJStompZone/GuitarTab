#!/bin/bash
#SBATCH --job-name=inf-v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=normal
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%j-inf-v2.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

# Optional: pass checkpoint path as first argument.
# Example:
#   sbatch run_inference_v2.sh outputs/xxx/best_model.pt
CHECKPOINT_PATH="${1:-outputs/2026-03-12_15-09-fix-timeshift/best_model.pt}"
RUN_TAG="$(date +%Y-%m-%d_%H-%M)-inference-v2"

export TQDM_DISABLE=1

time python inference.py \
  data=test_split \
  data.output_format=v2 \
  checkpoint_path="$CHECKPOINT_PATH" \
  experiment_name="$RUN_TAG"
