#!/bin/bash
#SBATCH --job-name=analysis-v1v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --time=04:00:00
#SBATCH --output=logs_inference/slurm-%j-%x.out

set -eo pipefail

echo "Running on node: $(hostname)"

source /etc/profile || true
module purge || true
module load miniconda3 || ml miniconda3
module load cuda || true
source "$(conda info --base)/etc/profile.d/conda.sh" || true
conda activate MusicFinal

BASE_DIR="/work/b10502010/GuitarTab"
cd "$BASE_DIR"

TOLERANCES=("5" "10" "20")
FORMATS=("v1" "v2")

for fmt in "${FORMATS[@]}"; do
  OUTPUT_DIR="outputs/2026-04-17_15-16-inference-cmb_${fmt}"
  for tol in "${TOLERANCES[@]}"; do
    echo "Analyze fmt=${fmt}, tolerance=${tol}"
    python analyze_output_robust.py "$OUTPUT_DIR" \
      --timeline-tolerance "$tol" \
      --output-dir "$OUTPUT_DIR/analysis_report_robust_formataware_t${tol}"
  done
done

echo "Analysis matrix completed."
