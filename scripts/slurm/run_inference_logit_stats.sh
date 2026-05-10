#!/bin/bash
#SBATCH --job-name=inf-logit-stats
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_inference_final/slurm-%j-%x.out

echo "Running on node: $(hostname)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

# ── Configuration ─────────────────────────────────────────────────────────
# Change these to match the checkpoint / format you want to analyze
CHECKPOINT_PATH="/home/b10502010/work/GuitarTab/ckpt/combine_v2_token_200_epochs/best_model.pt"
OUTPUT_FORMAT="v2"
RUN_TAG="$(date +%Y-%m-%d_%H-%M)-inf-logit-stats-cmb_${OUTPUT_FORMAT}"
VOCAB_SIZE=886   # output vocab size; update if your model differs
# ──────────────────────────────────────────────────────────────────────────

echo "=== Experiment 5: Logit Stats Collection ==="
echo "Checkpoint : $CHECKPOINT_PATH"
echo "Format     : $OUTPUT_FORMAT"
echo "Run tag    : $RUN_TAG"

time python inference.py \
    data.output_format=${OUTPUT_FORMAT} \
    data.selected_files_json=data_splits/test_files.json \
    checkpoint_path="${CHECKPOINT_PATH}" \
    experiment_name="${RUN_TAG}" \
    constrained_decoding=true \
    constrained_decoding_mode=input_skeleton \
    save_logit_stats=true

# ── Post-processing: generate analysis figures ─────────────────────────────
STATS_FILE="outputs/${RUN_TAG}/logit_stats.pt"
ANALYSIS_DIR="outputs/${RUN_TAG}/logit_analysis"

echo ""
echo "=== Generating Analysis Figures ==="
if [ -f "$STATS_FILE" ]; then
    python scripts/analyze_logit_stats.py \
        --stats_path "$STATS_FILE" \
        --output_dir "$ANALYSIS_DIR" \
        --vocab_size ${VOCAB_SIZE}
    echo "Figures saved to: $ANALYSIS_DIR"
else
    echo "WARNING: logit_stats.pt not found at $STATS_FILE"
fi

echo "Done."
