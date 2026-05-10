#!/bin/bash
#SBATCH --job-name=inf-struct-logit
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

export TQDM_DISABLE=1

# ── Configuration ─────────────────────────────────────────────────────────
CHECKPOINT_PATH="/home/b10502010/work/GuitarTab/ckpt/dadagp_v1_300epcohs_weight_decay/best_model.pt"
OUTPUT_FORMAT="v1"
VOCAB_SIZE=886   # output vocab size; update if your model differs
RUN_TAG_BASE="$(date +%Y-%m-%d_%H-%M)-struct-logit-v1"
# ──────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "Exp A + B: Structural Token & TAB Confidence Analysis (v1)"
echo "Checkpoint : $CHECKPOINT_PATH"
echo "Format     : $OUTPUT_FORMAT"
echo "================================================================"

# ── Condition A+B(constrained): v1 input_skeleton ──────────────────────
RUN_TAG_C="${RUN_TAG_BASE}-constrained"
echo ""
echo "=== Condition: Constrained (input_skeleton) ==="
echo "Run tag: $RUN_TAG_C"

time python inference.py \
    data.output_format=${OUTPUT_FORMAT} \
    data.selected_files_json=data_splits/test_files.json \
    checkpoint_path="${CHECKPOINT_PATH}" \
    experiment_name="${RUN_TAG_C}" \
    constrained_decoding=true \
    constrained_decoding_mode=input_skeleton \
    save_logit_stats=true \
    save_structural_logit_stats=true

# ── Condition B(unconstrained): v1 stats_only ──────────────────────────
RUN_TAG_U="${RUN_TAG_BASE}-unconstrained"
echo ""
echo "=== Condition: Unconstrained (stats_only) ==="
echo "Run tag: $RUN_TAG_U"

time python inference.py \
    data.output_format=${OUTPUT_FORMAT} \
    data.selected_files_json=data_splits/test_files.json \
    checkpoint_path="${CHECKPOINT_PATH}" \
    experiment_name="${RUN_TAG_U}" \
    constrained_decoding=false \
    save_logit_stats=true

# ── Exp A: Structural token analysis ───────────────────────────────────
STRUCT_STATS="outputs/${RUN_TAG_C}/structural_logit_stats.pt"
STRUCT_ANALYSIS_DIR="outputs/${RUN_TAG_C}/structural_logit_analysis"

echo ""
echo "=== Exp A: Analyzing Structural Token Logit Stats ==="
if [ -f "$STRUCT_STATS" ]; then
    python scripts/analyze_structural_logit_stats.py \
        --stats_path "$STRUCT_STATS" \
        --output_dir "$STRUCT_ANALYSIS_DIR"
    echo "Structural analysis saved to: $STRUCT_ANALYSIS_DIR"
else
    echo "WARNING: structural_logit_stats.pt not found at $STRUCT_STATS"
fi

# ── Exp B: Constrained vs Unconstrained TAB comparison ─────────────────
TAB_STATS_C="outputs/${RUN_TAG_C}/logit_stats.pt"
TAB_STATS_U="outputs/${RUN_TAG_U}/logit_stats.pt"
COMPARE_DIR="outputs/${RUN_TAG_BASE}-expB-comparison"

echo ""
echo "=== Exp B: Comparing Constrained vs Unconstrained TAB Logit Stats ==="
if [ -f "$TAB_STATS_C" ] && [ -f "$TAB_STATS_U" ]; then
    python scripts/analyze_logit_stats.py \
        --compare \
        --constrained_stats "$TAB_STATS_C" \
        --unconstrained_stats "$TAB_STATS_U" \
        --output_dir "$COMPARE_DIR" \
        --vocab_size ${VOCAB_SIZE}
    echo "Comparison figures saved to: $COMPARE_DIR"
else
    echo "WARNING: one or both logit_stats.pt files not found"
    echo "  constrained:   $TAB_STATS_C"
    echo "  unconstrained: $TAB_STATS_U"
fi

echo ""
echo "================================================================"
echo "All done."
echo "  Exp A results:  $STRUCT_ANALYSIS_DIR"
echo "  Exp B results:  $COMPARE_DIR"
echo "================================================================"
