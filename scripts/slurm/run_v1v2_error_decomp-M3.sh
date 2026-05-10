#!/bin/bash
#SBATCH --job-name=error-decomp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_inference_final/slurm-%j-error-decomp.out

# ============================================================================
# Exp 1 + Exp 2: M1/M2 × C0/C1/C2/C3 error decomposition
# (M3 = v2+aux, added after training completes — update CKPT_M3 below)
# ============================================================================

echo "=== Error Decomposition Run ==="
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

export TQDM_DISABLE=1

CKPT_M1="/home/b10502010/work/GuitarTab/ckpt/dadagp_v1_300epcohs_weight_decay/best_model.pt"
CKPT_M2="/home/b10502010/work/GuitarTab/ckpt/dadagp_v2_300epochs_weight_decay/best_model.pt"
# Update CKPT_M3 once v2+aux training completes:
CKPT_M3="/home/b10502010/work/GuitarTab/ckpt/v2_aux/best_model.pt"

TEST_JSON="data_splits/test_files.json"
# experiment_name is passed directly to inference.py;
# inference.yaml prepends "outputs/" → actual output = outputs/${OUTBASE}/${MODEL_TAG}_${COND_TAG}
OUTBASE="error_decomp_$(date +%Y-%m-%d_%H-%M)"

run_inference() {
    local MODEL_TAG="$1"
    local FMT="$2"
    local CKPT="$3"
    local COND_TAG="$4"
    local CONSTRAINED="$5"
    local CMODE="$6"
    local PITCH_MASK="$7"

    local EXP_NAME="${OUTBASE}/${MODEL_TAG}_${COND_TAG}"
    local OUT_DIR="outputs/${EXP_NAME}"
    echo ""
    echo "--- ${MODEL_TAG} | ${COND_TAG} ---"
    echo "  ckpt: $CKPT"
    echo "  fmt:  $FMT"
    echo "  constrained=$CONSTRAINED mode=$CMODE pitch_mask=$PITCH_MASK"
    echo "  output: $OUT_DIR"

    time python inference.py \
        data.output_format="$FMT" \
        data.selected_files_json="$TEST_JSON" \
        checkpoint_path="$CKPT" \
        experiment_name="$EXP_NAME" \
        constrained_decoding="$CONSTRAINED" \
        constrained_decoding_mode="$CMODE" \
        constrained_decoding_pitch_mask="$PITCH_MASK"

    echo "  → Running robust analysis ..."
    python analyze_output_robust.py "$OUT_DIR" \
        --output-dir "$OUT_DIR/analysis_report_robust"
}

# ─────────────────────────────
# M1 (v1 format)
# ─────────────────────────────
# run_inference "M1" "v1" "$CKPT_M1" "C0" "false" "input_skeleton" "true"
# run_inference "M1" "v1" "$CKPT_M1" "C1" "true"  "grammar"        "true"
# run_inference "M1" "v1" "$CKPT_M1" "C2" "true"  "input_skeleton" "false"
# run_inference "M1" "v1" "$CKPT_M1" "C3" "true"  "input_skeleton" "true"

# ─────────────────────────────
# M2 (v2 format)
# Note: C1 (grammar-only) is a no-op for v2 (no NOTE_ON/OFF grammar tokens)
# so C1 ≡ C0 for v2. We still run it to empirically confirm this.
# ─────────────────────────────
# run_inference "M2" "v2" "$CKPT_M2" "C0" "false" "input_skeleton" "true"
# run_inference "M2" "v2" "$CKPT_M2" "C1" "true"  "grammar"        "true"
# run_inference "M2" "v2" "$CKPT_M2" "C2" "true"  "input_skeleton" "false"
# run_inference "M2" "v2" "$CKPT_M2" "C3" "true"  "input_skeleton" "true"

# ─────────────────────────────
# M3 (v2+aux, only if ckpt exists)
# ─────────────────────────────
if [ -f "$CKPT_M3" ]; then
    run_inference "M3" "v2" "$CKPT_M3" "C0" "false" "input_skeleton" "true"
    run_inference "M3" "v2" "$CKPT_M3" "C1" "true"  "grammar"        "true"
    run_inference "M3" "v2" "$CKPT_M3" "C2" "true"  "input_skeleton" "false"
    run_inference "M3" "v2" "$CKPT_M3" "C3" "true"  "input_skeleton" "true"
else
    echo ""
    echo "SKIP M3: checkpoint not found at $CKPT_M3"
    echo "Run train_v2_pitch_aux.sh first, then rerun this script for M3."
fi

echo ""
echo "=== All runs complete ==="
echo "End time: $(date)"
echo "Output base: $OUTBASE"
