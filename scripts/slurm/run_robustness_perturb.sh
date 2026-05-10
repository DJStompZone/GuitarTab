#!/bin/bash
#SBATCH --job-name=robust-perturb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_inference_final/slurm-%j-robustness-perturb.out

# ============================================================================
# Exp 5: Input perturbation robustness evaluation
# Models: M1 (v1), M2 (v2), M3 (v2+aux, if ckpt exists)
# Conditions: C2 (skeleton, no pitch mask) and C3 (skeleton + pitch mask)
# Perturbation types:
#   pitch_jitter: ±1 or ±2 semitones on 5% / 10% of NOTE_ON events
#   time_noise:   ±5 or ±10 ticks on 5% / 10% of TIME_SHIFT events
#   drop_note_off: drop 5% / 10% of NOTE_OFF events
# ============================================================================

echo "=== Robustness Perturbation Experiment ==="
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
conda activate MusicFinal

export TQDM_DISABLE=1

CKPT_M1="/home/b10502010/work/GuitarTab/ckpt/dadagp_v1_300epcohs_weight_decay/best_model.pt"
CKPT_M2="/home/b10502010/work/GuitarTab/ckpt/dadagp_v2_300epochs_weight_decay/best_model.pt"
CKPT_M3="/home/b10502010/work/GuitarTab/ckpt/v2_aux/best_model.pt"

TEST_JSON="data_splits/test_files.json"
OUTBASE="outputs/robustness_$(date +%Y-%m-%d)"

run_perturb() {
    local MODEL_TAG="$1"
    local FMT="$2"
    local CKPT="$3"
    local COND="$4"
    local PTYPE="$5"
    local PFRAC="$6"
    local PLEVEL="$7"

    local RUN_TAG="${OUTBASE}/${MODEL_TAG}_${COND}_${PTYPE}_f${PFRAC}_k${PLEVEL}"
    echo ""
    echo "--- ${MODEL_TAG} | ${COND} | ${PTYPE} frac=${PFRAC} level=${PLEVEL} ---"

    python scripts/run_inference_with_perturbation.py \
        --checkpoint "$CKPT" \
        --format "$FMT" \
        --condition "$COND" \
        --model_tag "$MODEL_TAG" \
        --test_json "$TEST_JSON" \
        --perturb_type "$PTYPE" \
        --perturb_frac "$PFRAC" \
        --perturb_level "$PLEVEL" \
        --output_dir "$RUN_TAG"
}

# ─────────────────────────────
# Baseline (no perturbation) for each model × condition
# ─────────────────────────────
for COND in C2 C3; do
    run_perturb "M1" "v1" "$CKPT_M1" "$COND" "none" 0.0 0
    run_perturb "M2" "v2" "$CKPT_M2" "$COND" "none" 0.0 0
done

# ─────────────────────────────
# Pitch jitter: ±1 / ±2 at 5% / 10%
# ─────────────────────────────
for MODEL_TAG FMT CKPT in "M1 v1 $CKPT_M1" "M2 v2 $CKPT_M2"; do
    # shellcheck disable=SC2086
    set -- $MODEL_TAG $FMT $CKPT
    for COND in C2 C3; do
        for PFRAC in 0.05 0.10; do
            for PLEVEL in 1 2; do
                run_perturb "$1" "$2" "$3" "$COND" "pitch_jitter" "$PFRAC" "$PLEVEL"
            done
        done
    done
done

# ─────────────────────────────
# Time noise: ±5 / ±10 ticks at 5% / 10%
# ─────────────────────────────
for MODEL_TAG FMT CKPT in "M1 v1 $CKPT_M1" "M2 v2 $CKPT_M2"; do
    # shellcheck disable=SC2086
    set -- $MODEL_TAG $FMT $CKPT
    for COND in C2 C3; do
        for PFRAC in 0.05 0.10; do
            for PLEVEL in 5 10; do
                run_perturb "$1" "$2" "$3" "$COND" "time_noise" "$PFRAC" "$PLEVEL"
            done
        done
    done
done

# ─────────────────────────────
# Drop NOTE_OFF: 5% / 10%
# ─────────────────────────────
for MODEL_TAG FMT CKPT in "M1 v1 $CKPT_M1" "M2 v2 $CKPT_M2"; do
    # shellcheck disable=SC2086
    set -- $MODEL_TAG $FMT $CKPT
    for COND in C2 C3; do
        for PFRAC in 0.05 0.10; do
            run_perturb "$1" "$2" "$3" "$COND" "drop_note_off" "$PFRAC" 0
        done
    done
done

# ─────────────────────────────
# M3 (v2+aux) — only if ckpt exists
# ─────────────────────────────
if [ -f "$CKPT_M3" ]; then
    for COND in C2 C3; do
        run_perturb "M3" "v2" "$CKPT_M3" "$COND" "none" 0.0 0
        for PFRAC in 0.05 0.10; do
            for PLEVEL in 1 2; do
                run_perturb "M3" "v2" "$CKPT_M3" "$COND" "pitch_jitter" "$PFRAC" "$PLEVEL"
            done
            for PLEVEL in 5 10; do
                run_perturb "M3" "v2" "$CKPT_M3" "$COND" "time_noise" "$PFRAC" "$PLEVEL"
            done
            run_perturb "M3" "v2" "$CKPT_M3" "$COND" "drop_note_off" "$PFRAC" 0
        done
    done
else
    echo ""
    echo "SKIP M3: checkpoint not found at $CKPT_M3"
fi

echo ""
echo "=== All perturbation runs complete ==="
echo "End time: $(date)"
echo "Output base: $OUTBASE"
