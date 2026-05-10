#!/bin/bash
#SBATCH --job-name=error-decomp-anlys
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=ACD114010
#SBATCH --partition=gp1d
#SBATCH --output=logs_inference_final/slurm-%j-error-decomp-anlys.out
#
# 只對已有 inference 輸出做 robust analysis。
# Usage:
#   sbatch scripts/slurm/run_v1v2_error_decomp-analysis-only.sh <OUTPUT_DIR>
#   ERROR_DECOMP_RUN_DIR=/abs/path/to/outputs/foo sbatch scripts/slurm/...
#
# OUTPUT_DIR 可為：
#   - 某一輪的根目錄（底下有多個子資料夾，各含 predictions.pt / targets.pt），例如 outputs/error_decomp_2026-05-10_12-34
#   - 或直接指到單一 run 資料夾（該資料夾內就有 predictions.pt）

set -euo pipefail

echo "=== Error decomposition — analysis only ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

module purge
module load miniconda3
conda activate MusicFinal

export TQDM_DISABLE=1

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# cd "$PROJECT_ROOT"

OUTPUT_ROOT="/home/b10502010/work/GuitarTab/outputs/outputs/error_decomp_2026-05-10"
if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "錯誤：請指定輸出目錄。"
    echo "  $0 /path/to/outputs/error_decomp_YYYY-mm-dd_HH-MM"
    echo "或環境變數: ERROR_DECOMP_RUN_DIR=/path/to/dir $0"
    exit 1
fi

OUTPUT_ROOT="$(cd "$(dirname "$OUTPUT_ROOT")" && pwd)/$(basename "$OUTPUT_ROOT")"
if [[ ! -d "$OUTPUT_ROOT" ]]; then
    echo "錯誤：目錄不存在: $OUTPUT_ROOT"
    exit 1
fi

run_analysis() {
    local d="$1"
    echo ""
    echo "--- analyze: $d ---"
    time python analyze_output_robust.py "$d" \
        --output-dir "$d/analysis_report_robust"
}

if [[ -f "$OUTPUT_ROOT/predictions.pt" && -f "$OUTPUT_ROOT/targets.pt" ]]; then
    echo "Single run folder: $OUTPUT_ROOT"
    run_analysis "$OUTPUT_ROOT"
else
    echo "Scanning subdirs under: $OUTPUT_ROOT"
    found_any=0
    while IFS= read -r -d '' d; do
        if [[ -f "$d/predictions.pt" && -f "$d/targets.pt" ]]; then
            found_any=1
            run_analysis "$d"
        fi
    done < <(
        find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
    )

    if [[ "$found_any" -eq 0 ]]; then
        echo ""
        echo "錯誤：在 $OUTPUT_ROOT 底下找不到包含 predictions.pt + targets.pt 的子資料夾。"
        echo "若要分析單一 run，請把路徑指到該 run 資料夾本身。"
        exit 1
    fi
fi

echo ""
echo "=== Done ==="
echo "End: $(date)"
