CHECKPOINT=${1:-outputs/2026-03-27_15-43/best_model.pt}

echo "========================================"
echo "Inference on DadaGP test set"
echo "========================================"
python inference.py data=dadagp \
  +checkpoint_path=${CHECKPOINT}

echo ""
echo "========================================"
echo "Inference on Leduc test set"
echo "========================================"
python inference.py data=leduc \
  "data.selected_files_json=10984521/FrancoisLeducDatasetPublication/FrancoisLeducDatasetPublication/test_files.json" \
  +checkpoint_path=${CHECKPOINT}
