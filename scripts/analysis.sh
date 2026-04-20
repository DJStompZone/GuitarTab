#!/bin/bash

OUTPUT_DIR="/home/b10502010/work/GuitarTab/outputs/2026-04-17_15-16-inference-cmb_v2"

python analyze_output_robust.py $OUTPUT_DIR \
  --timeline-tolerance 10 \
  --output-dir $OUTPUT_DIR/analysis_report_robust
  # --config configs/inference.yaml \