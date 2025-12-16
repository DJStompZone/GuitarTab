#!/bin/bash
# Run post-processing with FRET method (simple index-based alignment)

python run_post_processing.py \
    --output-dir out-v2-500epoch \
    --method fret \
    --output-format v2

python run_post_processing.py \
    --output-dir out-v2-500epoch \
    --method reverse \
    --output-format v2 \
    --pitch-threshold 1 \
    --time-threshold 240
