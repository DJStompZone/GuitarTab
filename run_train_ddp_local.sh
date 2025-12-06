#!/bin/bash
# Local multi-GPU training script (for testing without SLURM)

# Set which GPUs to use (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1

echo "=========================================="
echo "Local Multi-GPU Training Test"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Launch with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train-dist.py \
    data=debug \
    training.num_epochs=2

echo "Test completed!"
