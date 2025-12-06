#!/bin/bash
#SBATCH --job-name=MIR-ddp
#SBATCH --nodes=1                # Single node (multi-GPU within node)
#SBATCH --ntasks-per-node=1      # FIXED: Only 1 task (not 4!)
#SBATCH --cpus-per-task=16       # FIXED: 4 CPUs per GPU × 4 GPUs = 16 total
#SBATCH --gres=gpu:4             # Request 4 GPUs
#SBATCH --account=ACD114010
#SBATCH --partition=normal
#SBATCH --time=24:00:00          # Max runtime

# Environment setup
module purge
ml nvhpc-hpcx-cuda12/24.7
module load miniconda3
source activate MusicFinal        # FIXED: Use 'source activate' instead of 'conda activate'

echo "=========================================="
echo "Multi-GPU Training with PyTorch DDP"
echo "Number of GPUs: 4"
echo "=========================================="

# Launch distributed training with torchrun
# IMPORTANT: Do NOT use srun with torchrun! torchrun handles process spawning internally.
# Using srun would launch multiple torchrun instances, causing process duplication.
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train-dist.py \
    data=selected

echo "Training completed!"
