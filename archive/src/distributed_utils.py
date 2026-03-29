"""
Distributed training utilities for PyTorch DDP.

Provides functions for initializing and managing distributed training across multiple GPUs.
"""

import os
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Dict, Any


def setup_distributed() -> Dict[str, Any]:
    """
    Initialize distributed environment and return configuration.

    Reads environment variables set by torchrun (RANK, LOCAL_RANK, WORLD_SIZE)
    and initializes the distributed process group with NCCL backend.

    Returns:
        Dict containing:
            - rank: Global rank of this process
            - local_rank: Local rank on this node
            - world_size: Total number of processes
            - is_main_process: True if rank 0
    """
    # Get distributed environment variables (set by torchrun)
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Initialize process group
    if not dist.is_initialized():
        # Use NCCL backend for GPU training
        dist.init_process_group(
            backend='nccl',
            init_method='env://',  # Use environment variables
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30)  # 30-minute timeout
        )

    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_main_process': (rank == 0)
    }


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """
    Return current process rank.

    Returns:
        Global rank of this process (0 if not initialized)
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Return total number of processes.

    Returns:
        Total number of processes (1 if not initialized)
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Returns:
        True if rank 0 or not initialized
    """
    return get_rank() == 0


def barrier():
    """
    Synchronization barrier across all processes.

    All processes wait here until all processes reach this point.
    No-op if distributed training is not initialized.
    """
    if dist.is_initialized():
        dist.barrier()


def reduce_dict(input_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Average metrics across all processes using all_reduce.

    Args:
        input_dict: Dictionary of metrics to average (e.g., {'loss': 0.5})

    Returns:
        Dictionary with averaged values across all processes

    Note:
        This function performs in-place all_reduce and returns averaged values.
        If distributed training is not initialized, returns input_dict unchanged.
    """
    if not dist.is_initialized():
        return input_dict

    world_size = get_world_size()

    # Convert dict to tensors for all_reduce
    names = []
    values = []
    for k, v in input_dict.items():
        names.append(k)
        values.append(v)

    # Stack into single tensor
    values = torch.tensor(values, dtype=torch.float32, device='cuda')

    # All-reduce: sum across all processes
    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    # Average by world size
    values /= world_size

    # Convert back to dict
    reduced_dict = {k: v.item() for k, v in zip(names, values)}

    return reduced_dict


def print_rank_0(message: str):
    """
    Print message only from rank 0 process.

    Args:
        message: String to print
    """
    if is_main_process():
        print(message)
