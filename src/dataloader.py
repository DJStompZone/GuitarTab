"""
DataLoader utilities for creating datasets and loaders from file splits.
"""

import json
from glob import glob
from pathlib import Path
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.tab_dataset import TabDataset, collate_fn


def create_dataset(
    data_dir: str,
    token_pattern: str = "**/*.tokens.txt",
    selected_files_json: Optional[str] = None,
    max_sequence_length: int = 512,
    max_pitch: int = 127,
    max_time_shift: int = 500,
    num_strings: int = 6,
    num_frets: int = 21,
    output_format: str = "v1",
    max_files: Optional[int] = None
) -> TabDataset:
    """
    Create a TabDataset from a file list.

    Args:
        data_dir: Root directory containing token files
        token_pattern: Glob pattern for token files
        selected_files_json: Path to JSON file with selected .gp file names
        max_sequence_length: Maximum sequence length for splitting
        max_pitch: Maximum MIDI pitch
        max_time_shift: Maximum time shift in ticks
        num_strings: Number of guitar strings
        num_frets: Number of frets
        output_format: "v1" or "v2"
            - v1: output = NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT
            - v2: output = TAB, TIME_SHIFT (no NOTE_ON/OFF)
        max_files: Limit number of files (for debugging)

    Returns:
        TabDataset instance
    """
    # Find all token files
    token_files = sorted(glob(
        str(Path(data_dir) / token_pattern),
        recursive=True
    ))

    # Filter by selected files if provided
    if selected_files_json is not None:
        with open(selected_files_json, 'r') as f:
            selected_files = set(json.load(f))

        # Filter: remove .tokens.txt suffix and check against selected files
        filtered = []
        for token_file in token_files:
            if token_file.endswith('.tokens.txt'):
                gp_file = token_file[:-len('.tokens.txt')]
                if gp_file in selected_files:
                    filtered.append(token_file)

        token_files = filtered

    # Limit files if requested
    if max_files is not None:
        token_files = token_files[:max_files]

    if not token_files:
        raise ValueError(f"No token files found matching criteria")

    # Create dataset
    dataset = TabDataset(
        token_files=token_files,
        max_sequence_length=max_sequence_length,
        max_pitch=max_pitch,
        max_time_shift=max_time_shift,
        num_strings=num_strings,
        num_frets=num_frets,
        output_format=output_format,
    )

    return dataset


def create_dataloader(
    dataset: TabDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create a DataLoader from a TabDataset.

    Args:
        dataset: TabDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (set True for GPU)

    Returns:
        DataLoader instance
    """
    input_pad_id, output_pad_id = dataset.get_pad_ids()
    collate_fn_partial = partial(
        collate_fn,
        input_pad_id=input_pad_id,
        output_pad_id=output_pad_id
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader
