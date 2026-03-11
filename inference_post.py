#!/usr/bin/env python3
"""
Inference script for Fretting-Transformer.
Runs autoregressive generation and computes accuracy metrics.
"""

import os
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from functools import partial

from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy
from src.dataloader import create_dataset, create_dataloader


def load_checkpoint(checkpoint_path: str, model, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    return model


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    """Run inference and compute accuracy."""

    print("=" * 80)
    print("Fretting-Transformer Inference")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Use test split by default for inference
    if cfg.data.selected_files_json is None:
        print("No selected_files_json specified, using test split by default...")
        cfg.data.selected_files_json = "data_splits/mini_test_files.json"

    # Create dataset
    print(f"Loading dataset from {cfg.data.selected_files_json}")
    dataset = create_dataset(
        data_dir=cfg.data.data_dir,
        token_pattern=cfg.data.token_pattern,
        selected_files_json=cfg.data.selected_files_json,
        max_sequence_length=cfg.data.max_sequence_length,
        max_pitch=cfg.data.max_pitch,
        max_time_shift=cfg.data.max_time_shift,
        num_strings=cfg.data.num_strings,
        num_frets=cfg.data.num_frets,
        max_files=cfg.data.get('max_files', None)
    )

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    print(f"Dataset: {len(dataset)} segments\n")

    input_vocab_size, output_vocab_size = dataset.get_vocab_sizes()

    # Create model
    print("Initializing model...")
    from omegaconf import OmegaConf
    model = FrettingTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        model_config=OmegaConf.to_container(cfg.model)
    ).to(device)

    # Load checkpoint
    checkpoint_path = cfg.get('checkpoint_path', 'outputs/fretting_transformer/checkpoint_epoch_200.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_checkpoint(checkpoint_path, model, device)

    # Run inference
    print("\n" + "=" * 80)
    print(f"Evaluating on {cfg.data.selected_files_json}")
    print("=" * 80)

    metrics, (input_ids, targets, predictions) = generate_and_compute_accuracy(
        model=model,
        dataloader=dataloader,
        output_vocab=dataset.output_vocab,
        device=device,
        max_length=cfg.training.get('ar_eval_max_length', 1024),
        num_beams=cfg.training.get('ar_eval_num_beams', 1),
        max_batches=cfg.get('max_eval_batches', None),  # None = all batches
        use_teacher_forcing=cfg.training.get('ar_eval_use_teacher_forcing', True),  # NEW
        temperature=cfg.training.get('ar_eval_temperature', 1.0)
    )

    print(input_ids.shape, targets.shape, predictions.shape)

    # Save predictions and targets
    os.makedirs(cfg.output_dir, exist_ok=True)
    input_ids_file = os.path.join(cfg.output_dir, "input_ids.pt")
    targets_file = os.path.join(cfg.output_dir, "targets.pt")
    predictions_file = os.path.join(cfg.output_dir, "predictions.pt")
    torch.save(input_ids, input_ids_file)
    torch.save(targets, targets_file)
    torch.save(predictions, predictions_file)


    print(f"\nResults (Before Post-Processing):")
    print(f"  Token Accuracy:  {metrics.token_accuracy:.2%}")
    print(f"  Pitch Accuracy:  {metrics.pitch_accuracy:.2%}")
    print(f"  Tab Accuracy:    {metrics.tab_accuracy:.2%}")
    print(f"  Total Tokens:    {metrics.total_tokens:,}")
    print(f"  Total Notes:     {metrics.total_notes:,}")

    # ========================================================================
    # Post-Processing
    # ========================================================================
    from tqdm import tqdm
    from src.post_processing import post_process_pitch_alignment
    from src.metrics import compute_tablature_accuracy

    print("\n" + "=" * 80)
    print("Applying pitch alignment post-processing...")
    print("=" * 80)

    # Get the vocabularies from the dataset object
    input_vocab = dataset.input_vocab
    output_vocab = dataset.output_vocab
    
    aligned_predictions = []
    
    # Convert tensors to lists for easier iteration
    input_ids_list = input_ids.cpu().tolist()
    predictions_list = predictions.cpu().tolist()
    targets_list = targets.cpu().tolist()
    
    for i in tqdm(range(len(predictions_list)), desc="Post-processing segments"):
    # for i in [0]:
        # compute tabular metric for each sequence
        m = compute_tablature_accuracy(
            predictions=torch.tensor([predictions_list[i]], dtype=torch.long),
            targets=torch.tensor([targets_list[i]], dtype=torch.long),
            output_vocab=output_vocab,
            pad_id=output_vocab.pad_id
        )
        print(f"\nSegment {i}:")
        print(f"  Before Post-Processing - Token Acc: {m.token_accuracy:.2%}, Pitch Acc: {m.pitch_accuracy:.2%}, Tab Acc: {m.tab_accuracy:.2%}")

        single_input_ids = input_ids_list[i]
        single_pred_ids = predictions_list[i]
        single_target_ids = targets_list[i]
        print(f"input len: {len(single_input_ids)}, pred len: {len(single_pred_ids)}, target len: {len(single_target_ids)}")

        # Apply alignment with full sequences (padding will be handled inside post_process_pitch_alignment)
        aligned_ids = post_process_pitch_alignment(
            input_ids=single_input_ids,
            pred_ids=single_target_ids,
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            target_ids=single_target_ids
        )
        aligned_predictions.append(aligned_ids)
        # Compute metrics after post-processing
        m_aligned = compute_tablature_accuracy(
            predictions=torch.tensor([aligned_ids], dtype=torch.long),
            targets=torch.tensor([targets_list[i]], dtype=torch.long),
            output_vocab=output_vocab,
            pad_id=output_vocab.pad_id
        )
        print(f"  After Post-Processing  - Token Acc: {m_aligned.token_accuracy:.2%}, Pitch Acc: {m_aligned.pitch_accuracy:.2%}, Tab Acc: {m_aligned.tab_accuracy:.2%}")

    # Stack aligned predictions into tensor (they are already padded to correct length)
    # Note: post_process_pitch_alignment already pads each sequence to original length
    # Here we just need to ensure all sequences match the batch's max_len
    max_len = targets.shape[1]
    padded_aligned_predictions = torch.full_like(targets, output_vocab.pad_id)
    for i, seq in enumerate(aligned_predictions):
        seq_len = min(len(seq), max_len)
        padded_aligned_predictions[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)

    # Move to the correct device
    padded_aligned_predictions = padded_aligned_predictions.to(device)

    # Save the aligned predictions
    aligned_predictions_file = os.path.join(cfg.output_dir, "predictions_aligned.pt")
    torch.save(padded_aligned_predictions, aligned_predictions_file)
    print(f"\nSaved aligned predictions to {aligned_predictions_file}")

    # Compute metrics for the aligned predictions
    print("\nComputing metrics for aligned predictions...")
    metrics_aligned = compute_tablature_accuracy(
        predictions=padded_aligned_predictions,
        targets=targets,
        output_vocab=output_vocab,
        pad_id=output_vocab.pad_id
    )

    print(f"\nResults (After Post-Processing):")
    print(f"  Token Accuracy:  {metrics_aligned.token_accuracy:.2%}")
    print(f"  Pitch Accuracy:  {metrics_aligned.pitch_accuracy:.2%}")
    print(f"  Tab Accuracy:    {metrics_aligned.tab_accuracy:.2%}")
    print(f"  Total Tokens:    {metrics_aligned.total_tokens:,}")
    print(f"  Total Notes:     {metrics_aligned.total_notes:,}")

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()