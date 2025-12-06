"""
Evaluation metrics for guitar tablature transcription.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from tqdm import tqdm


@dataclass
class TabAccuracyMetrics:
    """Container for tablature accuracy metrics."""

    # Token-level accuracy
    token_accuracy: float  # % of all tokens correct

    # Note-level accuracy (only on NOTE_ON positions)
    pitch_accuracy: float  # % of NOTE_ON with correct pitch
    tab_accuracy: float  # % of NOTE_ON with correct (string, fret)

    # Counts
    total_tokens: int
    total_notes: int

    def __str__(self) -> str:
        return (
            f"Token Acc: {self.token_accuracy:.2%} | "
            f"Pitch Acc: {self.pitch_accuracy:.2%} | "
            f"Tab Acc: {self.tab_accuracy:.2%} | "
            f"({self.total_notes} notes, {self.total_tokens} tokens)"
        )


@dataclass
class PostProcessingMetrics:
    """Container for post-processing comparison metrics."""

    # Raw model metrics
    raw_token_accuracy: float
    raw_pitch_accuracy: float
    raw_tab_accuracy: float

    # Post-processed metrics
    post_token_accuracy: float
    post_pitch_accuracy: float
    post_tab_accuracy: float

    # Improvements
    token_improvement: float
    pitch_improvement: float
    tab_improvement: float

    # Counts
    total_tokens: int
    total_notes: int

    # Method used
    method: str

    def __str__(self) -> str:
        return (
            f"Method: {self.method}\n"
            f"Raw:  Token {self.raw_token_accuracy:.2%} | "
            f"Pitch {self.raw_pitch_accuracy:.2%} | "
            f"Tab {self.raw_tab_accuracy:.2%}\n"
            f"Post: Token {self.post_token_accuracy:.2%} | "
            f"Pitch {self.post_pitch_accuracy:.2%} | "
            f"Tab {self.post_tab_accuracy:.2%}\n"
            f"Improvement: Token {self.token_improvement:+.2%} | "
            f"Pitch {self.pitch_improvement:+.2%} | "
            f"Tab {self.tab_improvement:+.2%}"
        )


def compute_tablature_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, output_vocab, pad_id: int = 0
) -> TabAccuracyMetrics:
    """
    Compute accuracy metrics for tablature transcription.

    Args:
        predictions: [B, L] - Predicted token IDs
        targets: [B, L] - Ground truth token IDs
        output_vocab: Vocabulary object with id_to_token mapping
        pad_id: Padding token ID to ignore

    Returns:
        TabAccuracyMetrics with computed accuracies
    """
    # Flatten tensors (use reshape to handle non-contiguous tensors)
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)

    # Mask out padding
    non_pad_mask = target_flat != pad_id
    pred_flat = pred_flat[non_pad_mask]
    target_flat = target_flat[non_pad_mask]

    total_tokens = len(target_flat)

    # Token-level accuracy (all tokens)
    correct_tokens = (pred_flat == target_flat).sum().item()
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    # Decode tokens to strings
    pred_tokens = [output_vocab.id_to_token[idx.item()] for idx in pred_flat]
    target_tokens = [output_vocab.id_to_token[idx.item()] for idx in target_flat]

    # Note-level accuracy (only NOTE_ON positions)
    note_positions = [
        i for i, token in enumerate(target_tokens) if token.startswith("NOTE_ON_")
    ]

    if len(note_positions) == 0:
        return TabAccuracyMetrics(
            token_accuracy=token_accuracy,
            pitch_accuracy=0.0,
            tab_accuracy=0.0,
            total_tokens=total_tokens,
            total_notes=0,
        )

    # Pitch accuracy: check if pitch matches
    pitch_correct = 0
    for pos in note_positions:
        pred_token = pred_tokens[pos]
        target_token = target_tokens[pos]
        if pred_token == target_token:
            pitch_correct += 1

    pitch_accuracy = pitch_correct / len(note_positions)

    # Tab accuracy: check if following TAB token matches
    # NOTE_ON is always followed by TAB in output sequence
    tab_correct = 0
    tab_positions = []

    for pos in note_positions:
        # Check if next token is TAB
        if pos + 1 < len(target_tokens):
            target_next = target_tokens[pos + 1]
            if target_next.startswith("TAB_"):
                tab_positions.append(pos + 1)
                pred_tab = pred_tokens[pos + 1]
                if pred_tab == target_next:
                    tab_correct += 1

    tab_accuracy = tab_correct / len(tab_positions) if tab_positions else 0.0

    return TabAccuracyMetrics(
        token_accuracy=token_accuracy,
        pitch_accuracy=pitch_accuracy,
        tab_accuracy=tab_accuracy,
        total_tokens=total_tokens,
        total_notes=len(note_positions),
    )


def generate_and_compute_accuracy(
    model,
    dataloader,
    output_vocab,
    device: str,
    max_length: int = 1024,
    num_beams: int = 1,  # DEPRECATED (kept for config compat)
    max_batches: Optional[int] = None,
    use_teacher_forcing: bool = True,  # NEW
    temperature: float = 1.0,  # NEW
) -> tuple[TabAccuracyMetrics, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate sequences autoregressively and compute accuracy.

    Args:
        model: FrettingTransformer model
        dataloader: DataLoader for validation/test set
        output_vocab: Output vocabulary
        device: Device to run on
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
        max_batches: Limit number of batches (for speed)
        return_predictions: If True, return (metrics, (predictions, targets))

    Returns:
        (metrics, (input_ids, targets, predictions))
    """
    model.eval()

    all_predictions = []
    all_input_ids = []
    all_targets = []

    # Determine total batches for progress bar
    total_batches = max_batches if max_batches is not None else len(dataloader)
    print("max_batches", max_batches)

    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader), total=total_batches, desc="Generating sequences"
        )
        for batch_idx, batch in pbar:
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["output_ids"].to(device)

            # Extract first target token per sample for teacher forcing
            if use_teacher_forcing:
                start_token_id = target_ids[:, 0]  # [B] - per-sample first tokens
            else:
                start_token_id = None  # Will use BOS=1

            # Generate (now uses custom implementation)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                start_token_id=start_token_id,
                eos_token_id=output_vocab.eos_id if hasattr(output_vocab, "eos_id") else 2,
                pad_token_id=output_vocab.pad_id,
                temperature=temperature,
                verbose=False
            )

            # Pad/trim both generated and targets to max_length for uniform shape
            B = target_ids.shape[0]
            gen_len = generated.shape[1]
            target_len = target_ids.shape[1]

            # Pad or trim generated to max_length
            if gen_len < max_length:
                padding = torch.full(
                    (B, max_length - gen_len),
                    output_vocab.pad_id,
                    dtype=generated.dtype,
                    device=generated.device,
                )
                generated = torch.cat([generated, padding], dim=1)
            elif gen_len > max_length:
                generated = generated[:, :max_length]

            # Pad or trim targets to max_length
            if target_len < max_length:
                padding = torch.full(
                    (B, max_length - target_len),
                    output_vocab.pad_id,
                    dtype=target_ids.dtype,
                    device=target_ids.device,
                )
                target_ids = torch.cat([target_ids, padding], dim=1)
            elif target_len > max_length:
                target_ids = target_ids[:, :max_length]

            # Note: input_ids also need padding for uniform concatenation
            # However, we keep them as-is since they don't need to match max_length
            # (they're already padded within batch by collate_fn)

            all_input_ids.append(input_ids)
            all_predictions.append(generated)
            all_targets.append(target_ids)

            # Update progress bar with current batch info
            pbar.set_postfix(
                {"batch_size": B, "gen_len": gen_len, "target_len": target_len}
            )

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Pad input_ids to uniform length across batches
    max_input_len = max(inp.shape[1] for inp in all_input_ids)
    padded_inputs = []
    for inp in all_input_ids:
        if inp.shape[1] < max_input_len:
            # Assume input uses pad_id=0 (standard for input vocab)
            padding = torch.zeros(
                (inp.shape[0], max_input_len - inp.shape[1]),
                dtype=inp.dtype,
                device=inp.device
            )
            inp = torch.cat([inp, padding], dim=1)
        padded_inputs.append(inp)
    input_ids = torch.cat(padded_inputs, dim=0)

    # Compute metrics
    metrics = compute_tablature_accuracy(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        pad_id=output_vocab.pad_id,
    )

    return metrics, (input_ids, targets, predictions)


def compute_postprocessing_metrics(
    raw_predictions: torch.Tensor,
    postprocessed_predictions: torch.Tensor,
    targets: torch.Tensor,
    output_vocab,
    method: str,
    pad_id: int = 0
) -> PostProcessingMetrics:
    """
    Compute comparison metrics between raw and post-processed predictions.

    Args:
        raw_predictions: Original model predictions [B, L]
        postprocessed_predictions: After post-processing [B, L]
        targets: Ground truth [B, L]
        output_vocab: Output vocabulary
        method: Post-processing method used
        pad_id: Padding token ID

    Returns:
        PostProcessingMetrics with all comparisons
    """
    # Compute metrics for raw predictions
    raw_metrics = compute_tablature_accuracy(
        predictions=raw_predictions,
        targets=targets,
        output_vocab=output_vocab,
        pad_id=pad_id
    )

    # Compute metrics for post-processed predictions
    post_metrics = compute_tablature_accuracy(
        predictions=postprocessed_predictions,
        targets=targets,
        output_vocab=output_vocab,
        pad_id=pad_id
    )

    # Calculate improvements
    token_improvement = post_metrics.token_accuracy - raw_metrics.token_accuracy
    pitch_improvement = post_metrics.pitch_accuracy - raw_metrics.pitch_accuracy
    tab_improvement = post_metrics.tab_accuracy - raw_metrics.tab_accuracy

    return PostProcessingMetrics(
        raw_token_accuracy=raw_metrics.token_accuracy,
        raw_pitch_accuracy=raw_metrics.pitch_accuracy,
        raw_tab_accuracy=raw_metrics.tab_accuracy,
        post_token_accuracy=post_metrics.token_accuracy,
        post_pitch_accuracy=post_metrics.pitch_accuracy,
        post_tab_accuracy=post_metrics.tab_accuracy,
        token_improvement=token_improvement,
        pitch_improvement=pitch_improvement,
        tab_improvement=tab_improvement,
        total_tokens=raw_metrics.total_tokens,
        total_notes=raw_metrics.total_notes,
        method=method
    )
