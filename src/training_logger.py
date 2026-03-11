"""
Training logger for tracking metrics and saving results.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


def plot_loss_curve(logger: "TrainingLogger", output_dir: Path) -> None:
    """
    Plot train/val loss curve and save to output_dir/loss_curve.png.
    Called on each checkpoint save and at training end.
    """
    history = logger.history
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not epochs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "b-o", label="Train loss", markersize=4)
    ax.plot(epochs, val_loss, "r-s", label="Val loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


class TrainingLogger:
    """Tracks training metrics and saves to JSON."""

    def __init__(self, log_file: Path):
        """
        Initialize training logger.

        Args:
            log_file: Path to JSON log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': None,
            'ar_eval': []  # List of dicts with epoch, metrics, and generated tokens
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float
    ):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
        """
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))

        self._save()

    def log_ar_eval(
        self,
        epoch: int,
        token_accuracy: float,
        pitch_accuracy: float,
        tab_accuracy: float,
        total_tokens: int,
        total_notes: int,
        generated_samples: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Log autoregressive evaluation results.

        Args:
            epoch: Epoch number
            token_accuracy: Token-level accuracy
            pitch_accuracy: Pitch accuracy
            tab_accuracy: Tab accuracy
            total_tokens: Total number of tokens
            total_notes: Total number of notes
            generated_samples: List of generated token sequences
        """
        ar_result = {
            'epoch': epoch,
            'metrics': {
                'token_accuracy': float(token_accuracy),
                'pitch_accuracy': float(pitch_accuracy),
                'tab_accuracy': float(tab_accuracy),
                'total_tokens': int(total_tokens),
                'total_notes': int(total_notes)
            }
        }

        if generated_samples is not None:
            ar_result['generated_samples'] = generated_samples

        self.history['ar_eval'].append(ar_result)

        self._save()

    def log_test(self, test_loss: float):
        """
        Log final test loss.

        Args:
            test_loss: Test set loss
        """
        self.history['test_loss'] = float(test_loss)
        self._save()

    def log_test_ar_eval(
        self,
        token_accuracy: float,
        pitch_accuracy: float,
        tab_accuracy: float,
        total_tokens: int,
        total_notes: int
    ):
        """
        Log final test set AR evaluation.

        Args:
            token_accuracy: Token-level accuracy
            pitch_accuracy: Pitch accuracy
            tab_accuracy: Tab accuracy
            total_tokens: Total number of tokens
            total_notes: Total number of notes
        """
        if 'test_ar_eval' not in self.history:
            self.history['test_ar_eval'] = {}

        self.history['test_ar_eval'] = {
            'token_accuracy': float(token_accuracy),
            'pitch_accuracy': float(pitch_accuracy),
            'tab_accuracy': float(tab_accuracy),
            'total_tokens': int(total_tokens),
            'total_notes': int(total_notes)
        }

        self._save()

    def _save(self):
        """Save history to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self):
        """Load history from JSON file if it exists."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.history = json.load(f)


def save_generated_samples(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_vocab,
    output_file: Path,
    max_samples: int = 10
):
    """
    Save generated token sequences to JSON.

    Args:
        predictions: [B, L] - Predicted token IDs
        targets: [B, L] - Target token IDs
        output_vocab: Output vocabulary
        output_file: Path to save JSON
        max_samples: Maximum number of samples to save
    """
    samples = []

    for i in range(min(len(predictions), max_samples)):
        pred_ids = predictions[i].tolist()
        target_ids = targets[i].tolist()

        # Convert IDs to tokens
        pred_tokens = [output_vocab.id_to_token.get(idx, f'<UNK_{idx}>') for idx in pred_ids]
        target_tokens = [output_vocab.id_to_token.get(idx, f'<UNK_{idx}>') for idx in target_ids]

        # Remove padding
        pad_id = output_vocab.pad_id
        pred_tokens = [tok for tok, idx in zip(pred_tokens, pred_ids) if idx != pad_id]
        target_tokens = [tok for tok, idx in zip(target_tokens, target_ids) if idx != pad_id]

        samples.append({
            'sample_id': i,
            'predictions': pred_tokens[:100],  # Limit length for readability
            'targets': target_tokens[:100],
            'pred_length': len(pred_tokens),
            'target_length': len(target_tokens)
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
