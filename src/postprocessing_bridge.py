"""
Post-Processing Bridge for Fretting-Transformer

Bridges between the inference pipeline (tensor IDs) and the fretting_postprocessor
module (token strings). Handles token format conversion and batch processing.
"""

import re
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence
from fretting_postprocessor import FrettingPostProcessor, GuitarConfig


class PostProcessingBridge:
    """
    Bridge between inference pipeline and fretting_postprocessor.

    Handles conversion between:
    - Tensor token IDs (inference) <-> Token strings (post-processor)
    - Dataset format (NOTE_ON_60) <-> Post-processor format (NOTE_ON<60>)
    - Batch processing with padding
    """

    def __init__(self, input_vocab, output_vocab, guitar_config: GuitarConfig):
        """
        Initialize bridge with vocabularies and guitar configuration.

        Args:
            input_vocab: Input vocabulary (NOTE_ON, NOTE_OFF, TIME_SHIFT)
            output_vocab: Output vocabulary (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB)
            guitar_config: Guitar configuration for post-processor
        """
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.processor = FrettingPostProcessor(guitar_config)

    def ids_to_token_strings(self, ids: torch.Tensor, vocab) -> List[str]:
        """
        Convert tensor IDs to token strings (post-processor format).

        Converts dataset format to post-processor format:
        - NOTE_ON_60 → NOTE_ON<60>
        - TAB_3_5 → TAB<3,5>
        - TIME_SHIFT_480 → TIME_SHIFT<480>

        Args:
            ids: Tensor of token IDs [L]
            vocab: Vocabulary object

        Returns:
            List of token strings in post-processor format
        """
        tokens = []

        for idx in ids:
            if idx == vocab.pad_id:
                continue  # Skip padding tokens

            token_str = vocab.id_to_token[idx.item()]

            # Convert format: NOTE_ON_60 → NOTE_ON<60>
            if token_str.startswith(("NOTE_ON_", "NOTE_OFF_")):
                prefix, pitch = token_str.rsplit("_", 1)
                tokens.append(f"{prefix}<{pitch}>")
            elif token_str.startswith("TIME_SHIFT_"):
                _, shift = token_str.rsplit("_", 1)
                tokens.append(f"TIME_SHIFT<{shift}>")
            elif token_str.startswith("TAB_"):
                _, string, fret = token_str.split("_")
                # Convert from 1-indexed (dataset) to 0-indexed (post-processor)
                string_0idx = int(string) - 1
                tokens.append(f"TAB<{string_0idx},{fret}>")
            else:
                # Special tokens: PAD, BOS, EOS, UNK
                tokens.append(token_str)

        return tokens

    def token_strings_to_ids(self, tokens: List[str], vocab) -> torch.Tensor:
        """
        Convert token strings to tensor IDs (dataset format).

        Converts post-processor format back to dataset format:
        - NOTE_ON<60> → NOTE_ON_60
        - TAB<3,5> → TAB_3_5
        - TIME_SHIFT<480> → TIME_SHIFT_480

        Args:
            tokens: List of token strings in post-processor format
            vocab: Vocabulary object

        Returns:
            Tensor of token IDs [L]
        """
        ids = []

        for token in tokens:
            # Convert format: NOTE_ON<60> → NOTE_ON_60
            if match := re.match(r'(NOTE_ON|NOTE_OFF)<(\d+)>', token):
                token_str = f"{match.group(1)}_{match.group(2)}"
            elif match := re.match(r'TIME_SHIFT<(\d+)>', token):
                token_str = f"TIME_SHIFT_{match.group(1)}"
            elif match := re.match(r'TAB<(\d+),(\d+)>', token):
                # Convert from 0-indexed (post-processor) to 1-indexed (dataset)
                string_1idx = int(match.group(1)) + 1
                fret = match.group(2)
                token_str = f"TAB_{string_1idx}_{fret}"
            else:
                # Special tokens or unmatched patterns
                token_str = token

            token_id = vocab.token_to_id.get(token_str, vocab.unk_id)
            ids.append(token_id)

        return torch.tensor(ids, dtype=torch.long)

    def process_batch(
        self,
        input_ids: torch.Tensor,
        predictions: torch.Tensor,
        method: str = 'neighbor_search'
    ) -> torch.Tensor:
        """
        Process a batch of predictions through post-processing.

        Workflow:
        1. For each sequence in batch:
           a. Convert IDs to token strings
           b. Apply post-processing (overlap correction / neighbor search)
           c. Convert back to IDs
        2. Pad to uniform length
        3. Ensure output shape matches input

        Args:
            input_ids: Input tensor [B, L_in]
            predictions: Prediction tensor [B, L_out]
            method: Post-processing method ('overlap' or 'neighbor_search')

        Returns:
            Post-processed predictions tensor [B, L_out]
        """
        B, L = predictions.shape
        postprocessed_batch = []

        for i in range(B):
            # Convert to token strings
            input_tokens = self.ids_to_token_strings(input_ids[i], self.input_vocab)
            pred_tokens = self.ids_to_token_strings(predictions[i], self.output_vocab)

            # Apply post-processing
            try:
                corrected_tokens = self.processor.process_tokens(
                    model_output_tokens=pred_tokens,
                    input_note_tokens=input_tokens,
                    method=method,
                    output_format='auto'  # 自動檢測並保持原格式
                )
            except Exception as e:
                print(f"Warning: Post-processing failed for sequence {i}: {e}")
                corrected_tokens = pred_tokens  # Fallback to original

            # Convert back to IDs
            corrected_ids = self.token_strings_to_ids(corrected_tokens, self.output_vocab)
            postprocessed_batch.append(corrected_ids)

        # Pad to uniform length
        postprocessed_predictions = pad_sequence(
            postprocessed_batch,
            batch_first=True,
            padding_value=self.output_vocab.pad_id
        ).to(predictions.device)  # Move to correct device immediately

        # Ensure output shape matches input shape [B, L]
        if postprocessed_predictions.shape[1] < L:
            # Pad right
            padding = torch.full(
                (B, L - postprocessed_predictions.shape[1]),
                self.output_vocab.pad_id,
                dtype=postprocessed_predictions.dtype,
                device=predictions.device
            )
            postprocessed_predictions = torch.cat([postprocessed_predictions, padding], dim=1)
        elif postprocessed_predictions.shape[1] > L:
            # Trim
            postprocessed_predictions = postprocessed_predictions[:, :L]

        # Ensure output is on the same device as input
        postprocessed_predictions = postprocessed_predictions.to(predictions.device)

        return postprocessed_predictions
