"""
Model definitions for Fretting-Transformer.

Custom T5 configuration based on paper specifications.
"""

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from typing import Optional


def create_model(
    input_vocab_size: int,
    output_vocab_size: int,
    d_model: int = 128,
    d_ff: int = 1024,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    pretrained: bool = False
) -> T5ForConditionalGeneration:
    """
    Create custom T5 model for guitar tablature transcription.

    Args:
        input_vocab_size: Size of input vocabulary
        output_vocab_size: Size of output vocabulary
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_layers: Number of encoder/decoder layers
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        pretrained: Whether to use pretrained weights (not used for custom config)

    Returns:
        T5ForConditionalGeneration model
    """
    # Create custom T5 configuration
    config = T5Config(
        vocab_size=input_vocab_size,  # Encoder vocabulary
        decoder_start_token_id=1,  # BOS token
        eos_token_id=2,  # EOS token
        pad_token_id=0,  # PAD token
        d_model=d_model,
        d_kv=d_model // num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        tie_word_embeddings=False,  # Don't tie encoder/decoder embeddings
    )

    # Initialize model from scratch
    model = T5ForConditionalGeneration(config)

    # Resize decoder token embeddings if needed (different vocab for output)
    if output_vocab_size != input_vocab_size:
        model.resize_token_embeddings(input_vocab_size)
        # Manually set decoder embedding size
        model.decoder.embed_tokens = nn.Embedding(
            output_vocab_size,
            d_model,
            padding_idx=0
        )
        model.lm_head = nn.Linear(d_model, output_vocab_size, bias=False)

    print(f"Created custom T5 model:")
    print(f"  Encoder vocab: {input_vocab_size}")
    print(f"  Decoder vocab: {output_vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  layers: {num_layers}")
    print(f"  heads: {num_heads}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


class FrettingTransformer(nn.Module):
    """
    Wrapper around T5 for guitar tablature transcription.

    Handles input/output formatting and provides clean interface.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        model_config: dict
    ):
        """
        Initialize Fretting-Transformer.

        Args:
            input_vocab_size: Size of input vocabulary
            output_vocab_size: Size of output vocabulary
            model_config: Model configuration dict
        """
        super().__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        # Create underlying T5 model
        self.model = create_model(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            d_model=model_config.get('d_model', 128),
            d_ff=model_config.get('d_ff', 1024),
            num_layers=model_config.get('num_layers', 3),
            num_heads=model_config.get('num_heads', 4),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            pretrained=model_config.get('pretrained', False)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through T5 encoder-decoder model.

        Args:
            input_ids: [B, L_enc] - Encoder input token IDs
            attention_mask: [B, L_enc] - Encoder padding mask (1=real, 0=pad)
            decoder_input_ids: [B, L_dec] - Decoder input token IDs (for teacher forcing)
            decoder_attention_mask: [B, L_dec] - Decoder padding mask (1=real, 0=pad)
                                     Note: Causal masking is applied automatically by T5
            labels: [B, L_dec] - Target labels for loss computation

        Returns:
            Model outputs (Seq2SeqLMOutput):
            - loss: scalar - Cross-entropy loss (if labels provided)
            - logits: [B, L_dec, output_vocab_size] - Output predictions
            - past_key_values: Cached key/value states for generation
            - encoder_last_hidden_state: [B, L_enc, d_model] - Final encoder states
            - decoder_hidden_states: Tuple of [B, L_dec, d_model] for each layer

        Note:
            - Encoder processes input_ids with bidirectional attention
            - Decoder attends to encoder outputs (cross-attention) and uses causal self-attention
            - Output vocab size differs from input vocab size (886 vs 760)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        start_token_id: Optional[torch.Tensor] = None,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        verbose: bool = False,
        **kwargs  # Ignore unused HF params (num_beams, etc.)
    ):
        """
        Custom autoregressive generation with teacher forcing support.

        Args:
            input_ids: [B, L_enc] - Encoder input
            attention_mask: [B, L_enc] - Encoder mask
            max_length: Maximum generation length
            start_token_id: [B] - First decoder token per sample
                           If None, uses decoder_start_token_id=1 (BOS)
            eos_token_id: Token ID to stop generation
            pad_token_id: Padding token ID
            temperature: Sampling temperature (1.0 = greedy)
            verbose: Print generation progress

        Returns:
            [B, L_gen] - Generated sequences (includes start token)

        Note:
            - Encodes input once and reuses encoder outputs (efficient)
            - Supports per-sample teacher forcing via start_token_id
            - Uses greedy decoding (beam search not yet implemented)
            - Properly handles EOS tokens and pads finished sequences
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Encode input once
        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Initialize decoder with start tokens (per-sample or BOS)
        if start_token_id is None:
            start_token_id = torch.full((batch_size,), 1, dtype=torch.long, device=device)  # BOS

        decoder_input_ids = start_token_id.unsqueeze(1)  # [B, 1]

        generated_tokens = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Autoregressive generation
        for step in range(max_length):
            with torch.no_grad():
                decoder_attention_mask = torch.ones_like(decoder_input_ids)

                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    attention_mask=attention_mask,
                )

                # Get logits for last position
                logits = outputs.logits[:, -1, :]  # [B, vocab_size]

                # Temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature

                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1)  # [B]

                # Mark finished sequences
                finished = finished | (next_token == eos_token_id)

                # Replace tokens in finished sequences with pad
                next_token = torch.where(finished, pad_token_id, next_token)

                generated_tokens.append(next_token)

                # Stop if all sequences finished
                if finished.all():
                    if verbose:
                        print(f"All sequences finished at step {step}")
                    break

                # Append to decoder input
                decoder_input_ids = torch.cat([
                    decoder_input_ids,
                    next_token.unsqueeze(1)
                ], dim=1)

        # Stack generated tokens
        generated_sequence = torch.stack(generated_tokens, dim=1)  # [B, L]

        # Prepend start tokens
        full_sequence = torch.cat([start_token_id.unsqueeze(1), generated_sequence], dim=1)

        return full_sequence
