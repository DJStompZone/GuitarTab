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
        num_beams: int = 1,
        **kwargs
    ):
        """
        Generate output sequences autoregressively (for inference).

        Args:
            input_ids: [B, L_enc] - Encoder input token IDs
            attention_mask: [B, L_enc] - Encoder padding mask
            max_length: Maximum generation length (absolute, not relative)
            num_beams: Number of beams for beam search (1 = greedy decoding)
            **kwargs: Additional generation arguments (temperature, top_k, top_p, etc.)

        Returns:
            [B * num_beams, L_gen] - Generated token IDs
            where L_gen <= max_length

        Note:
            - Uses causal decoding: generates one token at a time
            - Starts with BOS token (decoder_start_token_id)
            - Stops at EOS token or max_length
            - For beam search (num_beams > 1), batch dimension expands
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
