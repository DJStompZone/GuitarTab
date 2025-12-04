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
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Input attention mask
            decoder_input_ids: Decoder input token IDs (for teacher forcing)
            decoder_attention_mask: Decoder attention mask
            labels: Target labels for loss computation

        Returns:
            Model outputs with loss and logits
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
        Generate output sequences.

        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
