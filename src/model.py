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
        vocab_size=output_vocab_size,  # Decoder vocab
        decoder_start_token_id=1,      # BOS token
        eos_token_id=2,                # EOS token
        pad_token_id=0,                # PAD token
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
    
    # Replace encoder embedding
    model.encoder.embed_tokens = nn.Embedding(
        input_vocab_size,
        d_model,
        padding_idx=0
    )
    
    # Replace decoder embedding
    model.decoder.embed_tokens = nn.Embedding(
        output_vocab_size,
        d_model,
        padding_idx=0
    )

    # Replace LM head
    model.lm_head = nn.Linear(d_model, output_vocab_size, bias=False)

    # Weight tying (decoder embed ↔ lm_head)
    model.lm_head.weight = model.decoder.embed_tokens.weight

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
        logits_processor=None,
        **kwargs
    ):
        """
        Generate output sequences autoregressively (for inference).

        Args:
            input_ids: [B, L_enc] - Encoder input token IDs
            attention_mask: [B, L_enc] - Encoder padding mask
            max_length: Maximum generation length (absolute, not relative)
            start_token_id: [B] optional first token (teacher forcing friendly)
            eos_token_id: end token
            pad_token_id: fill for finished seq
            temperature: softmax temperature
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

        device = input_ids.device
        batch_size = input_ids.size(0)

        # Encode input once
        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            enc_hidden = encoder_outputs.last_hidden_state

        # Initialize decoder with start tokens (per-sample or BOS)
        if start_token_id is None:
            # default: BOS = id 1
            start_token_id = torch.full((batch_size,), 1, dtype=torch.long, device=device)


        # Only 1 token as initial input
        decoder_input = start_token_id.unsqueeze(1)    # [B, 1]

        # storage for output tokens
        generated = [start_token_id]   # list of length L, each [B]

        # all unfinished initially
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Sync constrained-decoding state with decoder prefix token.
        if logits_processor is not None:
            if batch_size == 1 and not hasattr(logits_processor, "processors"):
                logits_processor.update_state(start_token_id[0].item())
            else:
                logits_processor.update_state(start_token_id)

        past_key_values = None

        # Autoregressive loop
        for step in range(max_length):

            with torch.no_grad():
                # decoder_input contains ONLY the newest token each step after first
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input,                # [B, 1]
                    encoder_hidden_states=enc_hidden,       # [B, L_enc, d]
                    encoder_attention_mask=attention_mask,  # [B, L_enc]
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                # update KV cache
                past_key_values = decoder_outputs.past_key_values

                # logits for current step token only
                hidden = decoder_outputs.last_hidden_state     # [B, 1, d_model]
                logits = self.model.lm_head(hidden[:, -1, :])  # [B, vocab]

                if temperature != 1.0:
                    logits = logits / temperature

                if logits_processor is not None:
                    dec_ids = torch.stack(generated, dim=1)  # [B, L]
                    logits = logits_processor(dec_ids, logits)

                next_token = torch.argmax(logits, dim=-1)  # [B]

                if logits_processor is not None:
                    if batch_size == 1 and not hasattr(logits_processor, "processors"):
                        logits_processor.update_state(next_token[0].item())
                    else:
                        logits_processor.update_state(next_token)

            # apply EOS
            finished |= (next_token == eos_token_id)
            next_token = torch.where(finished, pad_token_id, next_token)

            generated.append(next_token)

            # Early stop if all done
            if finished.all():
                if verbose:
                    print(f"[generate] Finished at step {step}.")
                break

            # next decoder input is just this newly predicted token
            decoder_input = next_token.unsqueeze(1)   # shape [B, 1]

        # Stack outputs
        generated = torch.stack(generated, dim=1)  # [B, L_gen]
        return generated
