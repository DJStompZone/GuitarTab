"""
Tokenizer wrapper for different MIDI tokenization schemes.
Supports both dictionary-based (REMI) and miditok-based (CPWord, MIDILike) approaches.

This is the SINGLE SOURCE OF TRUTH for all vocabulary management.
All other code should use this class instead of loading dictionaries directly.
"""

import pickle
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict
from miditok import CPWord, MIDILike, TokenizerConfig
from symusic import Score
import numpy as np


class MIDITokenizer:
    """
    Unified interface for MIDI tokenization.

    This is the single source of truth for vocabulary management.
    All code should use this class instead of loading dictionaries directly.

    Supports:
    - REMI: Dictionary-based (our custom implementation)
    - CPWord: miditok Compound Word
    - MIDILike: miditok MIDI-Like
    """

    # Default dictionary paths
    DEFAULT_DICT_PATH = "./dictionary.pkl"
    DEFAULT_DICT_CHORD_PATH = "./dictionary_chord.pkl"

    def __init__(
        self,
        tokenizer_type: str = "remi",
        dictionary_path: str = None,
        use_chords: bool = False,
        vocab_size: int = None,
        **tokenizer_params
    ):
        """
        Initialize tokenizer.

        Args:
            tokenizer_type: One of ["remi", "cpword", "midilike"]
            dictionary_path: Path to pickle dictionary (for REMI only).
                           If None, auto-selects based on use_chords.
            use_chords: Whether to use chord information
            vocab_size: Vocabulary size (computed from dictionary or tokenizer)
            **tokenizer_params: Additional parameters for miditok tokenizers
        """
        self.tokenizer_type = tokenizer_type.lower()
        self.use_chords = use_chords

        if self.tokenizer_type == "remi":
            # Auto-select dictionary based on use_chords
            if dictionary_path is None:
                dictionary_path = self.DEFAULT_DICT_CHORD_PATH if use_chords else self.DEFAULT_DICT_PATH

            if not Path(dictionary_path).exists():
                raise FileNotFoundError(
                    f"Dictionary not found: {dictionary_path}\n"
                    f"Run 'python build_vocabularies.py' to generate dictionaries."
                )

            self.event2word, self.word2event = pickle.load(open(dictionary_path, "rb"))
            self.vocab_size = len(self.event2word)
            self.tokenizer = None
            self.dictionary_path = dictionary_path

        elif self.tokenizer_type in ["cpword", "midilike"]:
            # miditok-based tokenizers
            config = self._create_miditok_config(use_chords, **tokenizer_params)

            if self.tokenizer_type == "cpword":
                self.tokenizer = CPWord(config)
            elif self.tokenizer_type == "midilike":
                self.tokenizer = MIDILike(config)

            # Store vocabulary information
            if self.tokenizer.is_multi_voc:
                # Multi-vocabulary: store list of vocab sizes
                self.vocab_sizes = [len(vocab) for vocab in self.tokenizer.vocab]
                # Use sum for reporting (not product - that would be sparse!)
                self.vocab_size = self.vocab_sizes  # Return list for multi-voc
                print(f"Multi-vocabulary tokenizer: {self.vocab_sizes} ({sum(self.vocab_sizes)} total)")
            else:
                self.vocab_size = len(self.tokenizer)
                self.vocab_sizes = None

            self.event2word = None
            self.word2event = None
            self.dictionary_path = None

        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    @classmethod
    def from_config(cls, data_config) -> "MIDITokenizer":
        """
        Create tokenizer from Hydra data config.

        This is the recommended way to create tokenizers in training/generation code.

        Args:
            data_config: Hydra config with tokenizer_type and use_chords

        Returns:
            MIDITokenizer instance
        """
        return cls(
            tokenizer_type=data_config.get("tokenizer_type", "remi"),
            use_chords=data_config.get("use_chords", False),
        )

    def _create_miditok_config(self, use_chords: bool, **params) -> TokenizerConfig:
        """Create TokenizerConfig for miditok tokenizers."""
        # Default parameters matching our MIDI setup
        default_params = {
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8, (4, 12): 4},
            "num_velocities": 32,
            "special_tokens": ["PAD", "BOS", "EOS"],
            "use_chords": use_chords,
            "use_rests": False,
            "use_tempos": True,
            "use_time_signatures": False,
            "use_programs": False,
            "num_tempos": 32,
            "tempo_range": (40, 250),
        }

        # Override with user-provided params
        default_params.update(params)

        return TokenizerConfig(**default_params)

    def _flatten_compound_tokens(self, compound_tokens: List[List[int]]) -> List[int]:
        """
        Flatten compound tokens (multi-vocabulary) to single integers.

        CPWord/MIDILike use compound tokens like [3, 4, 3, 3, 3] representing
        values from multiple vocabularies. We flatten to single integers using
        a multi-base encoding scheme.

        Args:
            compound_tokens: List of compound tokens, each is a list of ints

        Returns:
            List of flattened token IDs
        """
        if not compound_tokens or not isinstance(compound_tokens[0], list):
            # Already flat
            return compound_tokens

        # Get vocabulary sizes for each position
        vocab_sizes = [len(vocab) for vocab in self.tokenizer.vocab]

        # Flatten each compound token
        flat_tokens = []
        for compound in compound_tokens:
            # Convert multi-base to single integer
            # flat_id = v0 + v1*s0 + v2*s0*s1 + v3*s0*s1*s2 + ...
            flat_id = 0
            multiplier = 1
            for val, size in zip(compound, vocab_sizes):
                flat_id += val * multiplier
                multiplier *= size
            flat_tokens.append(flat_id)

        return flat_tokens

    def _unflatten_token(self, flat_id: int) -> List[int]:
        """
        Convert a flattened token ID back to compound token.

        Inverse of _flatten_compound_tokens.

        Args:
            flat_id: Flattened token ID

        Returns:
            Compound token as list of ints
        """
        vocab_sizes = [len(vocab) for vocab in self.tokenizer.vocab]
        compound = []

        remaining = flat_id
        for size in vocab_sizes:
            compound.append(remaining % size)
            remaining //= size

        return compound

    def encode_midi(self, midi_path: str) -> List[int]:
        """
        Encode MIDI file to token indices.

        Args:
            midi_path: Path to MIDI file

        Returns:
            List of token indices
        """
        # TODO: bad design
        if self.tokenizer_type == "remi":
            # Use existing event-based encoding
            from src.utils.midi_utils import extract_events_from_midi
            events = extract_events_from_midi(midi_path, use_chords=self.use_chords)

            words = []
            for event in events:
                e = f"{event.name}_{event.value}"
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # Handle OOV
                    if event.name == "Note Velocity":
                        # Fallback for out-of-range velocities
                        words.append(self.event2word.get("Note Velocity_21", 0))
                    else:
                        # Unknown event - this is a real error that should be fixed
                        raise ValueError(
                            f"Unknown event '{e}' not in vocabulary!\n"
                            f"File: {midi_path}\n"
                            f"This usually means:\n"
                            f"  1. Wrong dictionary for tokenizer (use_chords mismatch?)\n"
                            f"  2. Dictionary needs to be rebuilt with build_vocabularies.py\n"
                            f"Vocabulary has {len(self.event2word)} tokens, use_chords={self.use_chords}"
                        )
            return words

        else:
            # Use miditok encoding
            midi = Score(midi_path)
            tokens = self.tokenizer(midi)

            # Convert TokSequence to list of ints
            if hasattr(tokens, 'ids'):
                compound_tokens = tokens.ids
            elif isinstance(tokens, list):
                # Handle multi-track case - concatenate all tracks
                compound_tokens = []
                for track_tokens in tokens:
                    if hasattr(track_tokens, 'ids'):
                        compound_tokens.extend(track_tokens.ids)
                    else:
                        compound_tokens.extend(track_tokens)
            else:
                compound_tokens = list(tokens)

            # Return compound tokens as-is (don't flatten - too sparse!)
            # For multi-voc, returns List[List[int]]
            # For single-voc, returns List[int]
            return compound_tokens

    def decode_tokens(self, tokens: List[int], output_path: str = None) -> Union[Score, None]:
        """
        Decode token indices back to MIDI.

        Args:
            tokens: List of token indices
            output_path: Optional path to save MIDI file

        Returns:
            Score object (for miditok) or None (for REMI with word2event)
        """
        if self.tokenizer_type == "remi":
            # Use existing event-based decoding
            from src.utils.midi_utils import write_midi
            if output_path:
                write_midi(tokens, self.word2event, output_path)
            return None

        else:
            # Use miditok decoding (tokens are already in correct format)
            midi = self.tokenizer(tokens)

            if output_path:
                midi.dump_midi(Path(output_path))

            return midi

    def __len__(self):
        """Return vocabulary size (sum for multi-voc, single int otherwise)."""
        if isinstance(self.vocab_size, list):
            return sum(self.vocab_size)
        return self.vocab_size

    def get_special_token_id(self, token_name: str) -> int:
        """
        Get special token ID.

        Args:
            token_name: One of ["PAD", "BOS", "EOS"]

        Returns:
            Token ID
        """
        if self.tokenizer_type == "remi":
            # REMI doesn't have explicit special tokens in the same way
            # Return 0 for PAD (common convention)
            if token_name == "PAD":
                return 0
            elif token_name == "BOS":
                # Use Bar_None as BOS
                return self.event2word.get("Bar_None", 0)
            elif token_name == "EOS":
                return 0  # No explicit EOS in REMI
            else:
                return 0
        else:
            # miditok tokenizers
            return self.tokenizer[f"{token_name}_None"]
