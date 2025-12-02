import pickle
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Union
import sys
import hashlib
import os

from src.utils.midi_utils import extract_events_from_midi
from src.utils.tokenizer import MIDITokenizer


class MIDIDataset(Dataset):
    """
    Dataset for MIDI music generation.
    Converts MIDI files to sequences of event tokens.
    """

    def __init__(
        self,
        midi_paths: List[str],
        dictionary_path: str = None,
        sequence_length: int = 1024,
        use_chords: bool = False,
        cache_dir: str = ".cache/midi_events",
        use_cache: bool = True,
        tokenizer_type: str = "remi",
        **tokenizer_params
    ):
        """
        Initialize the MIDI dataset.

        Args:
            midi_paths: List of paths to MIDI files
            dictionary_path: Path to the event2word dictionary pickle file (required for REMI)
            sequence_length: Length of input sequences
            use_chords: Whether to include chord information
            cache_dir: Directory to store cached parsed MIDI events
            use_cache: Whether to use caching for parsed MIDI events
            tokenizer_type: Type of tokenizer to use ("remi", "cpword", "midilike")
            **tokenizer_params: Additional parameters for miditok tokenizers
        """
        self.midi_paths = midi_paths
        self.sequence_length = sequence_length
        self.use_chords = use_chords
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.tokenizer_type = tokenizer_type

        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = MIDITokenizer(
            tokenizer_type=tokenizer_type,
            dictionary_path=dictionary_path,
            use_chords=use_chords,
            **tokenizer_params
        )

        # For backward compatibility
        if tokenizer_type == "remi":
            self.event2word = self.tokenizer.event2word
            self.word2event = self.tokenizer.word2event

        # Prepare data segments
        self.segments = self._prepare_data()

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    def _get_cache_path(self, midi_path_str: str) -> Path:
        """
        Generate cache file path for a MIDI file.

        Cache key includes: file path, mtime, use_chords flag, and tokenizer type
        This ensures cache is invalidated when file or tokenizer changes.
        """
        midi_path = Path(midi_path_str)
        mtime = os.path.getmtime(midi_path)

        # Create cache key from path, mtime, use_chords, and tokenizer_type
        cache_key = f"{midi_path.absolute()}_{mtime}_{self.use_chords}_{self.tokenizer_type}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

        return self.cache_dir / f"{cache_hash}.pkl"

    def _load_cached_events(self, midi_path: str):
        """Load cached events if available and valid."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(midi_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                # Cache corrupted - delete it and regenerate
                print(f"Cache corrupted for {midi_path}, regenerating...")
                cache_path.unlink()  # Delete corrupted cache
                return None
        return None

    def _save_cached_events(self, midi_path: str, events):
        """Save parsed events to cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(midi_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(events, f)
        except Exception as e:
            # Cache save failed - not critical, just skip caching for this file
            # Don't print warning as this could flood output
            pass

    def _prepare_data(self) -> np.ndarray:
        """
        Tokenize all MIDI files and create training segments.

        Returns:
            Array of shape (N, 2, sequence_length) where N is number of segments
            Each segment contains [input_sequence, target_sequence]
        """
        # Tokenize all MIDI files (with caching for REMI)
        all_words = []
        cache_hits = 0
        cache_misses = 0

        for path in tqdm(self.midi_paths, desc=f"Tokenizing MIDI ({self.tokenizer_type})"):
            if self.tokenizer_type == "remi":
                # Use event caching for REMI (existing approach)
                events = self._load_cached_events(path)

                if events is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    events = extract_events_from_midi(path, use_chords=self.use_chords)
                    self._save_cached_events(path, events)

                # Convert events to words
                words = []
                for event in events:
                    e = f"{event.name}_{event.value}"
                    if e in self.event2word:
                        words.append(self.event2word[e])
                    else:
                        if event.name == "Note Velocity":
                            # Fallback for out-of-range velocities
                            words.append(self.event2word.get("Note Velocity_21", 0))
                        else:
                            # Unknown event - this is a real error
                            raise ValueError(
                                f"Unknown event '{e}' not in vocabulary!\n"
                                f"File: {path}\n"
                                f"Tokenizer: {self.tokenizer_type}, use_chords={self.use_chords}\n"
                                f"This usually means wrong dictionary or use_chords mismatch.\n"
                                f"Run: python build_vocabularies.py"
                            )
            else:
                # Use miditok tokenizers (CPWord, MIDILike)
                # No caching needed as miditok is fast
                words = self.tokenizer.encode_midi(path)

            all_words.append(words)

        if self.use_cache and self.tokenizer_type == "remi":
            print(f"Cache stats: {cache_hits} hits, {cache_misses} misses ({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)")

        # Create segments of (input, target) pairs
        segments = []
        is_compound = (len(all_words) > 0 and len(all_words[0]) > 0 and
                      isinstance(all_words[0][0], list))

        for words in all_words:
            pairs = []
            for i in range(
                0, len(words) - self.sequence_length - 1, self.sequence_length
            ):
                x = words[i : i + self.sequence_length]
                y = words[i + 1 : i + self.sequence_length + 1]
                pairs.append([x, y])

            # Abandon last segments in a MIDI file (align to multiples of 5)
            pairs = pairs[: len(pairs) - (len(pairs) % 5)]
            segments.extend(pairs)

        # Convert to numpy array
        # For compound tokens (CPWord/MIDILike), shape will be (N, 2, seq_len, n_vocabs)
        # For single tokens (REMI), shape will be (N, 2, seq_len)
        if segments:
            segments = np.array(segments, dtype=object if is_compound else np.int64)
            # Force conversion to proper numeric array if possible
            if is_compound:
                try:
                    segments = np.array(segments.tolist())  # Convert from object to numeric
                except:
                    pass
        else:
            segments = np.array(segments)

        print(f"Dataset created with {len(segments)} segments, shape: {segments.shape}")

        return segments
