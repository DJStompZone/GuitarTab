import json
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Union, Optional
import sys
import hashlib
import os

from src.midi_utils import extract_events_from_midi, extract_note_on_off_events, extract_tab_events_from_jams
from src.tokenizer import MIDITokenizer


class MIDIDataset(Dataset):
    """
    Dataset for MIDI music generation and guitar tablature.
    Supports different tokenization schemes including TAB (NOTE ON/OFF -> TAB<string,fret>).
    """

    def __init__(
        self,
        midi_paths: List[str],
        jams_paths: Optional[List[str]] = None,
        dictionary_path: str = None,
        sequence_length: int = 1024,
        use_chords: bool = False,
        cache_dir: str = ".cache/midi_events",
        use_cache: bool = False,  # Disabled by default as requested
        tokenizer_type: str = "tab",
        **tokenizer_params
    ):
        """
        Initialize the MIDI dataset.

        Args:
            midi_paths: List of paths to MIDI files
            jams_paths: List of paths to JAMS files (for TAB tokenizer)
            dictionary_path: Path to the event2word dictionary JSON file (for REMI)
            sequence_length: Length of input sequences
            use_chords: Whether to include chord information
            cache_dir: Directory to store cached parsed MIDI events
            use_cache: Whether to use caching for parsed MIDI events (not recommended)
            tokenizer_type: Type of tokenizer to use ("tab", "remi", "cpword", "midilike")
            **tokenizer_params: Additional parameters for tokenizers
        """
        self.midi_paths = midi_paths
        self.jams_paths = jams_paths
        self.sequence_length = sequence_length
        self.use_chords = use_chords
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.tokenizer_type = tokenizer_type

        # Validate JAMS paths for TAB tokenizer
        if tokenizer_type == "tab" and jams_paths is None:
            raise ValueError("JAMS paths required for TAB tokenizer")

        if jams_paths is not None and len(jams_paths) != len(midi_paths):
            raise ValueError(f"Number of JAMS files ({len(jams_paths)}) must match MIDI files ({len(midi_paths)})")

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

        return self.cache_dir / f"{cache_hash}.json"

    def _load_cached_events(self, midi_path: str):
        """Load cached events if available and valid (not recommended - disabled by default)."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(midi_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                # Cache corrupted - delete it and regenerate
                print(f"Cache corrupted for {midi_path}, regenerating...")
                cache_path.unlink()  # Delete corrupted cache
                return None
        return None

    def _save_cached_events(self, midi_path: str, events):
        """Save parsed events to cache (not recommended - disabled by default)."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(midi_path)
        try:
            # Convert events to JSON-serializable format
            serializable_events = []
            for event in events:
                serializable_events.append({
                    'name': event.name,
                    'time': event.time,
                    'value': str(event.value),
                    'text': event.text
                })

            with open(cache_path, 'w') as f:
                json.dump(serializable_events, f)
        except Exception as e:
            # Cache save failed - not critical, just skip caching for this file
            # Don't print warning as this could flood output
            pass

    def _prepare_data(self) -> np.ndarray:
        """
        Tokenize all MIDI files and create training segments.

        For TAB tokenizer: Returns array of shape (N, 3, sequence_length)
            - segments[i][0] = input sequence (NOTE ON/OFF)
            - segments[i][1] = output sequence (TAB<string,fret>)
            - segments[i][2] = target sequence (shifted output for teacher forcing)

        For other tokenizers: Returns array of shape (N, 2, sequence_length)
            - segments[i][0] = input sequence
            - segments[i][1] = target sequence

        Returns:
            Array of training segments
        """
        # Tokenize all MIDI/JAMS files
        all_input_words = []
        all_output_words = []

        if self.tokenizer_type == "tab":
            # TAB tokenization: process MIDI and JAMS pairs
            desc = f"Tokenizing TAB (MIDI+JAMS)"
            iterator = zip(self.midi_paths, self.jams_paths)

            for midi_path, jams_path in tqdm(list(iterator), desc=desc):
                # Extract input events (NOTE ON/OFF from MIDI)
                input_events = extract_note_on_off_events(midi_path)
                input_words = []
                for event in input_events:
                    e = f"{event.name}_{event.value}"
                    if e in self.tokenizer.input_event2word:
                        input_words.append(self.tokenizer.input_event2word[e])
                    else:
                        # Use UNK token
                        input_words.append(self.tokenizer.input_event2word.get('SPECIAL_UNK', 3))

                # Extract output events (TAB from JAMS)
                output_events = extract_tab_events_from_jams(jams_path)
                output_words = []
                for event in output_events:
                    e = f"{event.name}_{event.value}"
                    if e in self.tokenizer.output_event2word:
                        output_words.append(self.tokenizer.output_event2word[e])
                    else:
                        # Use UNK token
                        output_words.append(self.tokenizer.output_event2word.get('SPECIAL_UNK', 3))

                all_input_words.append(input_words)
                all_output_words.append(output_words)

        elif self.tokenizer_type == "remi":
            # REMI tokenization
            for path in tqdm(self.midi_paths, desc=f"Tokenizing REMI"):
                events = extract_events_from_midi(path, use_chords=self.use_chords)

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
                all_input_words.append(words)
        else:
            # Use miditok tokenizers (CPWord, MIDILike)
            for path in tqdm(self.midi_paths, desc=f"Tokenizing {self.tokenizer_type}"):
                words = self.tokenizer.encode_midi(path)
                all_input_words.append(words)

        # Create segments of (input, target) pairs
        segments = []

        if self.tokenizer_type == "tab":
            # Create segments with separate input and output sequences
            for input_words, output_words in zip(all_input_words, all_output_words):
                # Use minimum length to avoid index errors
                min_len = min(len(input_words), len(output_words))

                for i in range(0, min_len - self.sequence_length - 1, self.sequence_length):
                    x_input = input_words[i : i + self.sequence_length]
                    x_output = output_words[i : i + self.sequence_length]
                    y_output = output_words[i + 1 : i + self.sequence_length + 1]

                    # Each segment: [input_seq, output_seq, target_seq]
                    segments.append([x_input, x_output, y_output])

        else:
            # Create standard autoregressive segments
            is_compound = (len(all_input_words) > 0 and len(all_input_words[0]) > 0 and
                          isinstance(all_input_words[0][0], list))

            for words in all_input_words:
                for i in range(0, len(words) - self.sequence_length - 1, self.sequence_length):
                    x = words[i : i + self.sequence_length]
                    y = words[i + 1 : i + self.sequence_length + 1]
                    segments.append([x, y])

        # Convert to numpy array
        if segments:
            # Try to convert to proper numeric array
            try:
                segments = np.array(segments, dtype=np.int64)
            except:
                # Fall back to object array for compound tokens
                segments = np.array(segments, dtype=object)
        else:
            segments = np.array(segments)

        print(f"Dataset created with {len(segments)} segments, shape: {segments.shape}")

        if self.tokenizer_type == "tab":
            print(f"  Format: [input_seq, output_seq, target_seq]")
            print(f"  Input vocab size: {self.tokenizer.input_vocab_size}")
            print(f"  Output vocab size: {self.tokenizer.output_vocab_size}")

        return segments
