"""
Constrained decoding for guitar tablature generation.

Implements LogitsProcessor classes to enforce valid token sequences at generation time,
ensuring NOTE_ON -> TAB -> NOTE_OFF -> TIME_SHIFT grammar rules.
"""

from typing import List, Optional, Dict, Tuple, Set
import torch
from dataclasses import dataclass

from src.tab_dataset import Vocabulary


# Standard guitar tuning (MIDI pitches for open strings)
# E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
NUM_STRINGS = 6
# num_frets default 25 to match configs/data/*.yaml; pass from config at call sites
DEFAULT_NUM_FRETS = 25


@dataclass
class PitchToTabMapping:
    """Pre-computed mapping from pitch to valid (string, fret) combinations."""
    
    pitch_to_tabs: Dict[int, List[Tuple[int, int]]]
    
    @classmethod
    def build(cls, tuning: List[int] = STANDARD_TUNING, num_frets: int = DEFAULT_NUM_FRETS) -> 'PitchToTabMapping':
        """Build pitch to tab mapping for O(1) lookup."""
        pitch_to_tabs: Dict[int, List[Tuple[int, int]]] = {}
        
        for string_idx, open_pitch in enumerate(tuning):
            string_num = string_idx + 1  # 1-indexed strings
            for fret in range(num_frets):
                pitch = open_pitch + fret
                if pitch not in pitch_to_tabs:
                    pitch_to_tabs[pitch] = []
                pitch_to_tabs[pitch].append((string_num, fret))
        
        return cls(pitch_to_tabs=pitch_to_tabs)
    
    def get_valid_tabs(self, pitch: int) -> List[Tuple[int, int]]:
        """Get all valid (string, fret) combinations for a pitch."""
        return self.pitch_to_tabs.get(pitch, [])


class TablatureLogitsProcessor:
    """
    LogitsProcessor for constrained guitar tablature generation.
    
    Enforces the following grammar rules:
    1. NOTE_ON must be followed by TAB
    2. TAB (string, fret) must produce the correct pitch
    3. NOTE_OFF must close active notes (same pitch)
    4. TIME_SHIFT only allowed after all active notes are closed
    5. NOTE_ON pitch must match the corresponding input pitch
    
    State Machine:
    - START -> NOTE_ON (matching input pitch) or EOS
    - NOTE_ON -> TAB (valid for current pitch)
    - TAB -> NOTE_ON or NOTE_OFF
    - NOTE_OFF -> NOTE_OFF, NOTE_ON, or TIME_SHIFT (if all closed)
    - TIME_SHIFT -> NOTE_ON or EOS
    """
    
    def __init__(
        self,
        output_vocab: Vocabulary,
        input_pitches: List[int],
        tuning: List[int] = STANDARD_TUNING,
        num_frets: int = DEFAULT_NUM_FRETS,
        device: str = 'cpu'
    ):
        """
        Initialize the logits processor.
        
        Args:
            output_vocab: Output vocabulary with token_to_id and id_to_token mappings
            input_pitches: List of expected pitches from encoder input (in order)
            tuning: Guitar tuning (MIDI pitches for open strings)
            num_frets: Number of frets (valid 0..num_frets-1). Should match config (e.g. 25).
            device: Device for tensor operations
        """
        self.vocab = output_vocab
        self.input_pitches = input_pitches
        self.tuning = tuning
        self.device = device
        
        # Pre-compute pitch to tab mapping
        self.pitch_to_tab_mapping = PitchToTabMapping.build(tuning, num_frets)
        
        # Pre-compute token ID sets for efficient masking
        self._precompute_token_sets()
        
        # Initialize state
        self.reset_state()
    
    def _precompute_token_sets(self):
        """Pre-compute token ID sets for common operations."""
        self.note_on_ids: Dict[int, int] = {}  # pitch -> token_id
        self.note_off_ids: Dict[int, int] = {}  # pitch -> token_id
        self.tab_ids: Dict[Tuple[int, int], int] = {}  # (string, fret) -> token_id
        self.time_shift_ids: Set[int] = set()
        
        for token, token_id in self.vocab.token_to_id.items():
            if token.startswith("NOTE_ON_"):
                try:
                    pitch = int(token.split("_")[2])
                    self.note_on_ids[pitch] = token_id
                except (IndexError, ValueError):
                    pass
            elif token.startswith("NOTE_OFF_"):
                try:
                    pitch = int(token.split("_")[2])
                    self.note_off_ids[pitch] = token_id
                except (IndexError, ValueError):
                    pass
            elif token.startswith("TAB_"):
                try:
                    parts = token.split("_")
                    string = int(parts[1])
                    fret = int(parts[2])
                    self.tab_ids[(string, fret)] = token_id
                except (IndexError, ValueError):
                    pass
            elif token.startswith("TIME_SHIFT_"):
                self.time_shift_ids.add(token_id)
        
        # Pre-compute pitch to tab token IDs
        self.pitch_to_tab_token_ids: Dict[int, List[int]] = {}
        for pitch, tabs in self.pitch_to_tab_mapping.pitch_to_tabs.items():
            self.pitch_to_tab_token_ids[pitch] = [
                self.tab_ids[(s, f)] for s, f in tabs if (s, f) in self.tab_ids
            ]
    
    def reset_state(self):
        """Reset internal state for new sequence generation."""
        self.active_notes: List[int] = []  # Stack of active pitches
        self.last_token_type: str = 'START'
        self.last_pitch: Optional[int] = None
        self.pitch_idx: int = 0  # Index into input_pitches
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply constrained decoding mask to logits.
        
        Args:
            input_ids: [seq_len] - Previously generated token IDs
            scores: [vocab_size] - Logits for next token
            
        Returns:
            Modified scores with invalid tokens masked to -inf
        """
        # Build mask based on current state
        mask = self._compute_valid_token_mask(scores.device)
        
        # Apply mask: set invalid tokens to -inf
        scores = scores.masked_fill(~mask, float('-inf'))
        
        return scores
    
    def _compute_valid_token_mask(self, device: torch.device) -> torch.Tensor:
        """Compute boolean mask of valid tokens based on current state."""
        mask = torch.zeros(self.vocab.vocab_size, dtype=torch.bool, device=device)
        
        if self.last_token_type == 'START':
            # Can generate NOTE_ON (matching input pitch) or EOS
            self._add_valid_note_on_tokens(mask)
            mask[self.vocab.eos_id] = True
            
        elif self.last_token_type == 'NOTE_ON':
            # Must generate valid TAB for current pitch
            self._add_valid_tab_tokens(mask)
            
        elif self.last_token_type == 'TAB':
            # Can generate NOTE_ON (more notes in chord) or NOTE_OFF
            self._add_valid_note_on_tokens(mask)
            self._add_valid_note_off_tokens(mask)
            
        elif self.last_token_type == 'NOTE_OFF':
            if len(self.active_notes) == 0:
                # All notes closed: can generate TIME_SHIFT, NOTE_ON, or EOS
                self._add_time_shift_tokens(mask)
                self._add_valid_note_on_tokens(mask)
                mask[self.vocab.eos_id] = True
            else:
                # Still have active notes: can only close more or start new
                self._add_valid_note_off_tokens(mask)
                self._add_valid_note_on_tokens(mask)
                
        elif self.last_token_type == 'TIME_SHIFT':
            # After TIME_SHIFT: can generate NOTE_ON or EOS
            self._add_valid_note_on_tokens(mask)
            mask[self.vocab.eos_id] = True
        
        return mask
    
    def _add_valid_note_on_tokens(self, mask: torch.Tensor):
        """Add valid NOTE_ON tokens to mask based on input pitch sequence."""
        if self.pitch_idx < len(self.input_pitches):
            expected_pitch = self.input_pitches[self.pitch_idx]
            if expected_pitch in self.note_on_ids:
                mask[self.note_on_ids[expected_pitch]] = True
    
    def _add_valid_tab_tokens(self, mask: torch.Tensor):
        """Add valid TAB tokens for the current pitch."""
        if self.last_pitch is not None and self.last_pitch in self.pitch_to_tab_token_ids:
            for token_id in self.pitch_to_tab_token_ids[self.last_pitch]:
                mask[token_id] = True
    
    def _add_valid_note_off_tokens(self, mask: torch.Tensor):
        """Add valid NOTE_OFF tokens for active notes."""
        for pitch in self.active_notes:
            if pitch in self.note_off_ids:
                mask[self.note_off_ids[pitch]] = True
    
    def _add_time_shift_tokens(self, mask: torch.Tensor):
        """Add all TIME_SHIFT tokens to mask."""
        for token_id in self.time_shift_ids:
            mask[token_id] = True
    
    def update_state(self, token_id: int):
        """
        Update internal state after a token is generated.
        
        Must be called after each token is selected during generation.
        
        Args:
            token_id: The token ID that was just generated
        """
        token_str = self.vocab.id_to_token.get(token_id, "")
        
        if token_str.startswith("NOTE_ON_"):
            try:
                pitch = int(token_str.split("_")[2])
                self.active_notes.append(pitch)
                self.last_pitch = pitch
                self.last_token_type = 'NOTE_ON'
                self.pitch_idx += 1
            except (IndexError, ValueError):
                pass
                
        elif token_str.startswith("TAB_"):
            self.last_token_type = 'TAB'
            
        elif token_str.startswith("NOTE_OFF_"):
            try:
                pitch = int(token_str.split("_")[2])
                if pitch in self.active_notes:
                    self.active_notes.remove(pitch)
                self.last_token_type = 'NOTE_OFF'
            except (IndexError, ValueError):
                pass
                
        elif token_str.startswith("TIME_SHIFT_"):
            self.last_token_type = 'TIME_SHIFT'
            
        elif token_id == self.vocab.eos_id:
            self.last_token_type = 'EOS'
            
        elif token_id == self.vocab.bos_id:
            self.last_token_type = 'START'


class BatchTablatureLogitsProcessor:
    """
    Batch-aware logits processor for constrained tablature generation.
    
    Maintains per-sample state for batched generation.
    """
    
    def __init__(
        self,
        output_vocab: Vocabulary,
        input_pitches_batch: List[List[int]],
        tuning: List[int] = STANDARD_TUNING,
        num_frets: int = DEFAULT_NUM_FRETS,
        device: str = 'cpu'
    ):
        """
        Initialize batch processor.
        
        Args:
            output_vocab: Output vocabulary
            input_pitches_batch: List of pitch sequences, one per batch sample
            tuning: Guitar tuning
            num_frets: Number of frets (should match config, e.g. 25)
            device: Device for tensor operations
        """
        self.batch_size = len(input_pitches_batch)
        self.processors = [
            TablatureLogitsProcessor(output_vocab, pitches, tuning, num_frets, device)
            for pitches in input_pitches_batch
        ]
        self.device = device
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply constrained decoding to batched logits.
        
        Args:
            input_ids: [batch_size, seq_len] - Previously generated tokens
            scores: [batch_size, vocab_size] - Logits for next token
            
        Returns:
            Modified scores with per-sample constraints applied
        """
        batch_size = scores.shape[0]
        
        for b in range(batch_size):
            if b < len(self.processors):
                scores[b] = self.processors[b](input_ids[b], scores[b])
        
        return scores
    
    def update_state(self, token_ids: torch.Tensor):
        """
        Update state for all samples after token selection.
        
        Args:
            token_ids: [batch_size] - Selected token IDs for each sample
        """
        for b, token_id in enumerate(token_ids.tolist()):
            if b < len(self.processors):
                self.processors[b].update_state(token_id)
    
    def reset_all(self):
        """Reset state for all processors."""
        for processor in self.processors:
            processor.reset_state()


def extract_pitches_from_input_ids(
    input_ids: torch.Tensor,
    input_vocab: Vocabulary
) -> List[int]:
    """
    Extract ordered list of pitches from input token IDs.
    
    Args:
        input_ids: [seq_len] - Input sequence token IDs
        input_vocab: Input vocabulary
        
    Returns:
        List of pitches in order of appearance
    """
    pitches = []
    
    for token_id in input_ids.tolist():
        token_str = input_vocab.id_to_token.get(token_id, "")
        if token_str.startswith("NOTE_ON_"):
            try:
                pitch = int(token_str.split("_")[2])
                pitches.append(pitch)
            except (IndexError, ValueError):
                pass
    
    return pitches


def extract_pitches_batch(
    input_ids_batch: torch.Tensor,
    input_vocab: Vocabulary
) -> List[List[int]]:
    """
    Extract pitches from batched input IDs.
    
    Args:
        input_ids_batch: [batch_size, seq_len] - Batched input sequences
        input_vocab: Input vocabulary
        
    Returns:
        List of pitch lists, one per batch sample
    """
    batch_size = input_ids_batch.shape[0]
    return [
        extract_pitches_from_input_ids(input_ids_batch[b], input_vocab)
        for b in range(batch_size)
    ]


def create_constrained_processor(
    input_ids: torch.Tensor,
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    num_frets: int = DEFAULT_NUM_FRETS,
    device: str = 'cpu'
) -> BatchTablatureLogitsProcessor:
    """
    Create a batch logits processor from input IDs.
    
    Convenience function for inference.
    
    Args:
        input_ids: [batch_size, seq_len] - Input token IDs
        input_vocab: Input vocabulary
        output_vocab: Output vocabulary
        num_frets: Number of frets (should match config, e.g. 25)
        device: Device for tensor operations
        
    Returns:
        Configured BatchTablatureLogitsProcessor
    """
    # Handle single sequence (add batch dimension)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    # Extract pitches from input
    input_pitches_batch = extract_pitches_batch(input_ids, input_vocab)
    
    # Create processor
    return BatchTablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches_batch=input_pitches_batch,
        num_frets=num_frets,
        device=device
    )
