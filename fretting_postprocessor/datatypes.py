"""
Data Types for Fretting Post-Processor
=======================================

Core data structures for representing tokens and notes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class TokenType(Enum):
    """Token types in v3 format (from paper Table 1)"""
    NOTE_ON = "NOTE_ON"
    NOTE_OFF = "NOTE_OFF"
    TIME_SHIFT = "TIME_SHIFT"
    TAB = "TAB"


@dataclass
class Token:
    """
    Base token representation for v3 format.

    Attributes:
        token_type: Type of token (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB)
        value: Integer value (pitch for NOTE_ON/OFF, ticks for TIME_SHIFT)
        string_fret: Optional tuple (string, fret) for TAB tokens
        position: Position in sequence (default 0)
    """
    token_type: TokenType
    value: int
    string_fret: Optional[Tuple[int, int]] = None
    position: int = 0

    def __repr__(self) -> str:
        """String representation of token"""
        if self.token_type == TokenType.TAB and self.string_fret:
            return f"TAB<{self.string_fret[0]},{self.string_fret[1]}>"
        else:
            return f"{self.token_type.value}<{self.value}>"


@dataclass
class Note:
    """
    Musical note representation integrating pitch and tablature information.

    This class bridges between input MIDI notes (pitch-based) and output
    tablature (string-fret based), tracking both representations and their
    correspondence.

    Attributes:
        pitch: MIDI pitch number (0-127)
        onset_ticks: Absolute time position in ticks
        duration_ticks: Note duration in ticks
        velocity: MIDI velocity (0-127)
        string: String index (0-5 for 6-string guitar), None if not assigned
        fret: Fret number (0-24 typically), None if not assigned
        matched: Whether this note has been matched in post-processing
        source: Origin of note ("input", "model", "corrected", "refined")

    Example:
        >>> # Create note from input (pitch-based)
        >>> note = Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80)
        >>>
        >>> # Add tablature information
        >>> note.string = 1
        >>> note.fret = 5
        >>>
        >>> # Calculate pitch from tablature
        >>> tuning = (40, 45, 50, 55, 59, 64)  # Standard tuning
        >>> calculated_pitch = note.get_pitch_from_tablature(tuning)
        >>> assert calculated_pitch == 50  # A2 + 5 frets = D3
    """
    pitch: int
    onset_ticks: int
    duration_ticks: int
    velocity: int
    string: Optional[int] = None
    fret: Optional[int] = None
    matched: bool = False
    source: str = "input"

    def has_tablature(self) -> bool:
        """
        Check if note has valid tablature information.

        Returns:
            True if both string and fret are assigned, False otherwise
        """
        return self.string is not None and self.fret is not None

    def get_pitch_from_tablature(self, tuning: Tuple[int, ...]) -> Optional[int]:
        """
        Calculate MIDI pitch from tablature using guitar tuning.

        Formula: pitch = tuning[string] + fret

        Args:
            tuning: Tuple of MIDI pitches for open strings

        Returns:
            Calculated pitch, or None if tablature is incomplete

        Example:
            >>> note = Note(pitch=0, onset_ticks=0, duration_ticks=480,
            ...             velocity=80, string=1, fret=5)
            >>> tuning = (40, 45, 50, 55, 59, 64)
            >>> note.get_pitch_from_tablature(tuning)
            50  # A2 (45) + 5 frets = D3 (50)
        """
        if not self.has_tablature():
            return None
        return tuning[self.string] + self.fret

    def get_offset_ticks(self) -> int:
        """
        Get the note offset time (onset + duration).

        Returns:
            Absolute time when note ends
        """
        return self.onset_ticks + self.duration_ticks

    def __repr__(self) -> str:
        """String representation for debugging"""
        tab_info = f" string={self.string} fret={self.fret}" if self.has_tablature() else ""
        return (f"Note(pitch={self.pitch}, onset={self.onset_ticks}, "
                f"dur={self.duration_ticks}{tab_info}, source={self.source})")
