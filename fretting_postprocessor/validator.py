"""
Pitch Validator
================

Validates and corrects tablature to ensure correct pitch generation.
This module is crucial for the post-processing pipeline to achieve 100% pitch accuracy.
"""

from typing import List, Tuple, Optional
from .datatypes import Note
from .config import GuitarConfig


class PitchValidator:
    """
    Validates and corrects guitar tablature to ensure accurate pitch generation.

    This class provides static methods for:
    1. Validating that a note's tablature produces the correct pitch
    2. Correcting invalid tablature by finding valid string-fret positions
    3. Finding alternative positions for a given pitch

    These operations are essential for the overlap correction and neighbor search
    algorithms described in the Fretting-Transformer paper (Sections 3.5 and 4.2).

    Examples:
        >>> config = GuitarConfig()
        >>> note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
        ...             velocity=80, string=1, fret=0)
        >>> PitchValidator.validate_note(note, config)
        True

        >>> # Wrong fret - pitch mismatch
        >>> note.fret = 5
        >>> PitchValidator.validate_note(note, config)
        False

        >>> # Correct the tablature
        >>> PitchValidator.correct_note_tablature(note, config)
        True
        >>> note.string, note.fret
        (0, 5)  # or (1, 0) - both are valid
    """

    @staticmethod
    def validate_note(note: Note, config: GuitarConfig) -> bool:
        """
        Validate that a note's tablature produces the correct pitch.

        This checks three conditions:
        1. The note has tablature (both string and fret are assigned)
        2. The string and fret are within valid ranges
        3. The calculated pitch matches the note's actual pitch

        Args:
            note: Note to validate
            config: Guitar configuration (tuning, fret range, etc.)

        Returns:
            True if tablature is valid and produces correct pitch, False otherwise

        Example:
            >>> config = GuitarConfig()
            >>> note = Note(pitch=50, onset_ticks=0, duration_ticks=480,
            ...             velocity=80, string=2, fret=0)
            >>> PitchValidator.validate_note(note, config)
            True  # String 2 (D3) fret 0 = pitch 50

            >>> note.fret = 5
            >>> PitchValidator.validate_note(note, config)
            False  # String 2 fret 5 = pitch 55, not 50
        """
        # Check if note has tablature
        if not note.has_tablature():
            return False

        # Check valid string range
        if not config.is_valid_string(note.string):
            return False

        # Check valid fret range
        if not config.is_valid_fret(note.fret):
            return False

        # Check pitch calculation
        effective_tuning = config.get_effective_tuning()
        calculated_pitch = effective_tuning[note.string] + note.fret

        return calculated_pitch == note.pitch

    @staticmethod
    def correct_note_tablature(note: Note, config: GuitarConfig,
                               preferred_string: Optional[int] = None) -> bool:
        """
        Correct a note's tablature to produce the correct pitch.

        This method finds valid (string, fret) positions for the note's pitch
        and assigns the best one. If a preferred_string is provided, it will
        prioritize that string if valid.

        Args:
            note: Note to correct (will be modified in-place)
            config: Guitar configuration
            preferred_string: Optional preferred string index to prioritize

        Returns:
            True if correction was successful, False if pitch cannot be played

        Modifies:
            note.string and note.fret are updated with valid values

        Example:
            >>> config = GuitarConfig()
            >>> note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
            ...             velocity=80, string=None, fret=None)
            >>> PitchValidator.correct_note_tablature(note, config)
            True
            >>> note.string, note.fret
            (0, 5)  # First valid position: E2 + 5 frets = A2 (45)

            >>> # With preferred string
            >>> note2 = Note(pitch=45, onset_ticks=0, duration_ticks=480,
            ...              velocity=80, string=None, fret=None)
            >>> PitchValidator.correct_note_tablature(note2, config, preferred_string=1)
            True
            >>> note2.string, note2.fret
            (1, 0)  # Preferred string 1: A2 + 0 frets = A2 (45)
        """
        # Find all valid positions for this pitch
        positions = config.pitch_to_string_fret(note.pitch)

        if not positions:
            # Pitch cannot be played on this guitar
            return False

        # If preferred string is provided and valid, use it
        if preferred_string is not None:
            for string, fret in positions:
                if string == preferred_string:
                    note.string = string
                    note.fret = fret
                    return True

        # Fallback: use first valid position (lowest string)
        note.string, note.fret = positions[0]
        return True

    @staticmethod
    def get_alternative_positions(note: Note, config: GuitarConfig,
                                  exclude_current: bool = True) -> List[Tuple[int, int]]:
        """
        Get all alternative (string, fret) positions for a note's pitch.

        This is used by the neighbor search algorithm to explore different
        tablature choices that produce the same pitch, optimizing for
        playability and position continuity.

        Args:
            note: Note to find alternatives for
            config: Guitar configuration
            exclude_current: If True, exclude the note's current position

        Returns:
            List of (string, fret) tuples that produce the same pitch

        Example:
            >>> config = GuitarConfig()
            >>> note = Note(pitch=50, onset_ticks=0, duration_ticks=480,
            ...             velocity=80, string=2, fret=0)
            >>> alts = PitchValidator.get_alternative_positions(note, config)
            >>> alts
            [(0, 10), (1, 5)]  # Excludes current (2, 0)

            >>> # Including current position
            >>> alts_all = PitchValidator.get_alternative_positions(
            ...     note, config, exclude_current=False
            ... )
            >>> alts_all
            [(0, 10), (1, 5), (2, 0)]
        """
        # Get all valid positions for this pitch
        all_positions = config.pitch_to_string_fret(note.pitch)

        if not exclude_current or not note.has_tablature():
            return all_positions

        # Filter out current position
        current_position = (note.string, note.fret)
        return [pos for pos in all_positions if pos != current_position]


# Convenience function for batch validation
def validate_sequence(notes: List[Note], config: GuitarConfig) -> Tuple[int, int]:
    """
    Validate all notes in a sequence.

    Args:
        notes: List of notes to validate
        config: Guitar configuration

    Returns:
        Tuple of (valid_count, total_count)

    Example:
        >>> config = GuitarConfig()
        >>> notes = [
        ...     Note(pitch=45, onset_ticks=0, duration_ticks=480,
        ...          velocity=80, string=1, fret=0),
        ...     Note(pitch=50, onset_ticks=480, duration_ticks=480,
        ...          velocity=80, string=2, fret=0),
        ... ]
        >>> valid, total = validate_sequence(notes, config)
        >>> valid, total
        (2, 2)
    """
    validator = PitchValidator()
    valid_count = sum(1 for note in notes if validator.validate_note(note, config))
    return valid_count, len(notes)


def calculate_pitch_accuracy(notes: List[Note], config: GuitarConfig) -> float:
    """
    Calculate pitch accuracy percentage for a sequence.

    Args:
        notes: List of notes with tablature
        config: Guitar configuration

    Returns:
        Pitch accuracy as percentage (0-100)

    Example:
        >>> config = GuitarConfig()
        >>> notes = [
        ...     Note(pitch=45, onset_ticks=0, duration_ticks=480,
        ...          velocity=80, string=1, fret=0),  # Valid
        ...     Note(pitch=50, onset_ticks=480, duration_ticks=480,
        ...          velocity=80, string=2, fret=5),  # Invalid (should be fret 0)
        ... ]
        >>> calculate_pitch_accuracy(notes, config)
        50.0
    """
    if not notes:
        return 0.0

    valid_count, total_count = validate_sequence(notes, config)
    return (valid_count / total_count) * 100.0
