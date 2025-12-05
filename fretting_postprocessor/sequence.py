"""
Note Sequence Container
=======================

Container class for managing sequences of notes with temporal ordering
and efficient lookup capabilities.
"""

from typing import List, Dict, Optional, Tuple
from .datatypes import Note


class NoteSequence:
    """
    Container for a sequence of notes with temporal ordering and lookup.

    This class provides efficient access patterns for post-processing algorithms,
    including time-based indexing and window queries for the overlap correction
    algorithm.

    Attributes:
        notes: Sorted list of Note objects
        source: Origin identifier ("input", "model", "corrected", etc.)

    Examples:
        >>> notes = [
        ...     Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80),
        ...     Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80),
        ... ]
        >>> sequence = NoteSequence(notes, source="input")
        >>> len(sequence)
        2
        >>> sequence.get_notes_at_time(0)
        [Note(pitch=60, ...)]
    """

    def __init__(self, notes: List[Note], source: str = "input"):
        """
        Initialize note sequence.

        Args:
            notes: List of Note objects (will be sorted)
            source: Source identifier for this sequence
        """
        # Sort by onset time, then by pitch for deterministic ordering
        self.notes = sorted(notes, key=lambda n: (n.onset_ticks, n.pitch))
        self.source = source

        # Build time-based index for efficient queries
        self._time_index: Dict[int, List[Note]] = {}
        self._build_time_index()

    def _build_time_index(self):
        """Build index mapping onset times to notes."""
        self._time_index.clear()
        for note in self.notes:
            if note.onset_ticks not in self._time_index:
                self._time_index[note.onset_ticks] = []
            self._time_index[note.onset_ticks].append(note)

    def get_notes_at_time(self, onset_ticks: int) -> List[Note]:
        """
        Get all notes that start at a specific time.

        Args:
            onset_ticks: Onset time to query

        Returns:
            List of notes starting at this time (may be empty)

        Example:
            >>> sequence.get_notes_at_time(0)
            [Note(pitch=60, onset_ticks=0, ...)]
        """
        return self._time_index.get(onset_ticks, [])

    def get_notes_in_window(self, onset_ticks: int, window_size: int = 5) -> List[Note]:
        """
        Get notes within a temporal window (±window_size notes).

        This is the core query method for the overlap correction algorithm,
        which searches within a ±5 note window for matches.

        Args:
            onset_ticks: Target onset time
            window_size: Number of notes to include before and after target

        Returns:
            List of notes within the window, sorted by onset

        Example:
            >>> # Get notes within ±5 positions of time 480
            >>> window_notes = sequence.get_notes_in_window(480, window_size=5)
            >>> # Returns up to 11 notes (5 before, target, 5 after)
        """
        # Find the index of the first note at or after the target time
        start_idx = 0
        for i, note in enumerate(self.notes):
            if note.onset_ticks >= onset_ticks:
                start_idx = i
                break
        else:
            # All notes are before target time
            start_idx = len(self.notes)

        # Extract window
        window_start = max(0, start_idx - window_size)
        window_end = min(len(self.notes), start_idx + window_size + 1)

        return self.notes[window_start:window_end]

    def get_notes_in_time_range(self, start_ticks: int, end_ticks: int) -> List[Note]:
        """
        Get all notes with onsets in the specified time range (inclusive).

        Args:
            start_ticks: Start of time range
            end_ticks: End of time range

        Returns:
            List of notes within the time range

        Example:
            >>> notes = sequence.get_notes_in_time_range(0, 960)
            >>> # Returns all notes starting between time 0 and 960
        """
        return [n for n in self.notes
                if start_ticks <= n.onset_ticks <= end_ticks]

    def find_closest_note(self,
                          target_pitch: int,
                          onset_ticks: int,
                          max_time_diff: int = 480) -> Optional[Note]:
        """
        Find the closest note to target pitch and time.

        This is used for matching model predictions to input notes during
        post-processing.

        Args:
            target_pitch: Target MIDI pitch
            onset_ticks: Target onset time
            max_time_diff: Maximum time difference to consider (default 480 ticks)

        Returns:
            Closest matching note, or None if no candidates found

        Scoring:
            Primary: Pitch difference (minimize)
            Secondary: Time difference (minimize)

        Example:
            >>> # Find note closest to pitch 60 at time 500
            >>> note = sequence.find_closest_note(60, 500, max_time_diff=240)
            >>> # Returns note with pitch 60 at time 480 (if exists)
        """
        # Get candidates within time window
        candidates = self.get_notes_in_time_range(
            onset_ticks - max_time_diff,
            onset_ticks + max_time_diff
        )

        if not candidates:
            return None

        # Score by pitch similarity (primary) and temporal proximity (secondary)
        def score_note(note: Note) -> Tuple[int, int]:
            pitch_diff = abs(note.pitch - target_pitch)
            time_diff = abs(note.onset_ticks - onset_ticks)
            return (pitch_diff, time_diff)

        return min(candidates, key=score_note)

    def __len__(self) -> int:
        """Return number of notes in sequence."""
        return len(self.notes)

    def __iter__(self):
        """Iterate over notes in temporal order."""
        return iter(self.notes)

    def __getitem__(self, index: int) -> Note:
        """Access note by index."""
        return self.notes[index]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"NoteSequence({len(self.notes)} notes, source='{self.source}')"
