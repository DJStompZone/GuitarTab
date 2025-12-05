"""
Token Serializer
================

Serializes internal Note representations back to v3 format token strings.
"""

from typing import List, Dict
from .datatypes import Note
from .sequence import NoteSequence


class TokenSerializer:
    """
    Serializer for converting Note sequences to v3 format token strings.

    Examples:
        >>> serializer = TokenSerializer()
        >>>
        >>> # Serialize to input format (NOTE_ON/OFF)
        >>> notes = [Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80)]
        >>> sequence = NoteSequence(notes)
        >>> tokens = serializer.serialize_to_input_format(sequence)
        >>> tokens
        ['NOTE_ON<60>', 'TIME_SHIFT<480>', 'NOTE_OFF<60>']
        >>>
        >>> # Serialize to output format (TAB)
        >>> notes = [Note(pitch=60, onset_ticks=0, duration_ticks=480,
        ...              velocity=80, string=3, fret=5)]
        >>> sequence = NoteSequence(notes)
        >>> tokens = serializer.serialize_to_output_format(sequence)
        >>> tokens
        ['TAB<3,5>', 'TIME_SHIFT<480>']
    """

    @staticmethod
    def serialize_to_input_format(sequence: NoteSequence) -> List[str]:
        """
        Serialize notes to input token format (NOTE_ON/OFF).

        Format: NOTE_ON<pitch> TIME_SHIFT<ticks> NOTE_OFF<pitch>

        Args:
            sequence: NoteSequence to serialize

        Returns:
            List of token strings in input format

        Example:
            >>> notes = [
            ...     Note(pitch=60, onset_ticks=0, duration_ticks=480, velocity=80),
            ...     Note(pitch=62, onset_ticks=480, duration_ticks=480, velocity=80),
            ... ]
            >>> sequence = NoteSequence(notes)
            >>> tokens = serializer.serialize_to_input_format(sequence)
            >>> tokens
            ['NOTE_ON<60>', 'TIME_SHIFT<480>', 'NOTE_OFF<60>',
             'NOTE_ON<62>', 'TIME_SHIFT<480>', 'NOTE_OFF<62>']

        Process:
            1. Create note-on and note-off events with timestamps
            2. Sort events by time (note-offs before note-ons at same time)
            3. Generate TIME_SHIFT tokens for time advances
            4. Generate NOTE_ON/NOTE_OFF tokens
        """
        tokens = []

        # Collect events (time, type, note)
        events = []
        for note in sequence:
            events.append((note.onset_ticks, 'on', note))
            events.append((note.get_offset_ticks(), 'off', note))

        # Sort events by time, with offs before ons at same time
        events.sort(key=lambda e: (e[0], e[1] == 'on'))

        # Generate tokens
        current_time = 0
        for time, event_type, note in events:
            # Add time shift if needed
            if time > current_time:
                tokens.append(f"TIME_SHIFT<{time - current_time}>")
                current_time = time

            # Add note event
            if event_type == 'on':
                tokens.append(f"NOTE_ON<{note.pitch}>")
            else:
                tokens.append(f"NOTE_OFF<{note.pitch}>")

        return tokens

    @staticmethod
    def serialize_to_output_format(sequence: NoteSequence) -> List[str]:
        """
        Serialize notes to output token format (TAB).

        Format: TAB<string,fret> TIME_SHIFT<ticks>

        Args:
            sequence: NoteSequence to serialize (must have tablature info)

        Returns:
            List of token strings in output format

        Example:
            >>> notes = [
            ...     Note(pitch=60, onset_ticks=0, duration_ticks=480,
            ...          velocity=80, string=3, fret=5),
            ...     Note(pitch=62, onset_ticks=480, duration_ticks=480,
            ...          velocity=80, string=3, fret=7),
            ... ]
            >>> sequence = NoteSequence(notes)
            >>> tokens = serializer.serialize_to_output_format(sequence)
            >>> tokens
            ['TAB<3,5>', 'TIME_SHIFT<480>',
             'TAB<3,7>', 'TIME_SHIFT<480>']

        Note:
            - Only notes with tablature information are included
            - Notes at same time are sorted by string number
            - TIME_SHIFT represents duration to next note group
        """
        tokens = []

        # Group notes by onset time
        notes_by_time: Dict[int, List[Note]] = {}
        for note in sequence:
            if not note.has_tablature():
                # Skip notes without tablature
                continue

            if note.onset_ticks not in notes_by_time:
                notes_by_time[note.onset_ticks] = []
            notes_by_time[note.onset_ticks].append(note)

        # Sort times
        sorted_times = sorted(notes_by_time.keys())

        # Generate tokens
        for i, onset_time in enumerate(sorted_times):
            # Add TAB tokens for all notes at this time
            # Sort by string number for deterministic output
            notes = sorted(notes_by_time[onset_time], key=lambda n: n.string)
            for note in notes:
                tokens.append(f"TAB<{note.string},{note.fret}>")

            # Add TIME_SHIFT after TAB tokens
            # Calculate time to next note group (or use duration if last note)
            if i < len(sorted_times) - 1:
                # Time until next note group
                time_shift = sorted_times[i + 1] - onset_time
            else:
                # For last note, use its duration
                time_shift = notes[0].duration_ticks

            tokens.append(f"TIME_SHIFT<{time_shift}>")

        return tokens

    @staticmethod
    def tokens_to_string(tokens: List[str], separator: str = " ") -> str:
        """
        Convert token list to single string.

        Args:
            tokens: List of token strings
            separator: Separator between tokens (default space)

        Returns:
            Single string of tokens

        Example:
            >>> tokens = ["NOTE_ON<60>", "TIME_SHIFT<480>", "NOTE_OFF<60>"]
            >>> serializer.tokens_to_string(tokens)
            'NOTE_ON<60> TIME_SHIFT<480> NOTE_OFF<60>'
        """
        return separator.join(tokens)

    @staticmethod
    def string_to_tokens(token_string: str, separator: str = " ") -> List[str]:
        """
        Parse string into token list.

        Args:
            token_string: String containing tokens
            separator: Separator between tokens (default space)

        Returns:
            List of token strings

        Example:
            >>> s = "NOTE_ON<60> TIME_SHIFT<480> NOTE_OFF<60>"
            >>> serializer.string_to_tokens(s)
            ['NOTE_ON<60>', 'TIME_SHIFT<480>', 'NOTE_OFF<60>']
        """
        return [t.strip() for t in token_string.split(separator) if t.strip()]
