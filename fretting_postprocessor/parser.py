"""
Token Parser
============

Parser for converting v3 format token sequences into internal Note representations.
"""

import re
import warnings
from typing import List, Optional, Dict
from .datatypes import Token, TokenType, Note
from .sequence import NoteSequence
from .config import GuitarConfig


class TokenParser:
    """
    Parser for v3 format tokens (from paper Table 1).

    Input format:  NOTE_ON<pitch> TIME_SHIFT<ticks> NOTE_OFF<pitch>
    Output format: TAB<string,fret> TIME_SHIFT<ticks>

    Examples:
        >>> parser = TokenParser()
        >>>
        >>> # Parse input notes (MIDI format)
        >>> input_tokens = [
        ...     "NOTE_ON<60>", "TIME_SHIFT<480>", "NOTE_OFF<60>",
        ...     "NOTE_ON<62>", "TIME_SHIFT<480>", "NOTE_OFF<62>",
        ... ]
        >>> input_sequence = parser.parse_input_tokens(input_tokens)
        >>> len(input_sequence)
        2
        >>>
        >>> # Parse output tablature
        >>> output_tokens = [
        ...     "TAB<3,5>", "TIME_SHIFT<480>",
        ...     "TAB<3,7>", "TIME_SHIFT<480>",
        ... ]
        >>> config = GuitarConfig()
        >>> output_sequence = parser.parse_output_tokens(
        ...     output_tokens, input_sequence, config
        ... )
    """

    # Token pattern regexes
    NOTE_ON_PATTERN = re.compile(r'NOTE_ON<(\d+)>')
    NOTE_OFF_PATTERN = re.compile(r'NOTE_OFF<(\d+)>')
    TIME_SHIFT_PATTERN = re.compile(r'TIME_SHIFT<(\d+)>')
    TAB_PATTERN = re.compile(r'TAB<(\d+),(\d+)>')

    def parse_input_tokens(self,
                          tokens: List[str],
                          default_velocity: int = 64) -> NoteSequence:
        """
        Parse input note sequence tokens (NOTE_ON/OFF format) into Note objects.

        The parser tracks active notes and completes them when NOTE_OFF is
        encountered, calculating duration from time shifts.

        Args:
            tokens: List of token strings in v3 input format
            default_velocity: Default velocity for notes (default 64)

        Returns:
            NoteSequence containing parsed notes

        Example:
            >>> tokens = [
            ...     "NOTE_ON<60>", "TIME_SHIFT<240>",
            ...     "NOTE_OFF<60>", "NOTE_ON<62>",
            ...     "TIME_SHIFT<240>", "NOTE_OFF<62>"
            ... ]
            >>> sequence = parser.parse_input_tokens(tokens)
            >>> sequence[0].pitch == 60
            True
            >>> sequence[0].duration_ticks == 240
            True

        Note:
            - Handles missing NOTE_OFF by terminating at sequence end
            - Ignores malformed tokens with warning
            - Supports chords (multiple NOTE_ON at same time)
        """
        notes = []
        current_time = 0
        active_notes: Dict[int, int] = {}  # pitch -> onset_time

        for token_str in tokens:
            # Parse NOTE_ON
            match = self.NOTE_ON_PATTERN.match(token_str)
            if match:
                pitch = int(match.group(1))
                if pitch in active_notes:
                    # Missing NOTE_OFF for previous note with same pitch
                    # Terminate it now
                    onset = active_notes.pop(pitch)
                    duration = current_time - onset
                    notes.append(Note(
                        pitch=pitch,
                        onset_ticks=onset,
                        duration_ticks=duration,
                        velocity=default_velocity,
                        source="input"
                    ))

                active_notes[pitch] = current_time
                continue

            # Parse TIME_SHIFT
            match = self.TIME_SHIFT_PATTERN.match(token_str)
            if match:
                ticks = int(match.group(1))
                current_time += ticks
                continue

            # Parse NOTE_OFF
            match = self.NOTE_OFF_PATTERN.match(token_str)
            if match:
                pitch = int(match.group(1))
                if pitch in active_notes:
                    onset = active_notes.pop(pitch)
                    duration = current_time - onset
                    notes.append(Note(
                        pitch=pitch,
                        onset_ticks=onset,
                        duration_ticks=duration,
                        velocity=default_velocity,
                        source="input"
                    ))
                continue

            # Malformed token
            warnings.warn(f"Malformed input token: {token_str}")

        # Handle any remaining active notes (missing NOTE_OFF at end)
        for pitch, onset in active_notes.items():
            duration = current_time - onset
            notes.append(Note(
                pitch=pitch,
                onset_ticks=onset,
                duration_ticks=duration,
                velocity=default_velocity,
                source="input"
            ))

        return NoteSequence(notes, source="input")

    def parse_output_tokens(self,
                           tokens: List[str],
                           input_sequence: NoteSequence,
                           guitar_config: GuitarConfig,
                           default_velocity: int = 64) -> NoteSequence:
        """
        Parse model output tokens (TAB format) into Note objects.

        This requires the input sequence to infer pitches and durations,
        as TAB tokens only specify string-fret positions.

        Args:
            tokens: List of token strings in v3 output format (TAB)
            input_sequence: Input note sequence for reference
            guitar_config: Guitar configuration for pitch calculation
            default_velocity: Default velocity if no match found

        Returns:
            NoteSequence containing parsed notes with tablature

        Example:
            >>> output_tokens = ["TAB<3,5>", "TIME_SHIFT<480>"]
            >>> config = GuitarConfig()
            >>> sequence = parser.parse_output_tokens(
            ...     output_tokens, input_sequence, config
            ... )
            >>> sequence[0].string == 3
            True
            >>> sequence[0].fret == 5
            True
            >>> sequence[0].pitch == 60  # Calculated from tablature
            True

        Process:
            1. Parse TAB tokens and buffer by time
            2. Calculate pitch from tablature (pitch = tuning[string] + fret)
            3. Try to match with input notes at same time for duration/velocity
            4. Use defaults if no match found
        """
        notes = []
        current_time = 0
        tab_buffer: List[tuple] = []  # Buffer for TAB tokens at current time

        effective_tuning = guitar_config.get_effective_tuning()

        for token_str in tokens:
            # Parse TAB
            match = self.TAB_PATTERN.match(token_str)
            if match:
                string = int(match.group(1))
                fret = int(match.group(2))
                tab_buffer.append((string, fret, current_time))
                continue

            # Parse TIME_SHIFT
            match = self.TIME_SHIFT_PATTERN.match(token_str)
            if match:
                ticks = int(match.group(1))

                # Process buffered TAB tokens before advancing time
                if tab_buffer:
                    input_notes = input_sequence.get_notes_at_time(current_time)

                    for string, fret, onset in tab_buffer:
                        # Calculate pitch from tablature
                        if string >= len(effective_tuning):
                            warnings.warn(
                                f"Invalid string {string} for {guitar_config.num_strings}-string guitar"
                            )
                            continue

                        pitch = effective_tuning[string] + fret

                        # Try to match with input note to get duration/velocity
                        matched_note = None
                        for inp_note in input_notes:
                            if inp_note.pitch == pitch and not inp_note.matched:
                                matched_note = inp_note
                                matched_note.matched = True
                                break

                        if matched_note:
                            duration = matched_note.duration_ticks
                            velocity = matched_note.velocity
                        else:
                            # Fallback: estimate duration from next time shift
                            duration = ticks
                            velocity = default_velocity

                        notes.append(Note(
                            pitch=pitch,
                            onset_ticks=onset,
                            duration_ticks=duration,
                            velocity=velocity,
                            string=string,
                            fret=fret,
                            source="model"
                        ))

                    tab_buffer.clear()

                current_time += ticks
                continue

            # Malformed token
            warnings.warn(f"Malformed output token: {token_str}")

        # Process any remaining buffered TAB tokens
        if tab_buffer:
            for string, fret, onset in tab_buffer:
                if string >= len(effective_tuning):
                    continue

                pitch = effective_tuning[string] + fret
                notes.append(Note(
                    pitch=pitch,
                    onset_ticks=onset,
                    duration_ticks=480,  # Default duration
                    velocity=default_velocity,
                    string=string,
                    fret=fret,
                    source="model"
                ))

        return NoteSequence(notes, source="model")
