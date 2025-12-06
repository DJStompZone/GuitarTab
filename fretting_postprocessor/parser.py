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

    def parse_note_on_off_output(
        self,
        tokens: List[str],
        input_sequence: NoteSequence,
        guitar_config: GuitarConfig,
        default_velocity: int = 64
    ) -> NoteSequence:
        """
        解析 NOTE_ON/OFF 格式的模型輸出並推導 tablature。

        與 parse_input_tokens() 類似，但額外推導 string/fret 位置。
        這允許後續的 neighbor_search 算法優化指法。

        Args:
            tokens: NOTE_ON/OFF format tokens (模型輸出)
            input_sequence: 輸入序列（用於推導 duration/velocity）
            guitar_config: 吉他配置
            default_velocity: 預設力度

        Returns:
            NoteSequence with inferred tablature

        Example:
            >>> tokens = ["NOTE_ON<60>", "TIME_SHIFT<240>", "NOTE_OFF<60>"]
            >>> config = GuitarConfig()
            >>> input_seq = NoteSequence([])  # Empty input
            >>> output_seq = parser.parse_note_on_off_output(tokens, input_seq, config)
            >>> output_seq[0].pitch
            60
            >>> output_seq[0].has_tablature()
            True

        Process:
            1. 解析 NOTE_ON/OFF tokens 得到 pitch 和 timing
            2. 為每個 note 推導最佳 (string, fret) 位置
            3. 嘗試匹配 input_sequence 的 duration/velocity
        """
        notes = []
        current_time = 0
        active_notes: Dict[int, int] = {}  # pitch -> onset_time

        # 解析 tokens (與 parse_input_tokens 相同邏輯)
        for token_str in tokens:
            # NOTE_ON
            match = self.NOTE_ON_PATTERN.match(token_str)
            if match:
                pitch = int(match.group(1))
                if pitch in active_notes:
                    # 處理缺失的 NOTE_OFF
                    onset = active_notes.pop(pitch)
                    duration = current_time - onset

                    # 推導 tablature
                    tablature = guitar_config.get_default_tablature_for_pitch(pitch)

                    # 嘗試從 input 匹配 velocity
                    velocity = self._find_velocity_from_input(
                        pitch, onset, input_sequence, default_velocity
                    )

                    if tablature:
                        string, fret = tablature
                        notes.append(Note(
                            pitch=pitch,
                            onset_ticks=onset,
                            duration_ticks=duration,
                            velocity=velocity,
                            string=string,
                            fret=fret,
                            source="model_inferred"
                        ))
                    else:
                        # 無法表示的音高（超出範圍）
                        warnings.warn(f"Cannot represent pitch {pitch} on guitar")

                active_notes[pitch] = current_time
                continue

            # TIME_SHIFT
            match = self.TIME_SHIFT_PATTERN.match(token_str)
            if match:
                ticks = int(match.group(1))
                current_time += ticks
                continue

            # NOTE_OFF
            match = self.NOTE_OFF_PATTERN.match(token_str)
            if match:
                pitch = int(match.group(1))
                if pitch in active_notes:
                    onset = active_notes.pop(pitch)
                    duration = current_time - onset

                    # 推導 tablature
                    tablature = guitar_config.get_default_tablature_for_pitch(pitch)

                    velocity = self._find_velocity_from_input(
                        pitch, onset, input_sequence, default_velocity
                    )

                    if tablature:
                        string, fret = tablature
                        notes.append(Note(
                            pitch=pitch,
                            onset_ticks=onset,
                            duration_ticks=duration,
                            velocity=velocity,
                            string=string,
                            fret=fret,
                            source="model_inferred"
                        ))
                continue

            # 其他 tokens (BOS, EOS, etc.)
            if token_str not in ['PAD', 'BOS', 'EOS', 'UNK']:
                warnings.warn(f"Unexpected token in NOTE_ON/OFF output: {token_str}")

        # 處理未關閉的 notes
        for pitch, onset in active_notes.items():
            duration = current_time - onset
            tablature = guitar_config.get_default_tablature_for_pitch(pitch)
            velocity = default_velocity

            if tablature:
                string, fret = tablature
                notes.append(Note(
                    pitch=pitch,
                    onset_ticks=onset,
                    duration_ticks=duration,
                    velocity=velocity,
                    string=string,
                    fret=fret,
                    source="model_inferred"
                ))

        return NoteSequence(notes, source="model_inferred")

    def parse_mixed_format_output(
        self,
        tokens: List[str],
        input_sequence: NoteSequence,
        guitar_config: GuitarConfig,
        default_velocity: int = 64
    ) -> NoteSequence:
        """
        Parse MIXED format output (NOTE_ON/OFF + TAB tokens).

        This format contains both pitch information (NOTE_ON/OFF) and tablature
        (TAB) for each note. We extract tablature from TAB tokens and use
        NOTE_ON/OFF for duration/timing information.

        Args:
            tokens: MIXED format tokens (model output)
            input_sequence: Input sequence (for velocity matching)
            guitar_config: Guitar configuration
            default_velocity: Default velocity

        Returns:
            NoteSequence with tablature from TAB tokens
        """
        notes = []
        current_time = 0
        tab_buffer = []  # List of (string, fret, onset, pitch_from_note_on)
        active_notes = {}  # pitch -> (onset, string, fret)

        effective_tuning = guitar_config.get_effective_tuning()

        for token_str in tokens:
            # NOTE_ON - track onset and pitch
            match = self.NOTE_ON_PATTERN.match(token_str)
            if match:
                pitch = int(match.group(1))
                # Store onset for this pitch (will be matched with TAB or NOTE_OFF)
                active_notes[pitch] = {'onset': current_time, 'string': None, 'fret': None}
                continue

            # TAB - extract tablature and match with active NOTE_ON
            match = self.TAB_PATTERN.match(token_str)
            if match:
                string = int(match.group(1))
                fret = int(match.group(2))

                # Calculate pitch from TAB
                if string < len(effective_tuning):
                    tab_pitch = effective_tuning[string] + fret

                    # Try to match with active NOTE_ON of same pitch
                    if tab_pitch in active_notes:
                        active_notes[tab_pitch]['string'] = string
                        active_notes[tab_pitch]['fret'] = fret
                    else:
                        # TAB without matching NOTE_ON - buffer it
                        tab_buffer.append((string, fret, current_time, tab_pitch))
                continue

            # NOTE_OFF - finalize note with duration
            match = self.NOTE_OFF_PATTERN.match(token_str)
            if match:
                pitch = int(match.group(1))

                if pitch in active_notes:
                    note_info = active_notes.pop(pitch)
                    onset = note_info['onset']
                    duration = current_time - onset
                    string = note_info['string']
                    fret = note_info['fret']

                    # If we have tablature from TAB token, use it
                    if string is not None and fret is not None:
                        # Find velocity from input
                        velocity = self._find_velocity_from_input(
                            pitch, onset, input_sequence, default_velocity
                        )

                        notes.append(Note(
                            pitch=pitch,
                            onset_ticks=onset,
                            duration_ticks=duration,
                            velocity=velocity,
                            string=string,
                            fret=fret,
                            source="model_mixed"
                        ))
                    # else: NOTE_OFF without TAB - skip (malformed)
                continue

            # TIME_SHIFT
            match = self.TIME_SHIFT_PATTERN.match(token_str)
            if match:
                ticks = int(match.group(1))

                # Flush buffered TAB tokens before advancing time
                if tab_buffer:
                    for string, fret, onset, pitch in tab_buffer:
                        # Estimate duration from time shift
                        duration = ticks
                        velocity = default_velocity

                        notes.append(Note(
                            pitch=pitch,
                            onset_ticks=onset,
                            duration_ticks=duration,
                            velocity=velocity,
                            string=string,
                            fret=fret,
                            source="model_mixed"
                        ))

                    tab_buffer.clear()

                current_time += ticks
                continue

            # Special tokens (PAD, BOS, EOS, UNK) - ignore
            if token_str in ['PAD', 'BOS', 'EOS', 'UNK']:
                continue

            # Unknown token - warn but continue
            warnings.warn(f"Unexpected token in MIXED output: {token_str}")

        # Process remaining buffered TAB or active notes
        if tab_buffer:
            for string, fret, onset, pitch in tab_buffer:
                notes.append(Note(
                    pitch=pitch,
                    onset_ticks=onset,
                    duration_ticks=480,  # Default duration
                    velocity=default_velocity,
                    string=string,
                    fret=fret,
                    source="model_mixed"
                ))

        for pitch, note_info in active_notes.items():
            onset = note_info['onset']
            string = note_info['string']
            fret = note_info['fret']

            if string is not None and fret is not None:
                duration = current_time - onset
                velocity = default_velocity

                notes.append(Note(
                    pitch=pitch,
                    onset_ticks=onset,
                    duration_ticks=duration,
                    velocity=velocity,
                    string=string,
                    fret=fret,
                    source="model_mixed"
                ))

        return NoteSequence(notes, source="model_mixed")

    def _find_velocity_from_input(
        self,
        pitch: int,
        onset: int,
        input_sequence: NoteSequence,
        default: int
    ) -> int:
        """
        從 input sequence 中尋找匹配的 note 並提取 velocity。

        Args:
            pitch: 目標音高
            onset: 起始時間
            input_sequence: 輸入序列
            default: 預設值

        Returns:
            Velocity value
        """
        time_window = 120  # ticks tolerance

        for note in input_sequence:
            if (note.pitch == pitch and
                abs(note.onset_ticks - onset) <= time_window):
                return note.velocity

        return default
