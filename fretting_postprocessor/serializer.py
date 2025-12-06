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

    @staticmethod
    def serialize_to_note_on_off(
        sequence: NoteSequence,
        use_note_off: bool = True
    ) -> List[str]:
        """
        將 NoteSequence 序列化為 NOTE_ON/NOTE_OFF format tokens。

        這是 serialize_to_input_format() 的增強版本，能從包含 tablature
        的 notes 中提取優化後的 pitch。

        Args:
            sequence: 要序列化的 note sequence
            use_note_off: 是否使用 NOTE_OFF (True) 或僅用 TIME_SHIFT (False)

        Returns:
            List of tokens in NOTE_ON/NOTE_OFF format

        Example:
            >>> # Note with optimized tablature from neighbor_search
            >>> note = Note(pitch=60, onset_ticks=0, duration_ticks=480,
            ...             velocity=80, string=3, fret=5)
            >>> note._guitar_config = GuitarConfig()
            >>> sequence = NoteSequence([note])
            >>> tokens = serializer.serialize_to_note_on_off(sequence)
            >>> tokens
            ['NOTE_ON<60>', 'TIME_SHIFT<480>', 'NOTE_OFF<60>']

        Process:
            1. 按時間分組 notes
            2. 計算每個 note 的實際 pitch (從 string/fret 或直接使用 pitch)
            3. 生成 NOTE_ON/NOTE_OFF/TIME_SHIFT tokens

        Note:
            如果 note 有 tablature 且附加了 _guitar_config，會從優化後的
            tablature 重新計算 pitch，確保 neighbor_search 的優化結果
            反映在輸出中。
        """
        if not sequence:
            return []

        tokens = []

        # 按 onset 排序
        sorted_notes = sorted(sequence, key=lambda n: (n.onset_ticks, n.pitch))

        if use_note_off:
            # 使用 NOTE_ON/NOTE_OFF 格式
            events = []

            for note in sorted_notes:
                # 計算最終 pitch (可能從優化後的 tablature 重新計算)
                final_pitch = TokenSerializer._get_final_pitch(note)

                events.append((note.onset_ticks, 'ON', final_pitch))
                events.append(
                    (note.onset_ticks + note.duration_ticks, 'OFF', final_pitch)
                )

            # 按時間排序所有事件 (OFF events 優先)
            events.sort(key=lambda e: (e[0], e[1] == 'OFF', -e[2]))

            current_time = 0
            for time, event_type, pitch in events:
                # 插入 TIME_SHIFT
                if time > current_time:
                    tokens.append(f"TIME_SHIFT<{time - current_time}>")
                    current_time = time

                # 插入 NOTE event
                if event_type == 'ON':
                    tokens.append(f"NOTE_ON<{pitch}>")
                else:
                    tokens.append(f"NOTE_OFF<{pitch}>")

        else:
            # 簡化格式: 僅用 NOTE_ON + TIME_SHIFT (duration)
            current_time = 0

            for note in sorted_notes:
                final_pitch = TokenSerializer._get_final_pitch(note)

                # TIME_SHIFT to onset
                if note.onset_ticks > current_time:
                    tokens.append(f"TIME_SHIFT<{note.onset_ticks - current_time}>")
                    current_time = note.onset_ticks

                # NOTE_ON
                tokens.append(f"NOTE_ON<{final_pitch}>")

                # TIME_SHIFT for duration
                tokens.append(f"TIME_SHIFT<{note.duration_ticks}>")
                current_time += note.duration_ticks

        return tokens

    @staticmethod
    def serialize_to_mixed_format(
        sequence: NoteSequence,
        use_note_off: bool = True
    ) -> List[str]:
        """
        Serialize NoteSequence to MIXED format (NOTE_ON/OFF + TAB).

        This format matches the training data structure where each note event
        has BOTH pitch tokens (NOTE_ON/OFF) AND tablature token (TAB).

        Example output: NOTE_ON<60> TAB<3,5> NOTE_OFF<60> TIME_SHIFT<480>

        Args:
            sequence: NoteSequence to serialize
            use_note_off: Whether to use NOTE_OFF tokens

        Returns:
            List of tokens in mixed format
        """
        if not sequence:
            return []

        tokens = []

        # Sort notes by onset
        sorted_notes = sorted(sequence, key=lambda n: (n.onset_ticks, n.pitch))

        if use_note_off:
            # Use NOTE_ON/NOTE_OFF format
            events = []

            for note in sorted_notes:
                # Get final pitch (possibly recalculated from optimized tablature)
                final_pitch = TokenSerializer._get_final_pitch(note)

                # Add NOTE_ON + TAB at onset
                events.append((note.onset_ticks, 'ON', final_pitch, note))
                # Add NOTE_OFF at offset
                events.append((note.onset_ticks + note.duration_ticks, 'OFF', final_pitch, None))

            # Sort all events by time
            events.sort(key=lambda e: (e[0], e[1] == 'OFF'))  # OFF events after ON

            current_time = 0
            for time, event_type, pitch, note in events:
                # Insert TIME_SHIFT if needed
                if time > current_time:
                    tokens.append(f"TIME_SHIFT<{time - current_time}>")
                    current_time = time

                # Insert NOTE event
                if event_type == 'ON':
                    tokens.append(f"NOTE_ON<{pitch}>")
                    # Add TAB token immediately after NOTE_ON
                    if note and note.has_tablature():
                        tokens.append(f"TAB<{note.string},{note.fret}>")
                else:  # OFF
                    tokens.append(f"NOTE_OFF<{pitch}>")
        else:
            # Simplified format: only NOTE_ON + TAB + TIME_SHIFT (duration)
            current_time = 0

            for note in sorted_notes:
                final_pitch = TokenSerializer._get_final_pitch(note)

                # TIME_SHIFT to onset
                if note.onset_ticks > current_time:
                    tokens.append(f"TIME_SHIFT<{note.onset_ticks - current_time}>")
                    current_time = note.onset_ticks

                # NOTE_ON
                tokens.append(f"NOTE_ON<{final_pitch}>")

                # TAB (if available)
                if note.has_tablature():
                    tokens.append(f"TAB<{note.string},{note.fret}>")

                # TIME_SHIFT for duration
                tokens.append(f"TIME_SHIFT<{note.duration_ticks}>")
                current_time += note.duration_ticks

        return tokens

    @staticmethod
    def _get_final_pitch(note: Note) -> int:
        """
        獲取 note 的最終 pitch。

        如果 note 有 tablature (經過 neighbor_search 優化) 且附加了
        _guitar_config，則重新計算 pitch。否則使用原始 pitch。

        這確保了 neighbor_search 的優化結果能反映在輸出中。

        Args:
            note: Note object

        Returns:
            Final pitch value (可能從優化後的 tablature 重新計算)

        Example:
            >>> # Note with tablature and config
            >>> note = Note(pitch=60, string=3, fret=5)
            >>> note._guitar_config = GuitarConfig()
            >>> final_pitch = TokenSerializer._get_final_pitch(note)
            >>> final_pitch
            60  # Recalculated from string 3 fret 5

            >>> # Note without tablature
            >>> note = Note(pitch=60)
            >>> final_pitch = TokenSerializer._get_final_pitch(note)
            >>> final_pitch
            60  # Original pitch
        """
        if note.has_tablature() and hasattr(note, '_guitar_config'):
            # 從優化後的 tablature 重新計算 pitch
            config = note._guitar_config
            effective_tuning = config.get_effective_tuning()

            if note.string < len(effective_tuning):
                return effective_tuning[note.string] + note.fret
            else:
                # Fallback to original pitch if string invalid
                return note.pitch
        else:
            # 使用原始 pitch
            return note.pitch
