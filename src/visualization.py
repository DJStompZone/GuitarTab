"""Helper functions for visualizing guitar tablature tokens."""

from collections import defaultdict
from typing import List, Union


def render_dadagp_tokens_as_tablature(tokens, ticks_per_beat=480, max_bars=4, chars_per_beat=8, bars_per_row=1):
    """
    Render DadaGP raw tokens as six-string guitar tablature.

    Args:
        tokens: List of DadaGP token objects (NoteToken, WaitToken, etc.)
        ticks_per_beat: MIDI ticks per quarter note (default 480)
        max_bars: Maximum number of bars to render
        chars_per_beat: Number of characters per beat
        bars_per_row: Number of bars to display horizontally per row

    Returns:
        String containing tablature notation
    """
    from src.dadagp_parser import NoteToken, WaitToken

    # Initialize 6 strings
    strings = {i: [] for i in range(1, 7)}

    current_tick = 0
    ticks_per_bar = ticks_per_beat * 4  # Assume 4/4 time

    # Parse tokens into timeline
    for token in tokens:
        if isinstance(token, WaitToken):
            current_tick += token.ticks
        elif isinstance(token, NoteToken):
            strings[token.string].append((current_tick, token.fret))

    # Calculate total bars
    max_tick = max(current_tick, 1)
    total_bars = (max_tick + ticks_per_bar - 1) // ticks_per_bar
    bars_to_render = min(total_bars, max_bars)

    # Build tablature string
    output = []
    output.append("Guitar Tablature (Standard Tuning: EADGBE) - from DadaGP tokens")
    output.append("="*80)

    # Render bars in rows
    for row_start in range(0, bars_to_render, bars_per_row):
        row_end = min(row_start + bars_per_row, bars_to_render)
        bars_in_row = row_end - row_start

        if row_start > 0:
            output.append("")  # Empty line between rows

        # Create bar lines for each string
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        bar_lines = {i: [] for i in range(1, 7)}
        bar_width = chars_per_beat * 4

        # Initialize bars for this row
        for string_num in range(1, 7):
            bar_lines[string_num] = ['-'] * (bar_width * bars_in_row)

        # Place notes for all bars in this row
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx
            bar_start = bar_num * ticks_per_bar
            bar_end = (bar_num + 1) * ticks_per_bar
            bar_offset = bar_idx * bar_width

            for string_num in range(1, 7):
                for tick, fret in strings[string_num]:
                    if bar_start <= tick < bar_end:
                        tick_in_bar = tick - bar_start
                        pos = int((tick_in_bar / ticks_per_bar) * bar_width) + bar_offset
                        pos = min(pos, len(bar_lines[string_num]) - 1)

                        fret_str = str(fret)
                        for i, char in enumerate(fret_str):
                            if pos + i < len(bar_lines[string_num]):
                                bar_lines[string_num][pos + i] = char

        # Print strings from high to low
        for i, string_num in enumerate([1, 2, 3, 4, 5, 6]):
            # Add bar separators between bars
            line_parts = []
            for bar_idx in range(bars_in_row):
                start = bar_idx * bar_width
                end = (bar_idx + 1) * bar_width
                line_parts.append(''.join(bar_lines[string_num][start:end]))
            line = '|'.join(line_parts)
            output.append(f"{string_names[i]}|{line}|")

        # Add bar markers below
        bar_markers = ' |'
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx + 1
            bar_markers += f"Bar {bar_num}".center(bar_width)
            if bar_idx < bars_in_row - 1:
                bar_markers += '|'
        bar_markers += '|'
        output.append(bar_markers)

    if total_bars > max_bars:
        output.append(f"\n... ({total_bars - max_bars} more bars not shown)")

    return '\n'.join(output)


def render_as_tablature(events, ticks_per_beat=480, max_bars=4, chars_per_beat=8, bars_per_row=1):
    """
    Render TAB events as six-string guitar tablature.

    Args:
        events: List of Event objects (must include TAB and TIME_SHIFT events)
        ticks_per_beat: MIDI ticks per quarter note (default 480)
        max_bars: Maximum number of bars to render
        chars_per_beat: Number of characters per beat

    Returns:
        String containing tablature notation
    """
    # Initialize 6 strings (high E to low E)
    # String numbering: 1=high E, 2=B, 3=G, 4=D, 5=A, 6=low E
    strings = {i: [] for i in range(1, 7)}

    current_tick = 0
    ticks_per_bar = ticks_per_beat * 4  # Assume 4/4 time

    # Parse events into timeline
    for event in events:
        if event.type == 'TIME_SHIFT':
            current_tick += event.delta
        elif event.type == 'TAB':
            strings[event.string].append((current_tick, event.fret))

    # Calculate total bars
    max_tick = max(current_tick, 1)
    total_bars = (max_tick + ticks_per_bar - 1) // ticks_per_bar
    bars_to_render = min(total_bars, max_bars)

    # Build tablature string
    output = []
    output.append("Guitar Tablature (Standard Tuning: EADGBE)")
    output.append("="*80)

    # Render bars in rows
    for row_start in range(0, bars_to_render, bars_per_row):
        row_end = min(row_start + bars_per_row, bars_to_render)
        bars_in_row = row_end - row_start

        if row_start > 0:
            output.append("")  # Empty line between rows

        # Create bar lines for each string
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        bar_lines = {i: [] for i in range(1, 7)}
        bar_width = chars_per_beat * 4

        # Initialize bars for this row
        for string_num in range(1, 7):
            bar_lines[string_num] = ['-'] * (bar_width * bars_in_row)

        # Place notes for all bars in this row
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx
            bar_start = bar_num * ticks_per_bar
            bar_end = (bar_num + 1) * ticks_per_bar
            bar_offset = bar_idx * bar_width

            for string_num in range(1, 7):
                for tick, fret in strings[string_num]:
                    if bar_start <= tick < bar_end:
                        tick_in_bar = tick - bar_start
                        pos = int((tick_in_bar / ticks_per_bar) * bar_width) + bar_offset
                        pos = min(pos, len(bar_lines[string_num]) - 1)

                        fret_str = str(fret)
                        for i, char in enumerate(fret_str):
                            if pos + i < len(bar_lines[string_num]):
                                bar_lines[string_num][pos + i] = char

        # Print strings from high to low
        for i, string_num in enumerate([1, 2, 3, 4, 5, 6]):
            # Add bar separators between bars
            line_parts = []
            for bar_idx in range(bars_in_row):
                start = bar_idx * bar_width
                end = (bar_idx + 1) * bar_width
                line_parts.append(''.join(bar_lines[string_num][start:end]))
            line = '|'.join(line_parts)
            output.append(f"{string_names[i]}|{line}|")

        # Add bar markers below
        bar_markers = ' |'
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx + 1
            bar_markers += f"Bar {bar_num}".center(bar_width)
            if bar_idx < bars_in_row - 1:
                bar_markers += '|'
        bar_markers += '|'
        output.append(bar_markers)

    if total_bars > max_bars:
        output.append(f"\n... ({total_bars - max_bars} more bars not shown)")

    return '\n'.join(output)


def render_dadagp_tokens_as_notes(tokens, ticks_per_beat=480, max_bars=4, chars_per_beat=8, bars_per_row=1):
    """
    Render DadaGP tokens as single-line note notation (like tablature with note names).

    Args:
        tokens: List of DadaGP token objects
        ticks_per_beat: MIDI ticks per quarter note
        max_bars: Maximum number of bars to render
        chars_per_beat: Number of characters per beat
        bars_per_row: Number of bars per row

    Returns:
        String containing note notation
    """
    from src.dadagp_parser import NoteToken, WaitToken, MetadataToken, compute_pitch

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Extract downtune
    downtune = 0
    for token in tokens:
        if isinstance(token, MetadataToken) and token.key == 'downtune':
            downtune = int(token.value)

    # Parse tokens
    notes = []
    current_tick = 0

    for token in tokens:
        if isinstance(token, WaitToken):
            current_tick += token.ticks
        elif isinstance(token, NoteToken):
            pitch = compute_pitch(token.string, token.fret, downtune)
            notes.append((current_tick, pitch))

    # Calculate bars
    ticks_per_bar = ticks_per_beat * 4
    max_tick = max(current_tick, 1) if notes else ticks_per_beat * 4
    total_bars = (max_tick + ticks_per_bar - 1) // ticks_per_bar
    bars_to_render = min(total_bars, max_bars)

    # Build notation
    output = []
    output.append("Note Notation (like tablature, showing note names) - from DadaGP tokens")
    output.append("="*80)

    # Render bars in rows
    for row_start in range(0, bars_to_render, bars_per_row):
        row_end = min(row_start + bars_per_row, bars_to_render)
        bars_in_row = row_end - row_start

        if row_start > 0:
            output.append("")

        bar_width = chars_per_beat * 4
        note_line = ['-'] * (bar_width * bars_in_row)

        # Place notes
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx
            bar_start = bar_num * ticks_per_bar
            bar_end = (bar_num + 1) * ticks_per_bar
            bar_offset = bar_idx * bar_width

            for tick, pitch in notes:
                if bar_start <= tick < bar_end:
                    tick_in_bar = tick - bar_start
                    pos = int((tick_in_bar / ticks_per_bar) * bar_width) + bar_offset
                    pos = min(pos, len(note_line) - 1)

                    octave = (pitch // 12) - 1
                    note = note_names[pitch % 12]
                    note_str = f"{note}{octave}"

                    for i, char in enumerate(note_str):
                        if pos + i < len(note_line):
                            note_line[pos + i] = char

        # Print note line with bar separators
        line_parts = []
        for bar_idx in range(bars_in_row):
            start = bar_idx * bar_width
            end = (bar_idx + 1) * bar_width
            line_parts.append(''.join(note_line[start:end]))
        line = '|'.join(line_parts)
        output.append(' |' + line + '|')

        # Add bar markers
        bar_markers = ' |'
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx + 1
            bar_markers += f"Bar {bar_num}".center(bar_width)
            if bar_idx < bars_in_row - 1:
                bar_markers += '|'
        bar_markers += '|'
        output.append(bar_markers)

    if total_bars > max_bars:
        output.append(f"\n... ({total_bars - max_bars} more bars not shown)")

    return '\n'.join(output)


def render_as_notes(events, ticks_per_beat=480, max_bars=4, chars_per_beat=8, bars_per_row=1):
    """
    Render NOTE_ON/OFF events as single-line note notation (like tablature with note names).

    Args:
        events: List of Event objects (must include NOTE_ON, NOTE_OFF, TIME_SHIFT)
        ticks_per_beat: MIDI ticks per quarter note
        max_bars: Maximum number of bars to render
        chars_per_beat: Number of characters per beat
        bars_per_row: Number of bars per row

    Returns:
        String containing note notation
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Parse events into timeline
    notes = []
    current_tick = 0

    for event in events:
        if event.type == 'TIME_SHIFT':
            current_tick += event.delta
        elif event.type == 'NOTE_ON':
            notes.append((current_tick, event.pitch))

    # Calculate total bars
    ticks_per_bar = ticks_per_beat * 4
    max_tick = max(current_tick, 1) if notes else ticks_per_beat * 4
    total_bars = (max_tick + ticks_per_bar - 1) // ticks_per_bar
    bars_to_render = min(total_bars, max_bars)

    # Build notation string
    output = []
    output.append("Note Notation (like tablature, showing note names)")
    output.append("="*80)

    # Render bars in rows
    for row_start in range(0, bars_to_render, bars_per_row):
        row_end = min(row_start + bars_per_row, bars_to_render)
        bars_in_row = row_end - row_start

        if row_start > 0:
            output.append("")

        bar_width = chars_per_beat * 4
        note_line = ['-'] * (bar_width * bars_in_row)

        # Place notes
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx
            bar_start = bar_num * ticks_per_bar
            bar_end = (bar_num + 1) * ticks_per_bar
            bar_offset = bar_idx * bar_width

            for tick, pitch in notes:
                if bar_start <= tick < bar_end:
                    tick_in_bar = tick - bar_start
                    pos = int((tick_in_bar / ticks_per_bar) * bar_width) + bar_offset
                    pos = min(pos, len(note_line) - 1)

                    octave = (pitch // 12) - 1
                    note = note_names[pitch % 12]
                    note_str = f"{note}{octave}"

                    for i, char in enumerate(note_str):
                        if pos + i < len(note_line):
                            note_line[pos + i] = char

        # Print note line with bar separators
        line_parts = []
        for bar_idx in range(bars_in_row):
            start = bar_idx * bar_width
            end = (bar_idx + 1) * bar_width
            line_parts.append(''.join(note_line[start:end]))
        line = '|'.join(line_parts)
        output.append(' |' + line + '|')

        # Add bar markers
        bar_markers = ' |'
        for bar_idx in range(bars_in_row):
            bar_num = row_start + bar_idx + 1
            bar_markers += f"Bar {bar_num}".center(bar_width)
            if bar_idx < bars_in_row - 1:
                bar_markers += '|'
        bar_markers += '|'
        output.append(bar_markers)

    if total_bars > max_bars:
        output.append(f"\n... ({total_bars - max_bars} more bars not shown)")

    return '\n'.join(output)


def visualize_midi_and_jams(midi_path, jams_path, max_bars=4, chars_per_beat=8):
    """
    Convenience function to visualize both tablature and note notation from files.

    Args:
        midi_path: Path to MIDI file
        jams_path: Path to JAMS file
        max_bars: Maximum number of bars to render
        chars_per_beat: Number of characters per beat

    Returns:
        Tuple of (tablature_string, notes_string)
    """
    from src.midi_utils import extract_note_on_off_events, extract_tab_events_from_jams

    input_events = extract_note_on_off_events(midi_path)
    output_events = extract_tab_events_from_jams(jams_path)

    tab_viz = render_as_tablature(output_events, max_bars=max_bars, chars_per_beat=chars_per_beat)
    note_viz = render_as_notes(input_events, max_bars=max_bars, chars_per_beat=chars_per_beat)

    return tab_viz, note_viz
