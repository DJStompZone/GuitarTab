"""
MIDI utilities for reading and writing MIDI files.
Based on the original utils.py but refactored for modularity.
"""

import numpy as np
import miditoolkit
import sys
from pathlib import Path

# Add parent directory to import chord_recognition
sys.path.append(str(Path(__file__).parent.parent.parent))
import chord_recognition

# Parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=np.int32)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# Parameters for output
DEFAULT_RESOLUTION = 480


class Item:
    """General storage for MIDI items."""

    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return f'Item(name={self.name}, start={self.start}, end={self.end}, velocity={self.velocity}, pitch={self.pitch})'


class Event:
    """Storage for MIDI events."""

    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return f'Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})'


def read_items(file_path):
    """
    Read notes and tempo changes from MIDI file.

    Args:
        file_path: Path to MIDI file

    Returns:
        Tuple of (note_items, tempo_items)
    """
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)

    # Read notes
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(
            Item(
                name='Note',
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch,
            )
        )
    note_items.sort(key=lambda x: x.start)

    # Read tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(
            Item(
                name='Tempo',
                start=tempo.time,
                end=None,
                velocity=None,
                pitch=int(tempo.tempo),
            )
        )
    tempo_items.sort(key=lambda x: x.start)

    # Expand tempo to all beats
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(
                Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=existing_ticks[tick],
                )
            )
        else:
            output.append(
                Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=output[-1].pitch,
                )
            )
    tempo_items = output

    return note_items, tempo_items


def quantize_items(items, ticks=120):
    """
    Quantize MIDI items to grid.

    Args:
        items: List of Item objects
        ticks: Quantization grid resolution

    Returns:
        Quantized items
    """
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items


def extract_chords(items):
    """
    Extract chord information from note items.

    Args:
        items: List of Item objects (notes)

    Returns:
        List of chord items
    """
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(
            Item(
                name='Chord',
                start=chord[0],
                end=chord[1],
                velocity=None,
                pitch=chord[2].split('/')[0],
            )
        )
    return output


def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    """
    Group items by bars.

    Args:
        items: List of Item objects
        max_time: Maximum time in ticks
        ticks_per_bar: Number of ticks per bar

    Returns:
        List of groups, each group is a list of items in one bar
    """
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups


def item2event(groups):
    """
    Convert items to events.

    Args:
        groups: List of item groups (bars)

    Returns:
        List of Event objects
    """
    events = []
    n_downbeat = 0

    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue

        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(name='Bar', time=None, value='None', text=f'{n_downbeat}'))

        for item in groups[i][1:-1]:
            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags - item.start))
            events.append(
                Event(
                    name='Position',
                    time=item.start,
                    value=f'{index+1}/{DEFAULT_FRACTION}',
                    text=f'{item.start}',
                )
            )

            if item.name == 'Note':
                # Velocity
                velocity_index = (
                    np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side='right') - 1
                )
                events.append(
                    Event(
                        name='Note Velocity',
                        time=item.start,
                        value=velocity_index,
                        text=f'{item.velocity}/{DEFAULT_VELOCITY_BINS[velocity_index]}',
                    )
                )
                # Pitch
                events.append(
                    Event(name='Note On', time=item.start, value=item.pitch, text=f'{item.pitch}')
                )
                # Duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                events.append(
                    Event(
                        name='Note Duration',
                        time=item.start,
                        value=index,
                        text=f'{duration}/{DEFAULT_DURATION_BINS[index]}',
                    )
                )

            elif item.name == 'Chord':
                events.append(
                    Event(name='Chord', time=item.start, value=item.pitch, text=f'{item.pitch}')
                )

            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event(
                        'Tempo Value', item.start, tempo - DEFAULT_TEMPO_INTERVALS[0].start, None
                    )
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event(
                        'Tempo Value', item.start, tempo - DEFAULT_TEMPO_INTERVALS[1].start, None
                    )
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event(
                        'Tempo Value', item.start, tempo - DEFAULT_TEMPO_INTERVALS[2].start, None
                    )
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)

                events.append(tempo_style)
                events.append(tempo_value)

    return events


def extract_events_from_midi(midi_path, use_chords=False):
    """
    Extract events from a MIDI file.

    Args:
        midi_path: Path to MIDI file
        use_chords: Whether to extract chord information

    Returns:
        List of Event objects
    """
    note_items, tempo_items = read_items(midi_path)
    note_items = quantize_items(note_items)
    max_time = note_items[-1].end

    # Combine items
    items = tempo_items + note_items
    if use_chords:
        chord_items = extract_chords(note_items)
        items = items + chord_items

    groups = group_items(items, max_time)
    events = item2event(groups)

    return events


def midi_to_words(midi_path, event2word, use_chords=False):
    """
    Convert a MIDI file to a sequence of word indices (tokens).

    Args:
        midi_path: Path to MIDI file
        event2word: Dictionary mapping event strings to word indices
        use_chords: Whether to extract chord information

    Returns:
        List of word indices representing the MIDI file
    """
    # Extract events from MIDI
    events = extract_events_from_midi(midi_path, use_chords=use_chords)

    # Convert events to words
    words = []
    for event in events:
        event_str = f"{event.name}_{event.value}"
        if event_str in event2word:
            words.append(event2word[event_str])
        else:
            # Skip unknown events
            print(f"Warning: Unknown event '{event_str}' in prompt MIDI, skipping")

    return words


def word_to_event(words, word2event):
    """
    Convert word indices to events.

    Args:
        words: List of word indices
        word2event: Dictionary mapping word indices to event strings

    Returns:
        List of Event objects
    """
    events = []
    for word in words:
        if word not in word2event:
            print(f"Warning: Unknown word '{word}' in prompt MIDI, skipping")
            breakpoint()
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events


def write_midi(words, word2event, output_path, prompt_path=None):
    """
    Write MIDI file from word sequence.

    Args:
        words: List of word indices
        word2event: Dictionary mapping word indices to event strings
        output_path: Path to save MIDI file
        prompt_path: Optional path to prompt MIDI file
    """
    events = word_to_event(words, word2event)

    # Get downbeat and notes (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []

    for i in range(len(events) - 3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')

        elif (
            events[i].name == 'Position'
            and events[i + 1].name == 'Note Velocity'
            and events[i + 2].name == 'Note On'
            and events[i + 3].name == 'Note Duration'
        ):
            position = int(events[i].value.split('/')[0]) - 1
            velocity_index = int(events[i + 1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[velocity_index])
            pitch = int(events[i + 2].value)
            duration_index = int(events[i + 3].value)
            duration = DEFAULT_DURATION_BINS[duration_index]
            temp_notes.append([position, velocity, pitch, duration])

        elif events[i].name == 'Position' and events[i + 1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i + 1].value])

        elif (
            events[i].name == 'Position'
            and events[i + 1].name == 'Tempo Class'
            and events[i + 2].name == 'Tempo Value'
        ):
            position = int(events[i].value.split('/')[0]) - 1
            if events[i + 1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i + 2].value)
            elif events[i + 1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i + 2].value)
            elif events[i + 1].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i + 2].value)
            temp_tempos.append([position, tempo])

    # Get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4
    notes = []
    current_bar = 0

    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(
                current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int
            )
            st = flags[position]
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))

    # Get specific time for chords
    chords = []
    if len(temp_chords) > 0:
        current_bar = 0
        for chord in temp_chords:
            if chord == 'Bar':
                current_bar += 1
            else:
                position, value = chord
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(
                    current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int
                )
                st = flags[position]
                chords.append([st, value])

    # Get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(
                current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int
            )
            st = flags[position]
            tempos.append([int(st), value])

    # Write MIDI file
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        last_time = DEFAULT_RESOLUTION * 4 * 4

        # Shift notes
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)

        # Add tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION * 4 * 4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos

        # Write chords as markers
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0] + last_time)
                )
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION

        # Write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)

        # Write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes

        # Write chords as markers
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))

    # Save file
    midi.dump(output_path)
