"""
MIDI utilities for reading and writing MIDI files.
Supports both REMI-style events and NOTE ON/OFF + TAB tokenization for guitar tablature.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import miditoolkit
import sys
import json
from pathlib import Path

# # Add parent directory to import chord_recognition
# sys.path.append(str(Path(__file__).parent.parent.parent))
# import chord_recognition

# Parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=np.int32)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# Parameters for output
DEFAULT_RESOLUTION = 480

# Guitar tablature parameters
NUM_STRINGS = 6
NUM_FRETS = 21
MAX_TIME_SHIFT = 500  # Maximum time shift in ticks (quantized)


# Event types for tokenization
@dataclass(frozen=True)
class Event:
    type: str


@dataclass(frozen=True)
class NoteOnEvent(Event):
    type: Literal["NOTE_ON"] = "NOTE_ON"
    pitch: int = 0


@dataclass(frozen=True)
class NoteOffEvent(Event):
    type: Literal["NOTE_OFF"] = "NOTE_OFF"
    pitch: int = 0


@dataclass(frozen=True)
class TimeShiftEvent(Event):
    type: Literal["TIME_SHIFT"] = "TIME_SHIFT"
    delta: int = 0


@dataclass(frozen=True)
class TabEvent(Event):
    type: Literal["TAB"] = "TAB"
    string: int = 0
    fret: int = 0


# Intermediate data structures for processing
@dataclass(frozen=True)
class RawNoteEvent:
    """Raw note event from MIDI before tokenization."""
    time: int
    event_type: Literal["NOTE_ON", "NOTE_OFF"]
    pitch: int
    velocity: int


@dataclass(frozen=True)
class RawTabEvent:
    """Raw tab event from JAMS before tokenization."""
    time: int
    string: int
    fret: int


@dataclass(frozen=True)
class RawJamsNote:
    """Raw note from JAMS with all information."""
    time: int
    duration: int
    string: int
    fret: int
    pitch: int  # computed from open_tuning + fret
    velocity: int


def extract_note_on_off_events(midi_path: str, quantize_ticks: int = 30) -> list[Event]:
    """
    Extract NOTE ON and NOTE OFF events with TIME SHIFT tokens from MIDI file.

    This format is simpler and more direct than REMI:
    - NOTE_ON_<pitch>
    - NOTE_OFF_<pitch>
    - TIME_SHIFT_<delta_ticks>

    Args:
        midi_path: Path to MIDI file
        quantize_ticks: Quantization resolution in ticks

    Returns:
        List of Event objects representing input sequence
    """
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)

    # Collect all note events
    note_events: list[RawNoteEvent] = []
    for note in midi_obj.instruments[0].notes:
        # Quantize timings
        start = (note.start // quantize_ticks) * quantize_ticks
        end = (note.end // quantize_ticks) * quantize_ticks

        note_events.append(
            RawNoteEvent(
                time=start,
                event_type="NOTE_ON",
                pitch=note.pitch,
                velocity=note.velocity,
            )
        )
        note_events.append(
            RawNoteEvent(
                time=end,
                event_type="NOTE_OFF",
                pitch=note.pitch,
                velocity=0,
            )
        )

    # Sort by time (NOTE_OFF after NOTE_ON at same time)
    note_events.sort(key=lambda x: (x.time, x.event_type == "NOTE_OFF"))

    # Convert to events with TIME SHIFT
    events: list[Event] = []
    current_time = 0

    for ne in note_events:
        # Add TIME_SHIFT if needed
        if ne.time > current_time:
            delta = ne.time - current_time
            # Split large time shifts into multiple tokens
            while delta > 0:
                shift = min(delta, MAX_TIME_SHIFT)
                events.append(TimeShiftEvent(delta=shift))
                current_time += shift
                delta -= shift

        # Add NOTE ON/OFF event
        if ne.event_type == "NOTE_ON":
            events.append(NoteOnEvent(pitch=ne.pitch))
        else:
            events.append(NoteOffEvent(pitch=ne.pitch))

    return events


def extract_tab_events_from_jams(
    jams_path: str, quantize_ticks: int = 30
) -> list[Event]:
    """
    Extract TAB tokens with TIME SHIFT from JAMS file.

    This format represents guitar tablature as:
    - TAB_<string>_<fret>
    - TIME_SHIFT_<delta_ticks>

    Args:
        jams_path: Path to JAMS file
        quantize_ticks: Quantization resolution in ticks

    Returns:
        List of Event objects representing output sequence
    """
    with open(jams_path, "r") as f:
        jams_data = json.load(f)

    # Collect all tab events
    tab_events: list[RawTabEvent] = []

    for annotation in jams_data["annotations"]:
        if annotation["namespace"] == "note_tab":
            string_index = annotation["sandbox"]["string_index"]
            open_tuning = annotation["sandbox"]["open_tuning"]

            for note_data in annotation["data"]:
                fret = note_data["value"]["fret"]
                time_ticks = int(note_data["time"])

                # Quantize timing
                time_ticks = (time_ticks // quantize_ticks) * quantize_ticks

                # Validate string and fret ranges
                if 1 <= string_index <= NUM_STRINGS and 0 <= fret < NUM_FRETS:
                    tab_events.append(
                        RawTabEvent(
                            time=time_ticks,
                            string=string_index,
                            fret=fret
                        )
                    )

    # Sort by time
    tab_events.sort(key=lambda x: x.time)

    # Convert to events with TIME SHIFT
    events: list[Event] = []
    current_time = 0

    for te in tab_events:
        # Add TIME_SHIFT if needed
        if te.time > current_time:
            delta = te.time - current_time
            # Split large time shifts into multiple tokens
            while delta > 0:
                shift = min(delta, MAX_TIME_SHIFT)
                events.append(TimeShiftEvent(delta=shift))
                current_time += shift
                delta -= shift

        # Add TAB event
        events.append(TabEvent(string=te.string, fret=te.fret))

    return events


def extract_paired_events_from_midi_and_jams(
    midi_path: str, jams_path: str, quantize_ticks: int = 30
) -> tuple[list[Event], list[Event]]:
    """
    Extract paired input/output events from MIDI and JAMS files.

    The output sequence is identical to input except with additional TAB tokens.
    Both sequences are processed in the same loop for consistency.

    Input:  NOTE_ON, NOTE_OFF, TIME_SHIFT
    Output: NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB<string,fret>

    Args:
        midi_path: Path to MIDI file
        jams_path: Path to JAMS file
        quantize_ticks: Quantization resolution in ticks

    Returns:
        Tuple of (input_events, output_events)
    """
    # Load MIDI data
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)

    # Load JAMS data
    with open(jams_path, "r") as f:
        jams_data = json.load(f)

    # Collect note events from MIDI
    note_events: list[RawNoteEvent] = []
    for note in midi_obj.instruments[0].notes:
        start = (note.start // quantize_ticks) * quantize_ticks
        end = (note.end // quantize_ticks) * quantize_ticks

        note_events.append(
            RawNoteEvent(
                time=start,
                event_type="NOTE_ON",
                pitch=note.pitch,
                velocity=note.velocity,
            )
        )
        note_events.append(
            RawNoteEvent(
                time=end,
                event_type="NOTE_OFF",
                pitch=note.pitch,
                velocity=0,
            )
        )

    # Collect tab events from JAMS
    tab_events: list[RawTabEvent] = []
    for annotation in jams_data["annotations"]:
        if annotation["namespace"] == "note_tab":
            string_index = annotation["sandbox"]["string_index"]

            for note_data in annotation["data"]:
                fret = note_data["value"]["fret"]
                time_ticks = int(note_data["time"])
                time_ticks = (time_ticks // quantize_ticks) * quantize_ticks

                if 1 <= string_index <= NUM_STRINGS and 0 <= fret < NUM_FRETS:
                    tab_events.append(
                        RawTabEvent(time=time_ticks, string=string_index, fret=fret)
                    )

    # Sort both by time
    note_events.sort(key=lambda x: (x.time, x.event_type == "NOTE_OFF"))
    tab_events.sort(key=lambda x: x.time)

    # Create a mapping from time to tab events for output sequence
    tab_map: dict[int, list[RawTabEvent]] = {}
    for te in tab_events:
        if te.time not in tab_map:
            tab_map[te.time] = []
        tab_map[te.time].append(te)

    # Generate both sequences in the same loop
    input_events: list[Event] = []
    output_events: list[Event] = []
    current_time = 0

    for ne in note_events:
        # Add TIME_SHIFT if needed (same for both sequences)
        if ne.time > current_time:
            delta = ne.time - current_time
            while delta > 0:
                shift = min(delta, MAX_TIME_SHIFT)
                input_events.append(TimeShiftEvent(delta=shift))
                output_events.append(TimeShiftEvent(delta=shift))
                current_time += shift
                delta -= shift

        # Add NOTE ON/OFF event (same for both sequences)
        if ne.event_type == "NOTE_ON":
            input_events.append(NoteOnEvent(pitch=ne.pitch))
            output_events.append(NoteOnEvent(pitch=ne.pitch))
        else:
            input_events.append(NoteOffEvent(pitch=ne.pitch))
            output_events.append(NoteOffEvent(pitch=ne.pitch))

        # Add TAB events for output sequence only (at NOTE_ON time)
        if ne.event_type == "NOTE_ON" and ne.time in tab_map:
            for te in tab_map[ne.time]:
                output_events.append(TabEvent(string=te.string, fret=te.fret))

    return input_events, output_events



class Item:
    """General storage for MIDI items."""

    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return f"Item(name={self.name}, start={self.start}, end={self.end}, velocity={self.velocity}, pitch={self.pitch})"


class Event:
    """Storage for MIDI events."""

    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return f"Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})"


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
                name="Note",
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
                name="Tempo",
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
                    name="Tempo",
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=existing_ticks[tick],
                )
            )
        else:
            output.append(
                Item(
                    name="Tempo",
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
                name="Chord",
                start=chord[0],
                end=chord[1],
                velocity=None,
                pitch=chord[2].split("/")[0],
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
        if "Note" not in [item.name for item in groups[i][1:-1]]:
            continue

        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(name="Bar", time=None, value="None", text=f"{n_downbeat}"))

        for item in groups[i][1:-1]:
            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags - item.start))
            events.append(
                Event(
                    name="Position",
                    time=item.start,
                    value=f"{index + 1}/{DEFAULT_FRACTION}",
                    text=f"{item.start}",
                )
            )

            if item.name == "Note":
                # Velocity
                velocity_index = (
                    np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side="right")
                    - 1
                )
                events.append(
                    Event(
                        name="Note Velocity",
                        time=item.start,
                        value=velocity_index,
                        text=f"{item.velocity}/{DEFAULT_VELOCITY_BINS[velocity_index]}",
                    )
                )
                # Pitch
                events.append(
                    Event(
                        name="Note On",
                        time=item.start,
                        value=item.pitch,
                        text=f"{item.pitch}",
                    )
                )
                # Duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                events.append(
                    Event(
                        name="Note Duration",
                        time=item.start,
                        value=index,
                        text=f"{duration}/{DEFAULT_DURATION_BINS[index]}",
                    )
                )

            elif item.name == "Chord":
                events.append(
                    Event(
                        name="Chord",
                        time=item.start,
                        value=item.pitch,
                        text=f"{item.pitch}",
                    )
                )

            elif item.name == "Tempo":
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event("Tempo Class", item.start, "slow", None)
                    tempo_value = Event(
                        "Tempo Value",
                        item.start,
                        tempo - DEFAULT_TEMPO_INTERVALS[0].start,
                        None,
                    )
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event("Tempo Class", item.start, "mid", None)
                    tempo_value = Event(
                        "Tempo Value",
                        item.start,
                        tempo - DEFAULT_TEMPO_INTERVALS[1].start,
                        None,
                    )
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event("Tempo Class", item.start, "fast", None)
                    tempo_value = Event(
                        "Tempo Value",
                        item.start,
                        tempo - DEFAULT_TEMPO_INTERVALS[2].start,
                        None,
                    )
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event("Tempo Class", item.start, "slow", None)
                    tempo_value = Event("Tempo Value", item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event("Tempo Class", item.start, "fast", None)
                    tempo_value = Event("Tempo Value", item.start, 59, None)

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
        event_name, event_value = word2event.get(word).split("_")
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
        if events[i].name == "Bar" and i > 0:
            temp_notes.append("Bar")
            temp_chords.append("Bar")
            temp_tempos.append("Bar")

        elif (
            events[i].name == "Position"
            and events[i + 1].name == "Note Velocity"
            and events[i + 2].name == "Note On"
            and events[i + 3].name == "Note Duration"
        ):
            position = int(events[i].value.split("/")[0]) - 1
            velocity_index = int(events[i + 1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[velocity_index])
            pitch = int(events[i + 2].value)
            duration_index = int(events[i + 3].value)
            duration = DEFAULT_DURATION_BINS[duration_index]
            temp_notes.append([position, velocity, pitch, duration])

        elif events[i].name == "Position" and events[i + 1].name == "Chord":
            position = int(events[i].value.split("/")[0]) - 1
            temp_chords.append([position, events[i + 1].value])

        elif (
            events[i].name == "Position"
            and events[i + 1].name == "Tempo Class"
            and events[i + 2].name == "Tempo Value"
        ):
            position = int(events[i].value.split("/")[0]) - 1
            if events[i + 1].value == "slow":
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i + 2].value)
            elif events[i + 1].value == "mid":
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i + 2].value)
            elif events[i + 1].value == "fast":
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i + 2].value)
            temp_tempos.append([position, tempo])

    # Get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4
    notes = []
    current_bar = 0

    for note in temp_notes:
        if note == "Bar":
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(
                current_bar_st,
                current_bar_et,
                DEFAULT_FRACTION,
                endpoint=False,
                dtype=int,
            )
            st = flags[position]
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))

    # Get specific time for chords
    chords = []
    if len(temp_chords) > 0:
        current_bar = 0
        for chord in temp_chords:
            if chord == "Bar":
                current_bar += 1
            else:
                position, value = chord
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(
                    current_bar_st,
                    current_bar_et,
                    DEFAULT_FRACTION,
                    endpoint=False,
                    dtype=int,
                )
                st = flags[position]
                chords.append([st, value])

    # Get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == "Bar":
            current_bar += 1
        else:
            position, value = tempo
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(
                current_bar_st,
                current_bar_et,
                DEFAULT_FRACTION,
                endpoint=False,
                dtype=int,
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
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0])
                )

    # Save file
    midi.dump(output_path)
