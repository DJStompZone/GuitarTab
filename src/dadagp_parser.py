"""
Parser for DadaGP token files (.tokens.txt).

Transforms DadaGP tokens into our NOTE_ON/NOTE_OFF/TIME_SHIFT/TAB format.
"""

from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path


# ============================================================================
# DadaGP Token Types (raw parsed tokens)
# ============================================================================

@dataclass(frozen=True)
class DadaGPToken:
    """Base class for all DadaGP tokens."""
    type: str


@dataclass(frozen=True)
class MetadataToken(DadaGPToken):
    """Metadata tokens like artist, tempo, downtune."""
    type: Literal["metadata"] = "metadata"
    key: str = ""
    value: str = ""


@dataclass(frozen=True)
class StructuralToken(DadaGPToken):
    """Structural tokens like start, new_measure."""
    type: Literal["structural"] = "structural"
    name: str = ""


@dataclass(frozen=True)
class NoteToken(DadaGPToken):
    """Note token: instrument:note:s{string}:f{fret}."""
    type: Literal["note"] = "note"
    instrument: str = ""  # leads, bass, clean, distorted, etc.
    string: int = 0
    fret: int = 0


@dataclass(frozen=True)
class WaitToken(DadaGPToken):
    """Wait token: wait:{ticks}."""
    type: Literal["wait"] = "wait"
    ticks: int = 0


@dataclass(frozen=True)
class EffectToken(DadaGPToken):
    """Effect token: nfx:{effect} or bfx:{effect}."""
    type: Literal["effect"] = "effect"
    effect_type: str = ""  # nfx or bfx
    effect_name: str = ""  # tie, bend, etc.


# ============================================================================
# Our Output Event Types (same as midi_utils.py)
# ============================================================================

@dataclass(frozen=True)
class Event:
    """Base event class."""
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


# ============================================================================
# Intermediate Data Structures
# ============================================================================

@dataclass(frozen=True)
class ActiveNote:
    """A note that is currently being played."""
    pitch: int
    string: int
    fret: int
    start_time: int


@dataclass(frozen=True)
class DadaGPMetadata:
    """Metadata extracted from DadaGP file."""
    artist: str = "unknown"
    tempo: int = 120
    downtune: int = 0


# Standard guitar tuning (MIDI pitches for open strings)
# E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]


# ============================================================================
# Token Parsing
# ============================================================================

def parse_dadagp_token(line: str) -> Optional[DadaGPToken]:
    """
    Parse a single line from DadaGP tokens file.

    Args:
        line: Single line from .tokens.txt file

    Returns:
        Parsed token or None if invalid
    """
    line = line.strip()
    if not line:
        return None

    # Metadata tokens: key:value
    if ":" in line and not line.startswith(("leads:", "bass:", "clean:", "distorted:", "pads:", "drums:")):
        parts = line.split(":", 1)
        if len(parts) == 2:
            key, value = parts
            if key in ["artist", "tempo", "downtune"]:
                return MetadataToken(key=key, value=value)

    # Structural tokens
    if line in ["start", "new_measure", "end"]:
        return StructuralToken(name=line)

    # Note tokens: instrument:note:s{string}:f{fret}
    if ":note:s" in line and ":f" in line:
        parts = line.split(":")
        if len(parts) >= 4:
            instrument = parts[0]
            # Extract string number (e.g., "s1" -> 1)
            string_str = parts[2]
            if string_str.startswith("s"):
                string_num = int(string_str[1:])
            else:
                return None

            # Extract fret number (e.g., "f0" -> 0)
            fret_str = parts[3]
            if fret_str.startswith("f"):
                fret_num = int(fret_str[1:])
            else:
                return None

            return NoteToken(instrument=instrument, string=string_num, fret=fret_num)

    # Wait tokens: wait:{ticks}
    if line.startswith("wait:"):
        ticks_str = line.split(":", 1)[1]
        return WaitToken(ticks=int(ticks_str))

    # Effect tokens: nfx:{effect} or bfx:{effect}
    if line.startswith(("nfx:", "bfx:")):
        parts = line.split(":", 1)
        if len(parts) == 2:
            effect_type, effect_name = parts
            return EffectToken(effect_type=effect_type, effect_name=effect_name)

    # Unknown token
    return None


def parse_dadagp_file(file_path: str) -> list[DadaGPToken]:
    """
    Parse entire DadaGP tokens file.

    Args:
        file_path: Path to .tokens.txt file

    Returns:
        List of parsed tokens
    """
    tokens: list[DadaGPToken] = []

    with open(file_path, "r") as f:
        for line in f:
            token = parse_dadagp_token(line)
            if token is not None:
                tokens.append(token)

    return tokens


# ============================================================================
# Token Transformation
# ============================================================================

def compute_pitch(string: int, fret: int, downtune: int = 0) -> int:
    """
    Compute MIDI pitch from string and fret.

    Args:
        string: String number (1-6 for standard guitar)
        fret: Fret number (0-24)
        downtune: Semitones to detune (0 = standard tuning)

    Returns:
        MIDI pitch
    """
    if string < 1 or string > 6:
        raise ValueError(f"Invalid string number: {string}")

    # Strings are numbered 1-6, but array is 0-indexed
    open_pitch = STANDARD_TUNING[string - 1]
    return open_pitch + fret - downtune


def dadagp_to_events(
    dadagp_tokens: list[DadaGPToken],
) -> tuple[list[Event], list[Event], list[tuple[int, int]]]:
    """
    Transform DadaGP tokens to our event format.

    Input sequence:  NOTE_ON, NOTE_OFF, TIME_SHIFT
    Output sequence: NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB

    Args:
        dadagp_tokens: List of parsed DadaGP tokens

    Returns:
        Tuple of (input_events, output_events, bar_positions)
        bar_positions: List of (input_idx, output_idx) where each bar starts
    """
    # Extract metadata
    metadata = DadaGPMetadata()
    for token in dadagp_tokens:
        if isinstance(token, MetadataToken):
            if token.key == "artist":
                metadata = DadaGPMetadata(
                    artist=token.value,
                    tempo=metadata.tempo,
                    downtune=metadata.downtune
                )
            elif token.key == "tempo":
                metadata = DadaGPMetadata(
                    artist=metadata.artist,
                    tempo=int(token.value),
                    downtune=metadata.downtune
                )
            elif token.key == "downtune":
                metadata = DadaGPMetadata(
                    artist=metadata.artist,
                    tempo=metadata.tempo,
                    downtune=int(token.value)
                )

    # Track active notes (notes that haven't been turned off yet)
    active_notes: list[ActiveNote] = []
    current_time = 0

    input_events: list[Event] = []
    output_events: list[Event] = []
    bar_positions: list[tuple[int, int]] = []

    # Process tokens
    for token in dadagp_tokens:
        # Track bar boundaries
        if isinstance(token, StructuralToken):
            if token.name == "new_measure":
                bar_positions.append((len(input_events), len(output_events)))
            continue

        if isinstance(token, NoteToken):
            # Compute pitch
            pitch = compute_pitch(token.string, token.fret, metadata.downtune)

            # Add NOTE_ON event
            input_events.append(NoteOnEvent(pitch=pitch))
            output_events.append(NoteOnEvent(pitch=pitch))

            # Add TAB event to output only
            output_events.append(TabEvent(string=token.string, fret=token.fret))

            # Track this note as active
            active_notes.append(
                ActiveNote(
                    pitch=pitch,
                    string=token.string,
                    fret=token.fret,
                    start_time=current_time
                )
            )

        elif isinstance(token, WaitToken):
            # Turn off all active notes (simplification: assume all notes end at wait)
            # This is a reasonable assumption for guitar tablature
            for note in active_notes:
                input_events.append(NoteOffEvent(pitch=note.pitch))
                output_events.append(NoteOffEvent(pitch=note.pitch))

            # Clear active notes
            active_notes = []

            # Add TIME_SHIFT
            input_events.append(TimeShiftEvent(delta=token.ticks))
            output_events.append(TimeShiftEvent(delta=token.ticks))

            # Advance time
            current_time += token.ticks

    # Turn off any remaining active notes at the end
    for note in active_notes:
        input_events.append(NoteOffEvent(pitch=note.pitch))
        output_events.append(NoteOffEvent(pitch=note.pitch))

    return input_events, output_events, bar_positions


def parse_dadagp_file_to_events(file_path: str) -> tuple[list[Event], list[Event], list[tuple[int, int]]]:
    """
    Parse DadaGP tokens file and convert to our event format.

    This is the main entry point for using DadaGP token files.

    Args:
        file_path: Path to .tokens.txt file

    Returns:
        Tuple of (input_events, output_events, bar_positions)
        bar_positions: List of (input_idx, output_idx) where each bar starts
    """
    tokens = parse_dadagp_file(file_path)
    return dadagp_to_events(tokens)
