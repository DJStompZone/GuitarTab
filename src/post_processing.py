"""
Post-processing for model predictions.
"""

from typing import List, Tuple, Optional

# Assuming these modules are in the python path
from src.dadagp_parser import Event, NoteOnEvent, NoteOffEvent, TimeShiftEvent, TabEvent
from src.tab_dataset import Vocabulary


# ============================================================================
# Constants
# ============================================================================

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # Standard tuning: E2, A2, D3, G3, B3, E4
# num_frets is passed from config by callers; default 25 to match configs/data/*.yaml


# ============================================================================
# Token <-> Event Conversion Helpers
# ============================================================================

def id_to_event(token_id: int, vocab: Vocabulary) -> Optional[Event]:
    """Converts a single token ID to an Event object."""
    token_str = vocab.id_to_token.get(token_id)
    if not token_str or "_" not in token_str:
        return None  # Skip special tokens like PAD, BOS, EOS

    parts = token_str.split('_')
    token_type = parts[0]

    try:
        if token_type == "NOTE" and parts[1] == "ON":
            return NoteOnEvent(pitch=int(parts[2]))
        elif token_type == "NOTE" and parts[1] == "OFF":
            return NoteOffEvent(pitch=int(parts[2]))
        elif token_type == "TIME" and parts[1] == "SHIFT":
            return TimeShiftEvent(delta=int(parts[2]))
        elif token_type == "TAB":
            return TabEvent(string=int(parts[1]), fret=int(parts[2]))
    except (IndexError, ValueError):
        return None # Malformed token string
        
    return None


def ids_to_events(ids: List[int], vocab: Vocabulary) -> List[Event]:
    """Converts a list of token IDs to a list of Event objects."""
    return [event for event in (id_to_event(id, vocab) for id in ids) if event]


def event_to_token_string(event: Event) -> str:
    """Converts an Event object back to its token string representation."""
    if isinstance(event, NoteOnEvent):
        return f"NOTE_ON_{event.pitch}"
    elif isinstance(event, NoteOffEvent):
        return f"NOTE_OFF_{event.pitch}"
    elif isinstance(event, TimeShiftEvent):
        return f"TIME_SHIFT_{event.delta}"
    elif isinstance(event, TabEvent):
        return f"TAB_{event.string}_{event.fret}"
    else:
        # This case should ideally not be reached with valid Event objects
        return "UNK"


def events_to_ids(events: List[Event], vocab: Vocabulary) -> List[int]:
    """Converts a list of Event objects back to a list of token IDs."""
    return [vocab.token_to_id.get(event_to_token_string(event), vocab.unk_id) for event in events]


# ============================================================================
# Pitch and Fretboard Helpers
# ============================================================================

def pitch_to_frets(pitch: int, num_frets: int = 25) -> List[Tuple[int, int]]:
    """Finds all possible (string, fret) combinations for a given MIDI pitch. num_frets should match config."""
    possible_frets = []
    for string_idx, open_pitch in enumerate(GUITAR_TUNING):
        fret = pitch - open_pitch
        if 0 <= fret < num_frets:
            possible_frets.append((string_idx + 1, fret)) # 1-based string index
    return possible_frets


def find_closest_fret(pitch: int, original_pos: Tuple[int, int], num_frets: int = 25) -> Optional[Tuple[int, int]]:
    """
    Finds the best (string, fret) for a pitch, closest to an original position.
    "Closeness" is measured by Manhattan distance on the fretboard.
    """
    candidates = pitch_to_frets(pitch, num_frets)
    if not candidates:
        return None  # Pitch is not playable on the standard fretboard

    if len(candidates) == 1:
        return candidates[0]

    # Find the candidate with the minimum distance to the original position
    min_dist = float('inf')
    best_candidate = None
    
    for candidate in candidates:
        dist = abs(candidate[0] - original_pos[0]) + abs(candidate[1] - original_pos[1])
        if dist < min_dist:
            min_dist = dist
            best_candidate = candidate
            
    return best_candidate


# ============================================================================
# Pitch Alignment Post-Processing
# ============================================================================

def post_process_pitch_alignment(
    input_ids: List[int],
    pred_ids: List[int],
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    num_frets: int = 25,
) -> List[int]:
    """
    Aligns the pitches of the predicted sequence with the input sequence.
    This function respects the immutability of Event objects by creating new
    instances for modifications rather than changing them in-place.

    Args:
        input_ids: Ground truth token IDs (NOTE_ON, NOTE_OFF, TIME_SHIFT).
        pred_ids: Predicted token IDs from the model (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB).
        input_vocab: Vocabulary for the input sequence.
        output_vocab: Vocabulary for the output sequence.
        num_frets: Number of frets (valid 0..num_frets-1). Should match config (e.g. 25).

    Returns:
        A new list of predicted token IDs with pitches aligned.
    """
    input_events = ids_to_events(input_ids, input_vocab)
    pred_events = ids_to_events(pred_ids, output_vocab)

    # Count NOTE_ON tokens in input and prediction
    num_input_note_ons = sum(1 for e in input_events if isinstance(e, NoteOnEvent))
    num_pred_note_ons_initial = sum(1 for e in pred_events if isinstance(e, NoteOnEvent))

    print(f"--- Pitch Alignment Analysis ---")
    print(f"Input NOTE_ON tokens: {num_input_note_ons}")
    print(f"Predicted NOTE_ON tokens (before alignment): {num_pred_note_ons_initial}")
    
    input_note_ons = [e for e in input_events if isinstance(e, NoteOnEvent)]
    
    # Get indices of NoteOn events in the prediction list for easier manipulation
    pred_note_on_indices = [i for i, e in enumerate(pred_events) if isinstance(e, NoteOnEvent)]

    num_notes_to_compare = min(len(input_note_ons), len(pred_note_on_indices))

    for i in range(num_notes_to_compare):
        input_note = input_note_ons[i]
        pred_note_idx = pred_note_on_indices[i]
        
        # Make sure we're accessing a valid note
        if not isinstance(pred_events[pred_note_idx], NoteOnEvent): continue

        pred_note = pred_events[pred_note_idx]

        if input_note.pitch != pred_note.pitch:
            # --- 1. Search ahead for a swap candidate ---
            found_swap = False
            search_window = range(i + 1, min(i + 6, len(pred_note_on_indices)))
            
            for j in search_window:
                candidate_idx = pred_note_on_indices[j]
                candidate_note = pred_events[candidate_idx]
                if not isinstance(candidate_note, NoteOnEvent): continue

                if candidate_note.pitch == input_note.pitch:
                    # Found a match to swap with.
                    pitch1, pitch2 = pred_note.pitch, candidate_note.pitch

                    # Swap NOTE_ON events by creating new instances
                    pred_events[pred_note_idx] = NoteOnEvent(pitch=pitch2)
                    pred_events[candidate_idx] = NoteOnEvent(pitch=pitch1)

                    # Swap TAB events
                    tab_idx_1 = pred_note_idx + 1
                    tab_idx_2 = candidate_idx + 1
                    if tab_idx_1 < len(pred_events) and tab_idx_2 < len(pred_events) and \
                       isinstance(pred_events[tab_idx_1], TabEvent) and \
                       isinstance(pred_events[tab_idx_2], TabEvent):
                        pred_events[tab_idx_1], pred_events[tab_idx_2] = pred_events[tab_idx_2], pred_events[tab_idx_1]

                    # Swap NOTE_OFF pitches by creating new instances
                    try:
                        note_off_1_idx = next(k for k, e in enumerate(pred_events[pred_note_idx:]) if isinstance(e, NoteOffEvent) and e.pitch == pitch1) + pred_note_idx
                        note_off_2_idx = next(k for k, e in enumerate(pred_events[candidate_idx:]) if isinstance(e, NoteOffEvent) and e.pitch == pitch2) + candidate_idx
                        
                        # Create new NoteOffEvents with swapped pitches
                        pred_events[note_off_1_idx] = NoteOffEvent(pitch=pitch2)
                        pred_events[note_off_2_idx] = NoteOffEvent(pitch=pitch1)
                    except StopIteration:
                        pass # If a NOTE_OFF isn't found, we can't swap them.

                    found_swap = True
                    break # Stop searching for swaps

            # --- 2. If no swap, force correct pitch and find new TAB ---
            if not found_swap:
                wrong_pitch = pred_note.pitch
                correct_pitch = input_note.pitch
                
                # Update NOTE_ON by creating a new event
                pred_events[pred_note_idx] = NoteOnEvent(pitch=correct_pitch)
                
                # Update corresponding NOTE_OFF by creating a new event
                try:
                    note_off_idx = next(k for k, e in enumerate(pred_events[pred_note_idx:]) if isinstance(e, NoteOffEvent) and e.pitch == wrong_pitch) + pred_note_idx
                    pred_events[note_off_idx] = NoteOffEvent(pitch=correct_pitch)
                except StopIteration:
                    pass # No NOTE_OFF found, can't update

                # Update TAB to closest valid fret by creating a new event
                tab_idx = pred_note_idx + 1
                if tab_idx < len(pred_events) and isinstance(pred_events[tab_idx], TabEvent):
                    original_tab = pred_events[tab_idx]
                    original_pos = (original_tab.string, original_tab.fret)
                    
                    new_pos = find_closest_fret(correct_pitch, original_pos, num_frets)
                    
                    if new_pos:
                        pred_events[tab_idx] = TabEvent(string=new_pos[0], fret=new_pos[1])

    # Convert the modified event sequence back to token IDs
    return events_to_ids(pred_events, output_vocab)
