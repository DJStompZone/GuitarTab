"""
Post-processing for model predictions.
"""

from typing import List, Tuple, Optional

# Assuming these modules are in the python path
from src.dadagp_parser import Event, NoteOnEvent, NoteOffEvent, TimeShiftEvent, TabEvent
from src.tab_dataset import Vocabulary, events_to_ids, event_to_token_string
from src.metrics import compute_tablature_accuracy


# ============================================================================
# Constants
# ============================================================================

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # Standard tuning: E2, A2, D3, G3, B3, E4
NUM_FRETS = 21  # Fret range 0-20 (matching vocabulary configuration in tab_dataset.py)


# ============================================================================
# Token <-> Event Conversion Helpers
# Note: event_to_token_string and events_to_ids are imported from tab_dataset.py
# We only need to implement the reverse conversion here
# ============================================================================

def token_string_to_event(token_str: str) -> Optional[Event]:
    """
    Convert token string to Event object.

    This is the inverse of event_to_token_string.

    Args:
        token_str: Token string (e.g., "NOTE_ON_60", "TAB_3_5")

    Returns:
        Event object or None if token is special or invalid
    """
    if not token_str or "_" not in token_str:
        # Special tokens (PAD, BOS, EOS, UNK) have no underscore
        return None

    try:
        if token_str.startswith("NOTE_ON_"):
            pitch = int(token_str.split("_")[2])
            return NoteOnEvent(pitch=pitch)
        elif token_str.startswith("NOTE_OFF_"):
            pitch = int(token_str.split("_")[2])
            return NoteOffEvent(pitch=pitch)
        elif token_str.startswith("TIME_SHIFT_"):
            delta = int(token_str.split("_")[2])
            return TimeShiftEvent(delta=delta)
        elif token_str.startswith("TAB_"):
            parts = token_str.split("_")
            string = int(parts[1])
            fret = int(parts[2])
            return TabEvent(string=string, fret=fret)
    except (IndexError, ValueError) as e:
        # Malformed token string
        print(f"  [WARNING] Failed to parse token '{token_str}': {e}")
        return None

    return None


def id_to_event(token_id: int, vocab: Vocabulary) -> Optional[Event]:
    """
    Convert token ID to Event object.

    Args:
        token_id: Token ID
        vocab: Vocabulary object

    Returns:
        Event object or None if ID is special or invalid
    """
    token_str = vocab.id_to_token.get(token_id)
    if not token_str:
        return None

    return token_string_to_event(token_str)


def ids_to_events(ids: List[int], vocab: Vocabulary) -> List[Event]:
    """
    Convert list of token IDs to list of Event objects.

    Note: Special tokens (PAD, BOS, EOS, UNK) are filtered out from the result.
    This is necessary for post-processing logic that operates on music events only.

    Args:
        ids: List of token IDs
        vocab: Vocabulary object

    Returns:
        List of Event objects (special tokens are excluded)
    """
    events = []

    for id in ids:
        event = id_to_event(id, vocab)
        if event is not None:
            events.append(event)

    return events







# ============================================================================
# Pitch and Fretboard Helpers
# ============================================================================

def pitch_to_frets(pitch: int) -> List[Tuple[int, int]]:
    """Finds all possible (string, fret) combinations for a given MIDI pitch."""
    possible_frets = []
    for string_idx, open_pitch in enumerate(GUITAR_TUNING):
        fret = pitch - open_pitch
        if 0 <= fret < NUM_FRETS:
            possible_frets.append((string_idx + 1, fret)) # 1-based string index
    return possible_frets


def find_closest_fret(pitch: int, original_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """
    Finds the best (string, fret) for a pitch, closest to an original position.
    "Closeness" is measured by Manhattan distance on the fretboard.
    """
    candidates = pitch_to_frets(pitch)
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

def get_token_info(token_id: int, vocab: Vocabulary) -> dict:
    """
    Get information from a token ID without converting to Event.

    Returns dict with:
        - type: 'NOTE_ON', 'NOTE_OFF', 'TAB', 'TIME_SHIFT', 'SPECIAL', or 'UNKNOWN'
        - pitch: int (for NOTE_ON/NOTE_OFF)
        - string: int (for TAB)
        - fret: int (for TAB)
        - delta: int (for TIME_SHIFT)
    """
    token_str = vocab.id_to_token.get(token_id, '')

    if not token_str or token_id in [vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.unk_id]:
        return {'type': 'SPECIAL', 'token_id': token_id}

    try:
        if token_str.startswith("NOTE_ON_"):
            pitch = int(token_str.split("_")[2])
            return {'type': 'NOTE_ON', 'pitch': pitch, 'token_id': token_id}
        elif token_str.startswith("NOTE_OFF_"):
            pitch = int(token_str.split("_")[2])
            return {'type': 'NOTE_OFF', 'pitch': pitch, 'token_id': token_id}
        elif token_str.startswith("TIME_SHIFT_"):
            delta = int(token_str.split("_")[2])
            return {'type': 'TIME_SHIFT', 'delta': delta, 'token_id': token_id}
        elif token_str.startswith("TAB_"):
            parts = token_str.split("_")
            string = int(parts[1])
            fret = int(parts[2])
            return {'type': 'TAB', 'string': string, 'fret': fret, 'token_id': token_id}
    except (IndexError, ValueError):
        pass

    return {'type': 'UNKNOWN', 'token_id': token_id}


def create_token_id(token_type: str, vocab: Vocabulary, **kwargs) -> int:
    """
    Create a token ID from components.

    Args:
        token_type: 'NOTE_ON', 'NOTE_OFF', 'TAB', or 'TIME_SHIFT'
        vocab: Vocabulary object
        **kwargs: pitch (for NOTE_ON/NOTE_OFF), string/fret (for TAB), delta (for TIME_SHIFT)
    """
    if token_type == 'NOTE_ON':
        token_str = f"NOTE_ON_{kwargs['pitch']}"
    elif token_type == 'NOTE_OFF':
        token_str = f"NOTE_OFF_{kwargs['pitch']}"
    elif token_type == 'TAB':
        token_str = f"TAB_{kwargs['string']}_{kwargs['fret']}"
    elif token_type == 'TIME_SHIFT':
        token_str = f"TIME_SHIFT_{kwargs['delta']}"
    else:
        return vocab.unk_id

    return vocab.token_to_id.get(token_str, vocab.unk_id)


def post_process_pitch_alignment(
    input_ids: List[int],
    pred_ids: List[int],
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    target_ids: Optional[List[int]] = None,
    original_pred_length: Optional[int] = None
) -> List[int]:
    """
    Aligns the pitches of the predicted sequence with the input sequence.
    Works directly on token IDs to preserve sequence length.

    Args:
        input_ids: Ground truth token IDs (NOTE_ON, NOTE_OFF, TIME_SHIFT), may include padding.
        pred_ids: Predicted token IDs from the model (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB), may include padding.
        input_vocab: Vocabulary for the input sequence.
        output_vocab: Vocabulary for the output sequence.
        target_ids: Optional target token IDs for comparison, may include padding.
        original_pred_length: Optional original prediction length (ignored, kept for compatibility).

    Returns:
        A new list of predicted token IDs with pitches aligned, same length as input.
    """
    print(f"--- Pitch Alignment Analysis ---")
    print(f"Input sequence length: {len(input_ids)} tokens")
    print(f"Prediction sequence length: {len(pred_ids)} tokens")

    # Work on a copy to avoid modifying the original
    aligned_ids = pred_ids.copy()

    # Parse all tokens in sequences
    input_tokens = [get_token_info(token_id, input_vocab) for token_id in input_ids]
    pred_tokens = [get_token_info(token_id, output_vocab) for token_id in aligned_ids]

    # Count token types
    num_input_note_ons = sum(1 for t in input_tokens if t['type'] == 'NOTE_ON')
    num_input_note_offs = sum(1 for t in input_tokens if t['type'] == 'NOTE_OFF')
    num_input_time_shifts = sum(1 for t in input_tokens if t['type'] == 'TIME_SHIFT')
    num_pred_note_ons_initial = sum(1 for t in pred_tokens if t['type'] == 'NOTE_ON')
    num_pred_note_offs_initial = sum(1 for t in pred_tokens if t['type'] == 'NOTE_OFF')
    num_pred_tabs_initial = sum(1 for t in pred_tokens if t['type'] == 'TAB')
    num_pred_time_shifts_initial = sum(1 for t in pred_tokens if t['type'] == 'TIME_SHIFT')

    print(f"Input NOTE_ON, NOTE_OFF, TIME_SHIFT tokens: {num_input_note_ons}, {num_input_note_offs}, {num_input_time_shifts}")
    print(f"Predicted NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB tokens (before alignment)")
    print(f"  {num_pred_note_ons_initial}, {num_pred_note_offs_initial}, {num_pred_time_shifts_initial}, {num_pred_tabs_initial}")

    if target_ids:
        target_tokens = [get_token_info(token_id, output_vocab) for token_id in target_ids]
        num_target_note_ons = sum(1 for t in target_tokens if t['type'] == 'NOTE_ON')
        num_target_note_offs = sum(1 for t in target_tokens if t['type'] == 'NOTE_OFF')
        num_target_time_shifts = sum(1 for t in target_tokens if t['type'] == 'TIME_SHIFT')
        num_target_tabs = sum(1 for t in target_tokens if t['type'] == 'TAB')
        print(f"Target NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB tokens (before alignment)")
        print(f"  {num_target_note_ons}, {num_target_note_offs}, {num_target_time_shifts}, {num_target_tabs}")

    # Extract NOTE_ON pitches and indices
    input_note_ons = [(i, t['pitch']) for i, t in enumerate(input_tokens) if t['type'] == 'NOTE_ON']
    pred_note_on_indices = [i for i, t in enumerate(pred_tokens) if t['type'] == 'NOTE_ON']

    print(f"Number of NOTE_ON events to align: {len(input_note_ons)}")

    num_notes_to_compare = min(len(input_note_ons), len(pred_note_on_indices))

    for i in range(num_notes_to_compare):
        input_idx, input_pitch = input_note_ons[i]
        pred_note_idx = pred_note_on_indices[i]
        pred_pitch = pred_tokens[pred_note_idx]['pitch']

        if input_pitch == pred_pitch:
            print(f"input NOTE_ON {i}: pitch={input_pitch}, pred NOTE_ON pitch={pred_pitch} MATCH!!")
            continue  # Pitch matches, no action needed

        print(f"input NOTE_ON {i}: pitch={input_pitch}, pred NOTE_ON pitch={pred_pitch}")

        # --- 1. Search ahead for a swap candidate ---
        found_swap = False
        search_window = range(i + 1, min(i + 6, len(pred_note_on_indices)))

        for j in search_window:
            candidate_idx = pred_note_on_indices[j]
            candidate_pitch = pred_tokens[candidate_idx]['pitch']

            if candidate_pitch == input_pitch:
                # Found a match to swap with.
                pitch1, pitch2 = pred_pitch, candidate_pitch

                # Swap NOTE_ON tokens
                aligned_ids[pred_note_idx] = create_token_id('NOTE_ON', output_vocab, pitch=pitch2)
                aligned_ids[candidate_idx] = create_token_id('NOTE_ON', output_vocab, pitch=pitch1)
                pred_tokens[pred_note_idx]['pitch'] = pitch2
                pred_tokens[candidate_idx]['pitch'] = pitch1

                # Swap TAB tokens
                tab_idx_1 = pred_note_idx + 1
                tab_idx_2 = candidate_idx + 1
                if (tab_idx_1 < len(aligned_ids) and tab_idx_2 < len(aligned_ids) and
                    pred_tokens[tab_idx_1]['type'] == 'TAB' and pred_tokens[tab_idx_2]['type'] == 'TAB'):
                    aligned_ids[tab_idx_1], aligned_ids[tab_idx_2] = aligned_ids[tab_idx_2], aligned_ids[tab_idx_1]
                    pred_tokens[tab_idx_1], pred_tokens[tab_idx_2] = pred_tokens[tab_idx_2], pred_tokens[tab_idx_1]

                # Swap NOTE_OFF tokens
                note_off_1_idx = None
                note_off_2_idx = None
                for k in range(pred_note_idx, len(pred_tokens)):
                    if pred_tokens[k]['type'] == 'NOTE_OFF' and pred_tokens[k]['pitch'] == pitch1:
                        note_off_1_idx = k
                        break
                for k in range(candidate_idx, len(pred_tokens)):
                    if pred_tokens[k]['type'] == 'NOTE_OFF' and pred_tokens[k]['pitch'] == pitch2:
                        note_off_2_idx = k
                        break

                if note_off_1_idx is not None and note_off_2_idx is not None:
                    aligned_ids[note_off_1_idx] = create_token_id('NOTE_OFF', output_vocab, pitch=pitch2)
                    aligned_ids[note_off_2_idx] = create_token_id('NOTE_OFF', output_vocab, pitch=pitch1)
                    pred_tokens[note_off_1_idx]['pitch'] = pitch2
                    pred_tokens[note_off_2_idx]['pitch'] = pitch1

                found_swap = True
                break  # Stop searching for swaps

        # --- 2. If no swap, force correct pitch and find new TAB ---
        if not found_swap:
            wrong_pitch = pred_pitch
            correct_pitch = input_pitch

            # Update NOTE_ON token
            aligned_ids[pred_note_idx] = create_token_id('NOTE_ON', output_vocab, pitch=correct_pitch)
            pred_tokens[pred_note_idx]['pitch'] = correct_pitch

            # Update corresponding NOTE_OFF token
            for k in range(pred_note_idx, len(pred_tokens)):
                if pred_tokens[k]['type'] == 'NOTE_OFF' and pred_tokens[k]['pitch'] == wrong_pitch:
                    aligned_ids[k] = create_token_id('NOTE_OFF', output_vocab, pitch=correct_pitch)
                    pred_tokens[k]['pitch'] = correct_pitch
                    break

            # Update TAB to closest valid fret
            tab_idx = pred_note_idx + 1
            if tab_idx < len(aligned_ids) and pred_tokens[tab_idx]['type'] == 'TAB':
                original_string = pred_tokens[tab_idx]['string']
                original_fret = pred_tokens[tab_idx]['fret']
                original_pos = (original_string, original_fret)

                new_pos = find_closest_fret(correct_pitch, original_pos)

                if new_pos:
                    aligned_ids[tab_idx] = create_token_id('TAB', output_vocab, string=new_pos[0], fret=new_pos[1])
                    pred_tokens[tab_idx]['string'] = new_pos[0]
                    pred_tokens[tab_idx]['fret'] = new_pos[1]


    # # ========================================================================
    # # Truncate extra notes beyond input length
    # # ========================================================================
    # if len(pred_note_on_indices) > len(input_note_ons):
    #     # Indices of NOTE_ONs to remove (beyond input length)
    #     notes_to_remove_indices = pred_note_on_indices[len(input_note_ons):]

    #     # Collect all event indices to remove (NOTE_ON + TAB + NOTE_OFF)
    #     indices_to_remove = set()

    #     for note_on_idx in notes_to_remove_indices:
    #         if note_on_idx >= len(pred_events):
    #             continue

    #         note_on_event = pred_events[note_on_idx]
    #         if not isinstance(note_on_event, NoteOnEvent):
    #             continue

    #         # Mark NOTE_ON for removal
    #         indices_to_remove.add(note_on_idx)

    #         # Mark TAB for removal (should be next token)
    #         tab_idx = note_on_idx + 1
    #         if tab_idx < len(pred_events) and isinstance(pred_events[tab_idx], TabEvent):
    #             indices_to_remove.add(tab_idx)

    #         # Mark corresponding NOTE_OFF for removal
    #         pitch = note_on_event.pitch
    #         try:
    #             note_off_idx = next(
    #                 k for k, e in enumerate(pred_events[note_on_idx:])
    #                 if isinstance(e, NoteOffEvent) and e.pitch == pitch
    #             ) + note_on_idx
    #             indices_to_remove.add(note_off_idx)
    #         except StopIteration:
    #             pass  # NOTE_OFF not found, skip it

    #     # Remove events in reverse order to preserve indices
    #     for idx in sorted(indices_to_remove, reverse=True):
    #         del pred_events[idx]

    #     num_removed_notes = len(notes_to_remove_indices)
    #     print(f"Truncated {num_removed_notes} extra notes (with their TAB and NOTE_OFF) beyond input length")


    # Print post-alignment statistics (re-parse tokens after modifications)
    aligned_tokens = [get_token_info(token_id, output_vocab) for token_id in aligned_ids]
    num_pred_note_ons_final = sum(1 for t in aligned_tokens if t['type'] == 'NOTE_ON')
    num_pred_note_offs_final = sum(1 for t in aligned_tokens if t['type'] == 'NOTE_OFF')
    num_pred_tabs_final = sum(1 for t in aligned_tokens if t['type'] == 'TAB')
    num_pred_time_shifts_final = sum(1 for t in aligned_tokens if t['type'] == 'TIME_SHIFT')

    print(f"Predicted NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB tokens (after alignment)")
    print(f"  {num_pred_note_ons_final}, {num_pred_note_offs_final}, {num_pred_time_shifts_final}, {num_pred_tabs_final}")

    # Calculate pitch accuracy
    aligned_note_ons = [(i, t['pitch']) for i, t in enumerate(aligned_tokens) if t['type'] == 'NOTE_ON']
    num_aligned_notes = min(len(input_note_ons), len(aligned_note_ons))

    if num_aligned_notes > 0:
        num_pitch_matches = sum(
            1 for i in range(num_aligned_notes)
            if input_note_ons[i][1] == aligned_note_ons[i][1]  # Compare pitches
        )
        pitch_accuracy = num_pitch_matches / num_aligned_notes * 100
        print(f"Pitch accuracy: {num_pitch_matches}/{num_aligned_notes} ({pitch_accuracy:.2f}%)")
    else:
        print(f"Pitch accuracy: N/A (no notes to compare)")

    # Note changes summary
    token_changes = {
        'NOTE_ON': num_pred_note_ons_final - num_pred_note_ons_initial,
        'NOTE_OFF': num_pred_note_offs_final - num_pred_note_offs_initial,
        'TAB': num_pred_tabs_final - num_pred_tabs_initial,
        'TIME_SHIFT': num_pred_time_shifts_final - num_pred_time_shifts_initial
    }

    print(f"Token count changes:")
    for token_type, change in token_changes.items():
        sign = '+' if change > 0 else ''
        print(f"  {token_type}: {sign}{change}")

    print(f"Final aligned sequence length: {len(aligned_ids)} tokens (same as input)")
    print(f"Sequence length preserved: {len(aligned_ids) == len(pred_ids)}")
    print(f"--- End of Pitch Alignment Analysis ---\n")

    return aligned_ids
