"""
Post-processing for model predictions.
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter

# Assuming these modules are in the python path
from src.dadagp_parser import Event, NoteOnEvent, NoteOffEvent, TimeShiftEvent, TabEvent
from src.tab_dataset import Vocabulary

# ============================================================================
# Constants
# ============================================================================

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # Standard tuning: E2, A2, D3, G3, B3, E4
# Note on DadaGP String Ordering:
# In DadaGP tokens, String 1 = Low E (40), String 6 = High E (64).
# This is opposite to standard guitar tab convention (where 1=High E),
# but matches the array index order of GUITAR_TUNING.

# num_frets is passed from config (e.g. configs/data/*.yaml) by callers; no hardcoded default here.
PITCH_ALIGNMENT_TIME_WINDOW = 960  # Ticks (approx 2 beats at 480 tpq)


# ============================================================================
# Token Info Helpers
# ============================================================================

def get_token_info(token_id: int, vocab: Vocabulary) -> dict:
    """
    Get parsed information from a token ID.
    """
    token_str = vocab.id_to_token.get(token_id, '')

    if not token_str or token_id in [vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.unk_id]:
        return {'type': 'SPECIAL', 'token_id': token_id, 'str': token_str}

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


# ============================================================================
# Pitch and Fretboard Helpers
# ============================================================================

def get_pitch_from_tab(string: int, fret: int, tuning_offset: int = 0) -> int:
    """
    Calculate MIDI pitch from string (1-6) and fret.
    
    Args:
        string: 1-based string index. 
                DadaGP Convention: 1=Low E, 6=High E.
        fret: Fret number
        tuning_offset: Global offset (Standard - Actual).
    """
    if 1 <= string <= 6:
        # DadaGP: String 1 -> Index 0 (E2), String 6 -> Index 5 (E4)
        tuning_idx = string - 1
        
        # Base pitch for this string in Standard Tuning
        base_pitch = GUITAR_TUNING[tuning_idx]
        
        # Adjust for global tuning offset
        actual_base_pitch = base_pitch - tuning_offset
        return actual_base_pitch + fret
    return 0  # Invalid


def pitch_to_frets(pitch: int, num_frets: int, tuning_offset: int = 0) -> List[Tuple[int, int]]:
    """
    Finds all possible (string, fret) combinations for a given MIDI pitch,
    considering the global tuning offset. num_frets should match config (e.g. 25 → frets 0-24).
    """
    possible_frets = []
    
    # Iterate through strings (Index 0=Low E=String 1, Index 5=High E=String 6)
    for tuning_idx, open_pitch_standard in enumerate(GUITAR_TUNING):
        # Calculate string number from tuning index (Index 0 -> String 1)
        string_num = tuning_idx + 1
        
        # Calculate the actual open pitch of this string
        open_pitch_actual = open_pitch_standard - tuning_offset
        
        fret = pitch - open_pitch_actual
        if 0 <= fret < num_frets:
            possible_frets.append((string_num, fret))
            
    return possible_frets


def find_closest_fret(
    pitch: int, original_pos: Tuple[int, int], num_frets: int, tuning_offset: int = 0
) -> Optional[Tuple[int, int]]:
    """
    Finds the best (string, fret) for a pitch, closest to an original position.
    """
    candidates = pitch_to_frets(pitch, num_frets, tuning_offset)
    if not candidates:
        return None  # Pitch is not playable

    if len(candidates) == 1:
        return candidates[0]

    # Find the candidate with the minimum distance to the original position
    min_dist = float('inf')
    best_candidate = candidates[0]
    
    for candidate in candidates:
        # Distance metric: simple Manhattan distance on string/fret grid
        dist = abs(candidate[0] - original_pos[0]) + abs(candidate[1] - original_pos[1])
        if dist < min_dist:
            min_dist = dist
            best_candidate = candidate
            
    return best_candidate


# ============================================================================
# Format Detection
# ============================================================================

def detect_output_format(ids: List[int], vocab: Vocabulary) -> str:
    """
    Detect whether output sequence is v1 (NOTE_ON+TAB) or v2 (TAB only) format.

    v1: NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT
    v2: TAB, TIME_SHIFT

    Returns:
        'v1' or 'v2'
    """
    note_on_count = 0
    tab_count = 0

    for token_id in ids[:100]:  # Sample first 100 tokens
        info = get_token_info(token_id, vocab)
        if info['type'] == 'NOTE_ON':
            note_on_count += 1
        elif info['type'] == 'TAB':
            tab_count += 1

    # If we see NOTE_ON tokens, it's v1
    # If we see TAB but no NOTE_ON, it's v2
    if note_on_count > 0:
        return 'v1'
    elif tab_count > 0:
        return 'v2'
    else:
        # Default to v1 if can't determine
        return 'v1'


# ============================================================================
# Analysis Helpers
# ============================================================================

def get_input_note_sequence(ids: List[int], vocab: Vocabulary) -> List[dict]:
    """
    Extract logical sequence of notes from Input IDs (Ground Truth).
    Input format is usually: NOTE_ON -> ... -> NOTE_OFF
    We track time.
    """
    notes = []
    current_time = 0
    
    for idx, token_id in enumerate(ids):
        info = get_token_info(token_id, vocab)
        
        if info['type'] == 'TIME_SHIFT':
            current_time += info['delta']
        elif info['type'] == 'NOTE_ON':
            notes.append({
                'idx': idx,
                'pitch': info['pitch'],
                'time': current_time,
                'token_id': token_id
            })
            
    return notes


def get_predicted_note_sequence(ids: List[int], vocab: Vocabulary, tuning_offset: int = 0) -> List[dict]:
    """
    Extract logical sequence of notes from Predicted IDs.
    
    Strategy:
    1. Scan for TAB tokens. TAB is the physical reality of the note.
    2. If a TAB is found, calculate its pitch (considering Tuning Offset).
    3. Look immediately backwards for a NOTE_ON (v1 format).
    """
    notes = []
    current_time = 0
    
    for idx, token_id in enumerate(ids):
        info = get_token_info(token_id, vocab)
        
        if info['type'] == 'TIME_SHIFT':
            current_time += info['delta']
        
        elif info['type'] == 'TAB':
            # Found a note event (physical)
            string = info['string']
            fret = info['fret']
            pitch = get_pitch_from_tab(string, fret, tuning_offset)
            
            note_entry = {
                'tab_idx': idx,
                'string': string,
                'fret': fret,
                'pitch': pitch,
                'time': current_time,
                'note_on_idx': None,
                'note_on_pitch': None
            }
            
            # Look backwards for NOTE_ON (Handle v1)
            if idx > 0:
                prev_info = get_token_info(ids[idx-1], vocab)
                if prev_info['type'] == 'NOTE_ON':
                    note_entry['note_on_idx'] = idx - 1
                    note_entry['note_on_pitch'] = prev_info['pitch']
            
            notes.append(note_entry)
            
    return notes


def infer_tuning_offset(input_notes: List[dict], pred_notes: List[dict]) -> int:
    """
    Infer global tuning offset by comparing Input Pitches vs Standard-Tuning Tab Pitches.
    
    Offset = Standard_Pitch - Actual_Pitch
    
    Example:
      Input (Actual) = 38 (D2)
      Pred Tab = 6th string, 0th fret -> Standard Pitch = 40 (E2)
      Offset = 40 - 38 = 2.
      So Tuning is Standard - 2 = D Standard.
    """
    num_compare = min(len(input_notes), len(pred_notes))
    if num_compare == 0:
        return 0
        
    diffs = []
    
    for i in range(num_compare):
        input_pitch = input_notes[i]['pitch']
        pred = pred_notes[i]
        
        # Calculate what the pitch would be in Standard Tuning
        standard_pitch = get_pitch_from_tab(pred['string'], pred['fret'], tuning_offset=0)
        
        diff = standard_pitch - input_pitch
        diffs.append(diff)
        
    if not diffs:
        return 0
        
    counts = Counter(diffs)
    most_common_diff, count = counts.most_common(1)[0]
    
    # Threshold: If > 30% of notes agree on an offset, assume it's global tuning
    # (30% is generous to handle errors, but strict enough to avoid noise)
    if count > len(diffs) * 0.3:
        return most_common_diff
        
    return 0


# ============================================================================
# Main Post-Processing Logic
# ============================================================================

def post_process_pitch_alignment(
    input_ids: List[int],
    pred_ids: List[int],
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    num_frets: int = 25,
    target_ids: Optional[List[int]] = None,
    original_pred_length: Optional[int] = None,
    output_format: Optional[str] = None
) -> Tuple[List[int], Dict[str, int]]:
    """
    Aligns the pitches of the predicted sequence (TABs) with the input sequence (NOTE_ONs).
    Supports both v1 (NOTE_ON + TAB) and v2 (TAB only).
    Also supports Global Tuning detection (Downtuning/Capo).

    Args:
        input_ids: Ground truth input sequence (NOTE_ON, NOTE_OFF, TIME_SHIFT)
        pred_ids: Predicted output sequence
        input_vocab: Vocabulary for input tokens
        output_vocab: Vocabulary for output tokens
        num_frets: Number of frets (valid frets 0..num_frets-1). Should match config (e.g. 25).
        target_ids: Optional target sequence (unused)
        original_pred_length: Optional original prediction length (unused)
        output_format: Output format ('v1' or 'v2'). If None, auto-detect from pred_ids.

    Returns:
        Tuple of (aligned_ids, statistics_dict)
    """

    # 0. Detect or validate output format
    if output_format is None:
        output_format = detect_output_format(pred_ids, output_vocab)
        print(f"Auto-detected output format: {output_format}")
    else:
        # Validate provided format
        output_format = output_format.lower()
        if output_format not in ['v1', 'v2']:
            print(f"Warning: Invalid output_format '{output_format}', defaulting to 'v1'")
            output_format = 'v1'
        else:
            print(f"Using specified output format: {output_format}")

    # Statistics
    stats = {
        'swaps_made': 0,
        'forced_corrections': 0,
        'internal_inconsistencies_fixed': 0,
        'total_checks': 0,
        'matches_found': 0,
        'tuning_offset': 0,
        'output_format': output_format
    }

    # Work on a copy
    aligned_ids = pred_ids.copy()

    # 1. Initial Parse (with 0 offset) to detect Tuning
    input_notes = get_input_note_sequence(input_ids, input_vocab)
    initial_pred_notes = get_predicted_note_sequence(aligned_ids, output_vocab, tuning_offset=0)
    
    # 2. Infer Tuning Offset
    tuning_offset = infer_tuning_offset(input_notes, initial_pred_notes)
    stats['tuning_offset'] = tuning_offset
    
    if tuning_offset != 0:
        print(f"  Detected Tuning Offset: {tuning_offset} (Standard - Offset = Actual)")
        # Re-parse predictions with correct offset
        pred_notes = get_predicted_note_sequence(aligned_ids, output_vocab, tuning_offset=tuning_offset)
    else:
        pred_notes = initial_pred_notes

    # 3. Determine Scope
    num_to_compare = min(len(input_notes), len(pred_notes))
    stats['total_checks'] = num_to_compare

    print(f"--- Pitch Alignment (Simple Algorithm) ---")
    print(f"Output Format: {output_format.upper()}")
    print(f"Input Notes: {len(input_notes)}, Predicted Tabs: {len(pred_notes)}")
    
    # 4. Alignment Loop
    for i in range(num_to_compare):
        input_note = input_notes[i]
        pred_note = pred_notes[i]
        
        target_pitch = input_note['pitch']
        current_tab_pitch = pred_note['pitch']
        
        # --- CASE A: Pitch Mismatch (External Error) ---
        if target_pitch != current_tab_pitch:
            
            # Logic: Force Correction (Simple & Robust)
            # Find best physical position for target pitch, CONSIDERING TUNING OFFSET
            old_pos = (pred_note['string'], pred_note['fret'])
            new_pos = find_closest_fret(target_pitch, old_pos, num_frets, tuning_offset=tuning_offset)
            
            if new_pos:
                new_string, new_fret = new_pos
                
                # 1. Fix TAB Token
                tab_idx = pred_note['tab_idx']
                aligned_ids[tab_idx] = create_token_id('TAB', output_vocab, string=new_string, fret=new_fret)

                # 2. Fix NOTE_ON Token (only if v1 format)
                if output_format == 'v1' and pred_note['note_on_idx'] is not None:
                    note_on_idx = pred_note['note_on_idx']
                    aligned_ids[note_on_idx] = create_token_id('NOTE_ON', output_vocab, pitch=target_pitch)

                # 3. Fix NOTE_OFF Token (only if v1 format)
                if output_format == 'v1':
                    # Search forward from tab_idx
                    old_pitch_val = current_tab_pitch
                    # However, if v1 was inconsistent (NoteOn=60, Tab=65), the NoteOff is likely 60 (matching NoteOn).
                    # So we should search for NoteOff matching the `note_on_pitch` if it exists, otherwise `tab_pitch`.
                    search_pitch = pred_note['note_on_pitch'] if pred_note['note_on_pitch'] is not None else old_pitch_val

                    found_note_off = False
                    for k in range(tab_idx + 1, len(aligned_ids)):
                        info = get_token_info(aligned_ids[k], output_vocab)
                        if info['type'] == 'NOTE_OFF' and info['pitch'] == search_pitch:
                            # Found it, update it
                            aligned_ids[k] = create_token_id('NOTE_OFF', output_vocab, pitch=target_pitch)
                            found_note_off = True
                            break
                        # Stop if we hit EOS (optional optimization)

                stats['forced_corrections'] += 1
            else:
                pass
                # print(f"Warning: Cannot play pitch {target_pitch} with offset {tuning_offset}. Skipping.")

        # --- CASE B: Pitch Match (Check Internal Consistency for v1) ---
        else:
            stats['matches_found'] += 1

            # Even if TAB matches Input, NOTE_ON might differ from TAB in v1
            # Only check for v1 format
            if output_format == 'v1' and pred_note['note_on_idx'] is not None:
                note_on_pitch = pred_note['note_on_pitch']
                if note_on_pitch != current_tab_pitch:
                    # Inconsistency! model said NoteOn=X, but Tab=Y. (And Y is correct per Input)
                    # We trust TAB (which matches Input), so we fix NoteOn to match TAB.
                    note_on_idx = pred_note['note_on_idx']
                    aligned_ids[note_on_idx] = create_token_id('NOTE_ON', output_vocab, pitch=current_tab_pitch)

                    # Also fix NoteOff if it followed the wrong NoteOn pitch
                    search_pitch = note_on_pitch
                    for k in range(pred_note['tab_idx'] + 1, len(aligned_ids)):
                        info = get_token_info(aligned_ids[k], output_vocab)
                        if info['type'] == 'NOTE_OFF' and info['pitch'] == search_pitch:
                            aligned_ids[k] = create_token_id('NOTE_OFF', output_vocab, pitch=current_tab_pitch)
                            break

                    stats['internal_inconsistencies_fixed'] += 1

    # Print Report
    print(f"\nAlignment Stats:")
    print(f"  Output Format: {output_format.upper()}")
    print(f"  Total Notes Compared: {stats['total_checks']}")
    print(f"  Matches Found (Initial): {stats['matches_found']}")
    print(f"  Forced Corrections (Wrong Pitch): {stats['forced_corrections']}")
    if output_format == 'v1':
        print(f"  Internal Fixes (Tab ok, NoteOn wrong): {stats['internal_inconsistencies_fixed']}")
    if tuning_offset != 0:
        print(f"  Inferred Tuning Offset: {tuning_offset}")

    return aligned_ids, stats
