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

NUM_FRETS = 21  # Fret range 0-20
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


def pitch_to_frets(pitch: int, tuning_offset: int = 0) -> List[Tuple[int, int]]:
    """
    Finds all possible (string, fret) combinations for a given MIDI pitch,
    considering the global tuning offset.
    """
    possible_frets = []
    
    # Iterate through strings (Index 0=Low E=String 1, Index 5=High E=String 6)
    for tuning_idx, open_pitch_standard in enumerate(GUITAR_TUNING):
        # Calculate string number from tuning index (Index 0 -> String 1)
        string_num = tuning_idx + 1
        
        # Calculate the actual open pitch of this string
        open_pitch_actual = open_pitch_standard - tuning_offset
        
        fret = pitch - open_pitch_actual
        if 0 <= fret < NUM_FRETS:
            possible_frets.append((string_num, fret))
            
    return possible_frets


def find_closest_fret(pitch: int, original_pos: Tuple[int, int], tuning_offset: int = 0) -> Optional[Tuple[int, int]]:
    """
    Finds the best (string, fret) for a pitch, closest to an original position.
    """
    candidates = pitch_to_frets(pitch, tuning_offset)
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


def get_predicted_tab_sequence(ids: List[int], vocab: Vocabulary, tuning_offset: int = 0) -> List[dict]:
    """
    Extract TAB token sequence from Predicted IDs.

    Each TAB entry includes:
    - Position in sequence (tab_idx)
    - Physical position (string, fret)
    - Calculated pitch (considering tuning offset)
    - Time (accumulated from TIME_SHIFT tokens)
    - Associated NOTE_ON info (if v1 format)
    - Matching status (for algorithm use)
    """
    tabs = []
    current_time = 0

    for idx, token_id in enumerate(ids):
        info = get_token_info(token_id, vocab)

        if info['type'] == 'TIME_SHIFT':
            current_time += info['delta']

        elif info['type'] == 'TAB':
            # Found a TAB token
            string = info['string']
            fret = info['fret']
            pitch = get_pitch_from_tab(string, fret, tuning_offset)

            tab_entry = {
                'tab_idx': idx,
                'string': string,
                'fret': fret,
                'pitch': pitch,
                'time': current_time,
                'note_on_idx': None,
                'note_on_pitch': None,
                'matched': False,  # For matching algorithm
                'assigned_input_idx': None  # Which input note this tab is matched to
            }

            # Look backwards for NOTE_ON (v1 format)
            if idx > 0:
                prev_info = get_token_info(ids[idx-1], vocab)
                if prev_info['type'] == 'NOTE_ON':
                    tab_entry['note_on_idx'] = idx - 1
                    tab_entry['note_on_pitch'] = prev_info['pitch']

            tabs.append(tab_entry)

    return tabs


def infer_tuning_offset(input_notes: List[dict], pred_tabs: List[dict]) -> int:
    """
    Infer global tuning offset by comparing Input Pitches vs Standard-Tuning Tab Pitches.

    Offset = Standard_Pitch - Actual_Pitch

    Example:
      Input (Actual) = 38 (D2)
      Pred Tab = 6th string, 0th fret -> Standard Pitch = 40 (E2)
      Offset = 40 - 38 = 2.
      So Tuning is Standard - 2 = D Standard.
    """
    num_compare = min(len(input_notes), len(pred_tabs))
    if num_compare == 0:
        return 0

    diffs = []

    for i in range(num_compare):
        input_pitch = input_notes[i]['pitch']
        pred = pred_tabs[i]

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
# Matching Strategy Functions (Timeline-based)
# ============================================================================

def find_exact_match(
    target_pitch: int,
    target_time: int,
    pred_tabs: List[dict],
    pitch_threshold: int = 1
) -> Optional[int]:
    """
    Strategy 1: Find a tab at exact time with matching pitch.

    Args:
        target_pitch: The pitch we're looking for
        target_time: The exact time we're looking at
        pred_tabs: List of predicted tab entries
        pitch_threshold: Maximum allowed pitch difference

    Returns:
        Index in pred_tabs of best match, or None
    """
    candidates = []

    for i, tab in enumerate(pred_tabs):
        if tab['matched']:  # Already used
            continue

        if tab['time'] == target_time:
            pitch_diff = abs(tab['pitch'] - target_pitch)
            if pitch_diff <= pitch_threshold:
                candidates.append((i, pitch_diff))

    if not candidates:
        return None

    # Choose the one with smallest pitch difference
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def find_time_window_match(
    target_pitch: int,
    target_time: int,
    pred_tabs: List[dict],
    time_threshold: int = 240,
    pitch_threshold: int = 2
) -> Optional[int]:
    """
    Strategy 2: Find best tab within a time window.

    Score = pitch_diff + (time_diff / time_threshold) * weight
    Lower score is better.

    Args:
        target_pitch: The pitch we're looking for
        target_time: The target time
        pred_tabs: List of predicted tab entries
        time_threshold: Maximum time window (ticks)
        pitch_threshold: Maximum allowed pitch difference

    Returns:
        Index in pred_tabs of best match, or None
    """
    candidates = []

    for i, tab in enumerate(pred_tabs):
        if tab['matched']:
            continue

        time_diff = abs(tab['time'] - target_time)
        if time_diff <= time_threshold:
            pitch_diff = abs(tab['pitch'] - target_pitch)
            if pitch_diff <= pitch_threshold:
                # Combined score: pitch matters more than time
                # Weight time_diff to be comparable to pitch_diff
                score = pitch_diff + (time_diff / time_threshold) * 2
                candidates.append((i, score, pitch_diff, time_diff))

    if not candidates:
        return None

    # Choose the one with lowest combined score
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def generate_optimal_tab(
    target_pitch: int,
    last_tab_position: Tuple[int, int],
    tuning_offset: int
) -> Optional[Tuple[int, int]]:
    """
    Strategy 3: Generate optimal tab position when no prediction tab is available.

    Finds the (string, fret) combination that:
    1. Can produce the target pitch
    2. Is closest to the last tab position (minimize hand movement)

    Args:
        target_pitch: The pitch we need to play
        last_tab_position: Previous (string, fret) for distance calculation
        tuning_offset: Global tuning offset

    Returns:
        Optimal (string, fret), or None if pitch is unplayable
    """
    candidates = pitch_to_frets(target_pitch, tuning_offset)
    if not candidates:
        return None

    min_dist = float('inf')
    best = candidates[0]

    for s, f in candidates:
        # Manhattan distance on fretboard
        dist = abs(s - last_tab_position[0]) + abs(f - last_tab_position[1])
        if dist < min_dist:
            min_dist = dist
            best = (s, f)

    return best


def match_input_to_prediction(
    input_notes: List[dict],
    pred_tabs: List[dict],
    tuning_offset: int,
    pitch_threshold: int = 1,
    time_threshold: int = 240
) -> List[dict]:
    """
    Match each input NOTE_ON to a corresponding predicted TAB.

    Uses a three-tier strategy:
    1. Exact time and pitch match
    2. Time window match (within threshold)
    3. Position-optimized fallback (when no pred tab is suitable)

    Args:
        input_notes: Ground truth notes from input sequence
        pred_tabs: Predicted tab tokens
        tuning_offset: Global tuning offset
        pitch_threshold: Pitch tolerance for exact match
        time_threshold: Time window size (ticks)

    Returns:
        List of match results, one per input note
    """
    matches = []
    last_tab_pos = (3, 0)  # Initial reference position (middle string, open)

    for i, input_note in enumerate(input_notes):
        target_pitch = input_note['pitch']
        target_time = input_note['time']

        match_result = {
            'input_idx': i,
            'input_pitch': target_pitch,
            'input_time': target_time,
            'strategy': None,
            'pred_tab_idx': -1,
            'correction_needed': False,
            'original_tab': None,
            'corrected_tab': None
        }

        # Strategy 1: Exact time and pitch match
        pred_idx = find_exact_match(target_pitch, target_time, pred_tabs, pitch_threshold)
        if pred_idx is not None:
            tab = pred_tabs[pred_idx]
            tab['matched'] = True
            tab['assigned_input_idx'] = i

            match_result['strategy'] = 'exact_time_pitch'
            match_result['pred_tab_idx'] = pred_idx
            match_result['original_tab'] = (tab['string'], tab['fret'])

            # Check if correction is needed (pitch slightly different)
            if tab['pitch'] != target_pitch:
                new_tab = find_closest_fret(target_pitch, (tab['string'], tab['fret']), tuning_offset)
                match_result['correction_needed'] = True
                match_result['corrected_tab'] = new_tab
                last_tab_pos = new_tab
            else:
                match_result['corrected_tab'] = (tab['string'], tab['fret'])
                last_tab_pos = (tab['string'], tab['fret'])

            matches.append(match_result)
            continue

        # Strategy 2: Time window match
        pred_idx = find_time_window_match(target_pitch, target_time, pred_tabs, time_threshold)
        if pred_idx is not None:
            tab = pred_tabs[pred_idx]
            tab['matched'] = True
            tab['assigned_input_idx'] = i

            match_result['strategy'] = 'time_window'
            match_result['pred_tab_idx'] = pred_idx
            match_result['original_tab'] = (tab['string'], tab['fret'])

            # Correction needed: fix to correct pitch
            new_tab = find_closest_fret(target_pitch, (tab['string'], tab['fret']), tuning_offset)
            match_result['correction_needed'] = True
            match_result['corrected_tab'] = new_tab
            last_tab_pos = new_tab

            matches.append(match_result)
            continue

        # Strategy 3: Position-optimized fallback
        new_tab = generate_optimal_tab(target_pitch, last_tab_pos, tuning_offset)
        if new_tab:
            match_result['strategy'] = 'position_fallback'
            match_result['pred_tab_idx'] = -1  # No pred tab available
            match_result['correction_needed'] = True
            match_result['corrected_tab'] = new_tab
            last_tab_pos = new_tab
        else:
            match_result['strategy'] = 'none'  # Unplayable

        matches.append(match_result)

    return matches


# ============================================================================
# Token Correction Logic
# ============================================================================

def find_nearest_unused_tab(
    target_time: int,
    pred_tabs: List[dict]
) -> Optional[int]:
    """
    Find the nearest unused (unmatched) tab to a target time.

    Args:
        target_time: Target time to search around
        pred_tabs: List of predicted tab entries

    Returns:
        Index of nearest unused tab, or None if all tabs are matched
    """
    unused_tabs = [(i, tab) for i, tab in enumerate(pred_tabs) if not tab['matched']]

    if not unused_tabs:
        return None

    # Find the one with minimum time distance
    min_dist = float('inf')
    best_idx = None

    for idx, tab in unused_tabs:
        time_dist = abs(tab['time'] - target_time)
        if time_dist < min_dist:
            min_dist = time_dist
            best_idx = idx

    return best_idx


def apply_tab_corrections(
    pred_ids: List[int],
    matches: List[dict],
    pred_tabs: List[dict],
    output_vocab: Vocabulary
) -> List[int]:
    """
    Apply corrections to prediction sequence based on matches.

    Modifies:
    - TAB tokens (to correct string/fret)
    - NOTE_ON tokens (if v1 format, to correct pitch)
    - NOTE_OFF tokens (if v1 format, to correct pitch)

    NEW: Also handles 'position_fallback' cases by repurposing unused tabs.

    Args:
        pred_ids: Original prediction token IDs
        matches: Match results from match_input_to_prediction
        pred_tabs: Predicted tab sequence info
        output_vocab: Vocabulary for token conversion

    Returns:
        Corrected prediction token IDs
    """
    corrected_ids = pred_ids.copy()

    # Phase 1: Handle exact matches and time window matches
    for match in matches:
        if match['strategy'] not in ['exact_time_pitch', 'time_window']:
            continue

        if not match['correction_needed']:
            continue

        pred_tab_idx = match['pred_tab_idx']
        tab_info = pred_tabs[pred_tab_idx]

        # 1. Correct TAB token
        new_string, new_fret = match['corrected_tab']
        tab_token_id = create_token_id('TAB', output_vocab, string=new_string, fret=new_fret)
        corrected_ids[tab_info['tab_idx']] = tab_token_id

        # 2. Correct NOTE_ON (if v1 format)
        if tab_info['note_on_idx'] is not None:
            note_on_token_id = create_token_id('NOTE_ON', output_vocab, pitch=match['input_pitch'])
            corrected_ids[tab_info['note_on_idx']] = note_on_token_id

            # 3. Correct NOTE_OFF (search forward from tab)
            old_pitch = tab_info['note_on_pitch']  # Original NOTE_ON pitch
            for k in range(tab_info['tab_idx'] + 1, len(corrected_ids)):
                info = get_token_info(corrected_ids[k], output_vocab)
                if info['type'] == 'NOTE_OFF' and info['pitch'] == old_pitch:
                    note_off_token_id = create_token_id('NOTE_OFF', output_vocab, pitch=match['input_pitch'])
                    corrected_ids[k] = note_off_token_id
                    break

    # Phase 2: Handle position_fallback by repurposing unused tabs
    for match in matches:
        if match['strategy'] != 'position_fallback':
            continue

        # Find nearest unused tab to repurpose
        target_time = match['input_time']
        nearest_unused_idx = find_nearest_unused_tab(target_time, pred_tabs)

        if nearest_unused_idx is None:
            # No unused tabs available, skip this correction
            continue

        # Mark this tab as used (so we don't reuse it)
        pred_tabs[nearest_unused_idx]['matched'] = True
        tab_info = pred_tabs[nearest_unused_idx]

        # 1. Replace TAB token with correct position
        new_string, new_fret = match['corrected_tab']
        tab_token_id = create_token_id('TAB', output_vocab, string=new_string, fret=new_fret)
        corrected_ids[tab_info['tab_idx']] = tab_token_id

        # 2. Replace NOTE_ON (if v1 format)
        if tab_info['note_on_idx'] is not None:
            note_on_token_id = create_token_id('NOTE_ON', output_vocab, pitch=match['input_pitch'])
            corrected_ids[tab_info['note_on_idx']] = note_on_token_id

            # 3. Replace NOTE_OFF (search forward from tab)
            old_pitch = tab_info['note_on_pitch']  # Original NOTE_ON pitch
            for k in range(tab_info['tab_idx'] + 1, len(corrected_ids)):
                info = get_token_info(corrected_ids[k], output_vocab)
                if info['type'] == 'NOTE_OFF' and info['pitch'] == old_pitch:
                    note_off_token_id = create_token_id('NOTE_OFF', output_vocab, pitch=match['input_pitch'])
                    corrected_ids[k] = note_off_token_id
                    break

    return corrected_ids


def build_output_from_input(
    input_ids: List[int],
    matches: List[dict],
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    output_format: str = 'v1'
) -> List[int]:
    """
    Build output sequence based on INPUT sequence, inserting matched TABs.

    Strategy (v1):
    - Start with BOS token (if not in input)
    - For each NOTE_ON: output NOTE_ON + corresponding TAB
    - For TIME_SHIFT: output TIME_SHIFT
    - For NOTE_OFF: output NOTE_OFF

    Strategy (v2):
    - Start with BOS token
    - For each NOTE_ON: output corresponding TAB only (no NOTE_ON/NOTE_OFF)
    - For TIME_SHIFT: output TIME_SHIFT

    This ensures 100% of input notes have corresponding TABs.

    Args:
        input_ids: Ground truth input sequence
        matches: Match results from match_input_to_prediction
        input_vocab: Input vocabulary
        output_vocab: Output vocabulary
        output_format: 'v1' or 'v2'

    Returns:
        Output token IDs based on input structure
    """
    output_ids = []
    current_time = 0
    note_on_count = 0  # Track which NOTE_ON we're at

    # Check if input starts with BOS, if not, add it to output
    if len(input_ids) == 0 or input_ids[0] != input_vocab.bos_id:
        output_ids.append(output_vocab.bos_id)

    for token_id in input_ids:
        info = get_token_info(token_id, input_vocab)

        if info['type'] == 'NOTE_ON':
            pitch = info['pitch']

            if output_format == 'v1':
                # v1: Output NOTE_ON + TAB
                note_on_token = create_token_id('NOTE_ON', output_vocab, pitch=pitch)
                output_ids.append(note_on_token)

            # Output corresponding TAB (for both v1 and v2)
            if note_on_count < len(matches):
                match = matches[note_on_count]

                # Use corrected_tab if available
                if match['corrected_tab'] is not None:
                    string, fret = match['corrected_tab']
                    tab_token = create_token_id('TAB', output_vocab, string=string, fret=fret)
                    output_ids.append(tab_token)
                else:
                    # This should rarely happen (strategy='none', unplayable pitch)
                    # Output a default tab (middle string, fret 0)
                    tab_token = create_token_id('TAB', output_vocab, string=3, fret=0)
                    output_ids.append(tab_token)
            else:
                # Safety: if we somehow have more NOTE_ONs than matches
                # Output default tab
                tab_token = create_token_id('TAB', output_vocab, string=3, fret=0)
                output_ids.append(tab_token)

            note_on_count += 1

        elif info['type'] == 'TIME_SHIFT':
            delta = info['delta']
            current_time += delta
            time_shift_token = create_token_id('TIME_SHIFT', output_vocab, delta=delta)
            output_ids.append(time_shift_token)

        elif info['type'] == 'NOTE_OFF':
            if output_format == 'v1':
                # v1: Output NOTE_OFF
                pitch = info['pitch']
                note_off_token = create_token_id('NOTE_OFF', output_vocab, pitch=pitch)
                output_ids.append(note_off_token)
            # v2: Skip NOTE_OFF

        elif info['type'] == 'SPECIAL':
            # Handle BOS, EOS, PAD, UNK, etc.
            if token_id == input_vocab.bos_id:
                output_ids.append(output_vocab.bos_id)
            elif token_id == input_vocab.eos_id:
                output_ids.append(output_vocab.eos_id)
            elif token_id == input_vocab.pad_id:
                output_ids.append(output_vocab.pad_id)
            elif token_id == input_vocab.unk_id:
                output_ids.append(output_vocab.unk_id)
            # Otherwise skip truly unknown tokens

    return output_ids


# ============================================================================
# Statistics and Reporting
# ============================================================================

def compute_statistics(matches: List[dict], pred_tabs: List[dict], tuning_offset: int) -> Dict[str, int]:
    """
    Compute detailed statistics from matching results.
    """
    stats = {
        'total_input_notes': len(matches),
        'total_pred_tabs': len(pred_tabs),

        'exact_matches': 0,
        'exact_matches_no_correction': 0,
        'exact_matches_with_correction': 0,

        'time_window_matches': 0,
        'position_fallbacks': 0,
        'unmatched': 0,

        'unused_pred_tabs': 0,

        'tuning_offset': tuning_offset,

        # Backward compatibility fields
        'swaps_made': 0,
        'forced_corrections': 0
    }

    # Count match strategies
    for match in matches:
        strategy = match['strategy']

        if strategy == 'exact_time_pitch':
            stats['exact_matches'] += 1
            if match['correction_needed']:
                stats['exact_matches_with_correction'] += 1
                stats['forced_corrections'] += 1  # Backward compat
            else:
                stats['exact_matches_no_correction'] += 1

        elif strategy == 'time_window':
            stats['time_window_matches'] += 1
            stats['forced_corrections'] += 1  # Always needs correction
            stats['swaps_made'] += 1  # Time-shifted match

        elif strategy == 'position_fallback':
            stats['position_fallbacks'] += 1
            stats['forced_corrections'] += 1  # Backward compat

        elif strategy == 'none':
            stats['unmatched'] += 1

    # Count unused prediction tabs
    stats['unused_pred_tabs'] = sum(1 for tab in pred_tabs if not tab['matched'])

    return stats


def print_report(stats: Dict[str, int]) -> None:
    """
    Print detailed matching report.
    """
    print(f"\n--- Pitch Alignment Report (Input-based Output) ---")
    print(f"Input Notes (Ground Truth): {stats['total_input_notes']}")
    print(f"Predicted Tabs Available: {stats['total_pred_tabs']}")

    print(f"\nMatching Results:")
    print(f"  Strategy 1 (Exact Time+Pitch):")
    print(f"    - Perfect matches: {stats['exact_matches_no_correction']}")
    print(f"    - With pitch correction: {stats['exact_matches_with_correction']}")
    print(f"    - Subtotal: {stats['exact_matches']}")

    print(f"  Strategy 2 (Time Window): {stats['time_window_matches']}")
    print(f"  Strategy 3 (Position Fallback): {stats['position_fallbacks']}")
    print(f"  Unmatched (Unplayable): {stats['unmatched']}")

    print(f"\nOutput Coverage:")
    total_matched = stats['exact_matches'] + stats['time_window_matches'] + stats['position_fallbacks']
    print(f"  Total input notes processed: {stats['total_input_notes']}")
    print(f"  Notes with TABs assigned: {total_matched} (100.00%)")

    print(f"\nPrediction Quality:")
    print(f"  Model-generated tabs used: {stats['exact_matches'] + stats['time_window_matches']}")
    print(f"  Optimized tabs generated: {stats['position_fallbacks']}")
    print(f"  Unused pred tabs: {stats['unused_pred_tabs']}")

    if stats['tuning_offset'] != 0:
        print(f"\nTuning Offset Detected: {stats['tuning_offset']} semitones")


# ============================================================================
# Main Post-Processing Logic
# ============================================================================

def post_process_pitch_alignment(
    input_ids: List[int],
    pred_ids: List[int],
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    target_ids: Optional[List[int]] = None,
    original_pred_length: Optional[int] = None,
    pitch_threshold: int = 1,
    time_threshold: int = 240,
    output_format: str = 'v1'
) -> Tuple[List[int], Dict[str, int]]:
    """
    Aligns predicted TABs with input NOTE_ONs using timeline-based matching.

    NEW ALGORITHM (Timeline-based):
    - For each input NOTE_ON, find the best matching predicted TAB based on time and pitch
    - Uses three-tier matching strategy:
        1. Exact time + pitch match
        2. Time window match (±time_threshold)
        3. Position-optimized fallback (when no suitable pred tab exists)

    Supports both v1 (NOTE_ON + TAB) and v2 (TAB only) token formats.
    Automatically detects global tuning offset (downtuning/capo).

    Args:
        input_ids: Ground truth input sequence (NOTE_ON, NOTE_OFF, TIME_SHIFT)
        pred_ids: Predicted output sequence (TAB, possibly NOTE_ON/OFF, TIME_SHIFT)
        input_vocab: Vocabulary for input tokens
        output_vocab: Vocabulary for output tokens
        target_ids: Optional target sequence (unused in current implementation)
        original_pred_length: Optional original prediction length (unused)
        pitch_threshold: Pitch tolerance for exact matching (semitones)
        time_threshold: Time window for matching (ticks, default 240 = half beat @ 480 tpq)
        output_format: Output format ('v1' for NOTE_ON+TAB+NOTE_OFF, 'v2' for TAB only). Default: 'v1'

    Returns:
        Tuple of (corrected_ids, statistics_dict)
    """

    # 2. Parse input sequence (ground truth)
    input_notes = get_input_note_sequence(input_ids, input_vocab)

    # 3. Initial parse of prediction (with tuning_offset=0)
    initial_pred_tabs = get_predicted_tab_sequence(pred_ids, output_vocab, tuning_offset=0)

    # 4. Infer tuning offset
    tuning_offset = infer_tuning_offset(input_notes, initial_pred_tabs)

    # 5. Re-parse with correct tuning if needed
    if tuning_offset != 0:
        pred_tabs = get_predicted_tab_sequence(pred_ids, output_vocab, tuning_offset=tuning_offset)
    else:
        pred_tabs = initial_pred_tabs

    # 6. Match input notes to predicted tabs
    matches = match_input_to_prediction(
        input_notes,
        pred_tabs,
        tuning_offset,
        pitch_threshold,
        time_threshold
    )

    # 7. Build output sequence based on INPUT (not PREDICTION)
    # This ensures every input NOTE_ON has a corresponding TAB
    # Output format is specified by the output_format parameter (v1 or v2)
    output_ids = build_output_from_input(
        input_ids,
        matches,
        input_vocab,
        output_vocab,
        output_format=output_format
    )

    # 7. Compute statistics
    stats = compute_statistics(matches, pred_tabs, tuning_offset)

    # 8. Print report
    print_report(stats)

    return output_ids, stats
