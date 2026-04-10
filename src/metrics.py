"""
Evaluation metrics for guitar tablature transcription.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import torch
from tqdm import tqdm
from collections import Counter, defaultdict

# Standard tuning: E2, A2, D3, G3, B3, E4
GUITAR_TUNING = [40, 45, 50, 55, 59, 64]

@dataclass
class TabAccuracyMetrics:
    """Container for tablature accuracy metrics."""

    # Token-level accuracy
    token_accuracy: float  # % of all tokens correct

    # Note-level accuracy
    # v1: based on NOTE_ON positions
    # v2: based on TAB positions
    pitch_accuracy: float  # % of notes with correct pitch
    tab_accuracy: float  # % of notes with correct (string, fret)

    # playability score (difficulty)
    difficulty: float

    # Counts
    total_tokens: int
    total_notes: int

    def __str__(self) -> str:
        return (
            f"Token Acc: {self.token_accuracy:.2%} | "
            f"Pitch Acc: {self.pitch_accuracy:.2%} | "
            f"Tab Acc: {self.tab_accuracy:.2%} | "
            # f"Lev Sim: {self.levenshtein_similarity:.2%} | "
            f"({self.total_notes} notes, {self.total_tokens} tokens)"
        )

def compute_levenshtein_distance(s1, s2):
    """
    Compute Levenshtein distance between two sequences.
    """
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def compute_pitch_from_tab_token(token_str: str, tuning: Optional[List[int]] = None) -> int:
    """
    Compute MIDI pitch from a TAB token string (e.g., 'TAB_3_5').
    
    Args:
        token_str: TAB token string (e.g., 'TAB_3_5')
        tuning: Optional tuning list [open_pitch_string1, ..., open_pitch_string6].
                If None, uses standard tuning.
    
    Returns:
        MIDI pitch, or -1 if invalid.
    """
    if not token_str.startswith("TAB_"):
        return -1
    try:
        _, s, f = token_str.split("_")
        string_idx = int(s)
        fret = int(f)
        # string is 1-indexed (1-6)
        if tuning is None:
            tuning = GUITAR_TUNING
        if 1 <= string_idx <= len(tuning):
            return tuning[string_idx - 1] + fret
    except ValueError:
        pass
    return -1


def infer_tuning_from_input_output(
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    input_vocab,
    output_vocab,
) -> Optional[Dict[int, int]]:
    """
    Infer tuning from NOTE_ON and TAB events.

    For v1 format: Use NOTE_ON and TAB from target (they are paired sequentially)
    For v2 format: Use NOTE_ON from input and TAB from target (paired by order)

    Uses the same method as analyze_tuning-testset.py:
    1. Collect all (pitch, string, fret) tuples
    2. For each string, compute derived_open = pitch - fret
    3. Use most common derived_open for each string
    4. Calculate global offset from standard tuning
    5. Return tuning with global offset applied to all strings

    Args:
        input_ids: [L_input] - Input token IDs
        target_ids: [L_output] - Target token IDs
        input_vocab: Input vocabulary
        output_vocab: Output vocabulary

    Returns:
        Dict mapping string (1-6) to open string pitch, or None if inference fails
    """
    # Decode target tokens
    target_tokens = [
        output_vocab.id_to_token[idx.item()]
        for idx in target_ids
        if idx != output_vocab.pad_id and idx != output_vocab.eos_id
    ]

    # Check format: v1 has NOTE_ON in target, v2 doesn't
    has_note_on = any(t.startswith("NOTE_ON_") for t in target_tokens)

    note_tab_pairs = []

    if has_note_on:
        # v1 format: Extract NOTE_ON + TAB pairs from target
        # In v1, NOTE_ON is immediately followed by TAB
        current_pitch = None
        for token in target_tokens:
            if token.startswith("NOTE_ON_"):
                try:
                    current_pitch = int(token.split("_")[2])
                except (ValueError, IndexError):
                    current_pitch = None
            elif token.startswith("TAB_"):
                if current_pitch is not None:
                    try:
                        parts = token.split("_")
                        string = int(parts[1])
                        fret = int(parts[2])
                        if 1 <= string <= 6:
                            note_tab_pairs.append({
                                'pitch': current_pitch,
                                'string': string,
                                'fret': fret
                            })
                    except (ValueError, IndexError):
                        pass
                current_pitch = None  # Reset after TAB
    else:
        # v2 format: Pair NOTE_ON from input with TAB from target
        # Extract NOTE_ON pitches from input
        input_tokens = [
            input_vocab.id_to_token[idx.item()]
            for idx in input_ids
            if idx != input_vocab.pad_id and idx != input_vocab.eos_id
        ]

        note_pitches = []
        for token in input_tokens:
            if token.startswith("NOTE_ON_"):
                try:
                    pitch = int(token.split("_")[2])
                    note_pitches.append(pitch)
                except (ValueError, IndexError):
                    pass

        # Extract TAB events from target
        tabs = []
        for token in target_tokens:
            if token.startswith("TAB_"):
                try:
                    parts = token.split("_")
                    string = int(parts[1])
                    fret = int(parts[2])
                    if 1 <= string <= 6:
                        tabs.append((string, fret))
                except (ValueError, IndexError):
                    pass

        # Pair them in order
        min_len = min(len(note_pitches), len(tabs))
        for i in range(min_len):
            note_tab_pairs.append({
                'pitch': note_pitches[i],
                'string': tabs[i][0],
                'fret': tabs[i][1]
            })

    if len(note_tab_pairs) == 0:
        return None

    # Collect evidence for each string: derived_open = pitch - fret
    string_tunings = defaultdict(list)
    for pair in note_tab_pairs:
        derived_open = pair['pitch'] - pair['fret']
        string_tunings[pair['string']].append(derived_open)

    # Get most common open pitch for each string
    per_string_tuning = {}
    for s in range(1, 7):
        if s in string_tunings and len(string_tunings[s]) > 0:
            common_open = Counter(string_tunings[s]).most_common(1)[0][0]
            per_string_tuning[s] = common_open

    if not per_string_tuning:
        return None

    # Calculate offsets from standard tuning for each string
    offsets = []
    for s, derived_open in per_string_tuning.items():
        standard_open = GUITAR_TUNING[s - 1]
        offset = derived_open - standard_open
        offsets.append(offset)

    # Use most common offset as global tuning offset
    global_offset = Counter(offsets).most_common(1)[0][0]

    # Build estimated tuning with global offset
    estimated_tuning = {}
    for s in range(1, 7):
        estimated_tuning[s] = GUITAR_TUNING[s - 1] + global_offset

    return estimated_tuning

def compute_tablature_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_vocab,
    input_ids: Optional[torch.Tensor] = None,
    input_vocab = None,
    # pad_id: int = -100 # 0
) -> TabAccuracyMetrics:
    """
    Compute accuracy metrics for tablature transcription.
    Supports both v1 (NOTE_ON, NOTE_OFF, TAB...) and v2 (TAB, TIME_SHIFT...) formats.

    Args:
        predictions: [B, L] - Predicted token IDs
        targets: [B, L] - Ground truth token IDs
        output_vocab: Vocabulary object with id_to_token mapping
        input_ids: Optional [B, L_input] - Input token IDs (needed for tuning inference in v2)
        input_vocab: Optional input vocabulary (needed for tuning inference in v2)
        pad_id: Padding token ID to ignore

    Returns:
        TabAccuracyMetrics with computed accuracies
    """
    # For v2, try to infer tuning from input_ids if available (before flattening)
    inferred_tuning = None
    if input_ids is not None and input_vocab is not None:
        # Check if v2 format by looking at first target token
        if len(targets.shape) == 2 and targets.shape[0] > 0:
            # Sample a few targets to detect format
            sample_targets = targets[0]
            sample_target_tokens = [
                output_vocab.id_to_token[idx.item()] 
                for idx in sample_targets 
                if idx != output_vocab.pad_id and idx != output_vocab.eos_id
            ]
            has_note_on = any(t.startswith("NOTE_ON_") for t in sample_target_tokens)
            
            if not has_note_on:
                # v2 format: try to infer tuning from all samples
                # We'll infer tuning per sample and use the most common one, or use standard if inference fails
                B = predictions.shape[0]
                tunings = []
                for i in range(B):
                    sample_tuning = infer_tuning_from_input_output(
                        input_ids[i], targets[i], input_vocab, output_vocab
                    )
                    if sample_tuning is not None:
                        tunings.append(sample_tuning)
                
                # Use the most common tuning, or standard if none found
                if tunings:
                    # For simplicity, use the first inferred tuning
                    # In practice, all samples from the same song should have the same tuning
                    inferred_tuning = tunings[0]
    
    # Flatten tensors
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)

    # Mask out padding
    # print("pad_id", output_vocab.pad_id)
    non_pad_mask = target_flat != output_vocab.pad_id
    pred_flat = pred_flat[non_pad_mask]
    target_flat = target_flat[non_pad_mask]

    total_tokens = len(target_flat)

    # Token-level accuracy (all tokens)
    correct_tokens = (pred_flat == target_flat).sum().item()
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    # Decode tokens to strings
    pred_tokens = [output_vocab.id_to_token[idx.item()] for idx in pred_flat]
    target_tokens = [output_vocab.id_to_token[idx.item()] for idx in target_flat]

    # playability score
    tab_positions_for_difficulty = extract_tab_positions(pred_tokens)
    difficulty = difficulty_score(tab_positions_for_difficulty)

    # Detect format based on target tokens
    has_note_on = any(t.startswith("NOTE_ON_") for t in target_tokens)
    
    if has_note_on:
        # --- v1 Logic (NOTE_ON based) ---
        note_positions = [
            i for i, token in enumerate(target_tokens) if token.startswith("NOTE_ON_")
        ]

        if len(note_positions) == 0:
            return TabAccuracyMetrics(
                token_accuracy=token_accuracy,
                pitch_accuracy=0.0,
                tab_accuracy=0.0,
                difficulty=difficulty,
                total_tokens=total_tokens,
                total_notes=0,
                # levenshtein_distance=0.0,
                # levenshtein_similarity=0.0
            )

        # Pitch accuracy: check if pitch matches
        pitch_correct = 0
        for pos in note_positions:
            pred_token = pred_tokens[pos]
            target_token = target_tokens[pos]
            if pred_token == target_token:
                pitch_correct += 1

        pitch_accuracy = pitch_correct / len(note_positions)

        # Tab accuracy: check if following TAB token matches
        # NOTE_ON is always followed by TAB in output sequence
        tab_correct = 0
        tab_eval_positions = []

        for pos in note_positions:
            # Check if next token is TAB
            if pos + 1 < len(target_tokens):
                target_next = target_tokens[pos + 1]
                if target_next.startswith("TAB_"):
                    tab_eval_positions.append(pos + 1)
                    pred_tab = pred_tokens[pos + 1]
                    if pred_tab == target_next:
                        tab_correct += 1

        tab_accuracy = tab_correct / len(tab_eval_positions) if tab_eval_positions else 0.0
        total_notes = len(note_positions)

    else:
        # --- v2 Logic (TAB based) ---
        # No NOTE_ON tokens, so we evaluate based on TAB tokens
        tab_indices = [
            i for i, token in enumerate(target_tokens) if token.startswith("TAB_")
        ]

        if len(tab_indices) == 0:
            # Maybe it's an empty sequence or only silences?
            return TabAccuracyMetrics(
                token_accuracy=token_accuracy,
                pitch_accuracy=0.0,
                tab_accuracy=0.0,
                difficulty=difficulty,
                total_tokens=total_tokens,
                total_notes=0,
                # levenshtein_distance=0.0,
                # levenshtein_similarity=0.0
            )
        
        # Tab Accuracy: Exact token match at TAB positions
        tab_correct = 0
        for pos in tab_indices:
            if pred_tokens[pos] == target_tokens[pos]:
                tab_correct += 1
        
        tab_accuracy = tab_correct / len(tab_indices)

        # Pitch Accuracy: Computed pitch match at TAB positions
        # Use inferred tuning if available, otherwise use standard tuning
        tuning_list = None
        if inferred_tuning is not None:
            # Convert dict to list [open_pitch_string1, ..., open_pitch_string6]
            tuning_list = [inferred_tuning.get(s, GUITAR_TUNING[s-1]) for s in range(1, 7)]
        
        pitch_correct = 0
        for pos in tab_indices:
            target_token = target_tokens[pos]
            pred_token = pred_tokens[pos]
            
            target_pitch = compute_pitch_from_tab_token(target_token, tuning=tuning_list)
            
            # For prediction, we need to check if it's a TAB token first
            pred_pitch = compute_pitch_from_tab_token(pred_token, tuning=tuning_list)
            
            if target_pitch != -1 and pred_pitch != -1:
                if target_pitch == pred_pitch:
                    pitch_correct += 1
            # If pred is not a TAB or invalid TAB, it's a mismatch (unless target is also invalid which shouldn't happen)
            
        pitch_accuracy = pitch_correct / len(tab_indices)
        total_notes = len(tab_indices)


    return TabAccuracyMetrics(
        token_accuracy=token_accuracy,
        pitch_accuracy=pitch_accuracy,
        tab_accuracy=tab_accuracy,
        difficulty=difficulty,
        total_tokens=total_tokens,
        total_notes=total_notes,
    )

def generate_and_compute_accuracy(
    model,
    dataloader,
    output_vocab,
    device: str,
    max_length: int = 1024,
    num_beams: int = 1,  # DEPRECATED (kept for config compat)
    max_batches: Optional[int] = None,
    use_teacher_forcing: bool = True,  # NEW
    temperature: float = 1.0,  # NEW
    input_vocab = None,  # NEW: needed for tuning inference in v2
    use_constrained_decoding: bool = False,
    constrained_decoding_mode: str = "input_skeleton",
    num_frets: int = 25,
    disable_kv_cache: bool = False,
) -> tuple[TabAccuracyMetrics, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate sequences autoregressively and compute accuracy.

    Args:
        model: FrettingTransformer model
        dataloader: DataLoader for validation/test set
        output_vocab: Output vocabulary
        device: Device to run on
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
        max_batches: Limit number of batches (for speed)
        return_predictions: If True, return (metrics, (predictions, targets))

    Returns:
        (metrics, (input_ids, targets, predictions))
    """
    model.eval()

    all_predictions = []
    all_input_ids = []
    all_targets = []

    # Determine total batches for progress bar
    total_batches = max_batches if max_batches is not None else len(dataloader)
    print("max_batches", max_batches)

    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader), total=total_batches, desc="Generating sequences"
        )
        for batch_idx, batch in pbar:
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["output_ids"].to(device)

            # Extract first target token per sample for teacher forcing
            if use_teacher_forcing:
                start_token_id = target_ids[:, 0]  # [B] - per-sample first tokens
            else:
                start_token_id = None  # Will use BOS=1

            logits_processor = None
            if use_constrained_decoding and input_vocab is not None:
                from src.constrained_decoding import create_constrained_processor
                logits_processor = create_constrained_processor(
                    input_ids,
                    input_vocab,
                    output_vocab,
                    mode=constrained_decoding_mode,
                    num_frets=num_frets,
                    device=device,
                )

            # Generate (now uses custom implementation)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                start_token_id=start_token_id,
                eos_token_id=output_vocab.eos_id if hasattr(output_vocab, "eos_id") else 2,
                pad_token_id=output_vocab.pad_id,
                temperature=temperature,
                verbose=False,
                logits_processor=logits_processor,
                disable_kv_cache=disable_kv_cache,
            )

            # Pad/trim both generated and targets to max_length for uniform shape
            B = target_ids.shape[0]
            gen_len = generated.shape[1]
            target_len = target_ids.shape[1]

            # Pad or trim generated to max_length
            if gen_len < max_length:
                padding = torch.full(
                    (B, max_length - gen_len),
                    output_vocab.pad_id,
                    dtype=generated.dtype,
                    device=generated.device,
                )
                generated = torch.cat([generated, padding], dim=1)
            elif gen_len > max_length:
                generated = generated[:, :max_length]

            # Pad or trim targets to max_length
            if target_len < max_length:
                padding = torch.full(
                    (B, max_length - target_len),
                    output_vocab.pad_id,
                    dtype=target_ids.dtype,
                    device=target_ids.device,
                )
                target_ids = torch.cat([target_ids, padding], dim=1)
            elif target_len > max_length:
                target_ids = target_ids[:, :max_length]

            # Note: input_ids also need padding for uniform concatenation
            # However, we keep them as-is since they don't need to match max_length
            # (they're already padded within batch by collate_fn)

            all_input_ids.append(input_ids)
            all_predictions.append(generated)
            all_targets.append(target_ids)

            # Update progress bar with current batch info
            pbar.set_postfix(
                {"batch_size": B, "gen_len": gen_len, "target_len": target_len}
            )

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Pad input_ids to uniform length across batches
    max_input_len = max(inp.shape[1] for inp in all_input_ids)
    padded_inputs = []
    for inp in all_input_ids:
        if inp.shape[1] < max_input_len:
            # Assume input uses pad_id=0 (standard for input vocab)
            padding = torch.zeros(
                (inp.shape[0], max_input_len - inp.shape[1]),
                dtype=inp.dtype,
                device=inp.device
            )
            inp = torch.cat([inp, padding], dim=1)
        padded_inputs.append(inp)
    input_ids = torch.cat(padded_inputs, dim=0)

    # Compute metrics
    metrics = compute_tablature_accuracy(
        predictions=predictions,
        targets=targets,
        output_vocab=output_vocab,
        input_ids=input_ids,
        input_vocab=input_vocab,
        # pad_id=output_vocab.pad_id,
    )

    return metrics, (input_ids, targets, predictions)

def extract_tab_positions(tokens):
    """
    tokens: list of string tokens like ["NOTE_ON_55", "TAB_3_0", ...]
    return: list of (string:int, fret:int)
    """
    positions = []
    for t in tokens:
        if t.startswith("TAB_"):
            # format: TAB_<string>_<fret>
            try:
                _, s, f = t.split("_")
                positions.append((int(s), int(f)))
            except ValueError:
                pass
    return positions

def difficulty_score(positions, alpha=0.25):
    """
    Difficulty score
    positions: list of (string, fret)
    return: mean difficulty score
    """

    if len(positions) < 2:
        return 0.0

    diffs = []

    for (s1, f1), (s2, f2) in zip(positions[:-1], positions[1:]):

        # fret stretch
        delta_fret = f2 - f1
        if delta_fret > 0:
            fret_stretch = 0.50 * abs(delta_fret)
        else:
            fret_stretch = 0.75 * abs(delta_fret)

        # locality 
        locality = alpha * (f1 + f2)

        along = fret_stretch + locality

        # across
        delta_string = abs(s2 - s1)
        across = 0.25 if delta_string <= 1 else 0.50

        difficulty = along + across
        diffs.append(difficulty)

    return sum(diffs) / len(diffs)
