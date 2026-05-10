"""
Evaluation metrics for guitar tablature transcription.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Literal
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
    # Unified (v1/v2): pred TAB pitch vs target TAB pitch (same inferred tuning)
    pitch_accuracy: float  # % of notes with correct pitch
    tab_accuracy: float  # % of notes with correct (string, fret)

    # playability score (difficulty)
    difficulty: float

    # Counts
    total_tokens: int
    total_notes: int
    # Legacy v1 metric: NOTE_ON token exact-match accuracy
    note_token_pitch_accuracy: Optional[float] = None

    def __str__(self) -> str:
        legacy_part = ""
        if self.note_token_pitch_accuracy is not None:
            legacy_part = f"Pitch Acc (NOTE_ON token): {self.note_token_pitch_accuracy:.2%} | "
        return (
            f"Token Acc: {self.token_accuracy:.2%} | "
            f"Pitch Acc: {self.pitch_accuracy:.2%} | "
            f"{legacy_part}"
            f"Tab Acc: {self.tab_accuracy:.2%} | "
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
        input_ids: Optional [B, L_input] - Input token IDs (for per-sample tuning inference)
        input_vocab: Optional input vocabulary (for tuning inference)
        pad_id: Padding token ID to ignore

    Returns:
        TabAccuracyMetrics with computed accuracies
    """
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
    
    # Legacy v1 metric (original): NOTE_ON token exact match.
    note_token_pitch_accuracy: Optional[float] = None
    if has_note_on:
        note_positions = [
            i for i, token in enumerate(target_tokens) if token.startswith("NOTE_ON_")
        ]
        if note_positions:
            pitch_token_correct = sum(
                1 for pos in note_positions if pred_tokens[pos] == target_tokens[pos]
            )
            note_token_pitch_accuracy = pitch_token_correct / len(note_positions)

    # --- Unified v1/v2 note-level logic ---
    if predictions.dim() != 2 or targets.dim() != 2:
        return TabAccuracyMetrics(
            token_accuracy=token_accuracy,
            pitch_accuracy=0.0,
            note_token_pitch_accuracy=note_token_pitch_accuracy,
            tab_accuracy=0.0,
            difficulty=difficulty,
            total_tokens=total_tokens,
            total_notes=0,
        )

    tab_correct = 0
    pitch_correct = 0
    total_notes = 0
    batch_size = targets.shape[0]

    for b in range(batch_size):
        sample_target_ids = targets[b]
        sample_pred_ids = predictions[b]

        # Keep model-vs-target tab alignment on output sequence.
        sample_mask = sample_target_ids != output_vocab.pad_id
        sample_target_valid = sample_target_ids[sample_mask]
        sample_pred_valid = sample_pred_ids[sample_mask]

        sample_target_tokens = [
            output_vocab.id_to_token[idx.item()] for idx in sample_target_valid
        ]
        sample_pred_tokens = [
            output_vocab.id_to_token[idx.item()] for idx in sample_pred_valid
        ]

        tab_indices = [
            i for i, token in enumerate(sample_target_tokens) if token.startswith("TAB_")
        ]
        if not tab_indices:
            continue

        # Infer per-sample tuning whenever possible (critical for non-standard tuning songs).
        tuning_list = None
        if input_ids is not None and input_vocab is not None and b < input_ids.shape[0]:
            sample_tuning = infer_tuning_from_input_output(
                input_ids[b], sample_target_ids, input_vocab, output_vocab
            )
            if sample_tuning is not None:
                tuning_list = [sample_tuning.get(s, GUITAR_TUNING[s - 1]) for s in range(1, 7)]

        for tab_pos in tab_indices:
            total_notes += 1

            target_tab = sample_target_tokens[tab_pos]
            pred_tab = sample_pred_tokens[tab_pos]

            if pred_tab == target_tab:
                tab_correct += 1

            target_pitch = compute_pitch_from_tab_token(target_tab, tuning=tuning_list)
            pred_pitch = compute_pitch_from_tab_token(pred_tab, tuning=tuning_list)
            if (
                target_pitch != -1
                and pred_pitch != -1
                and target_pitch == pred_pitch
            ):
                pitch_correct += 1

    tab_accuracy = tab_correct / total_notes if total_notes > 0 else 0.0
    pitch_accuracy = pitch_correct / total_notes if total_notes > 0 else 0.0


    return TabAccuracyMetrics(
        token_accuracy=token_accuracy,
        pitch_accuracy=pitch_accuracy,
        note_token_pitch_accuracy=note_token_pitch_accuracy,
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
    constrained_decoding_pitch_mask: bool = True,  # C3=True, C2=False
    num_frets: int = 25,
    disable_kv_cache: bool = False,
    collect_logit_stats: bool = False,       # Exp 5 / Exp B: per-TAB-step logit stats
    collect_structural_logit_stats: bool = False,  # Exp A: per-FIXED_TOKEN logit stats
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
    all_logit_stats: List = []            # collected when collect_logit_stats=True
    all_structural_logit_stats: List = [] # collected when collect_structural_logit_stats=True
    global_sample_offset = 0              # cumulative sample count across batches

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
            if input_vocab is not None and (use_constrained_decoding or collect_logit_stats or collect_structural_logit_stats):
                from src.constrained_decoding import create_constrained_processor
                tuning_batch = None
                if constrained_decoding_mode == "input_skeleton":
                    tuning_batch = []
                    batch_size_now = input_ids.shape[0]
                    for b in range(batch_size_now):
                        sample_tuning = infer_tuning_from_input_output(
                            input_ids[b], target_ids[b], input_vocab, output_vocab
                        )
                        if sample_tuning is None:
                            tuning_batch.append(GUITAR_TUNING)
                        else:
                            tuning_batch.append(
                                [sample_tuning.get(s, GUITAR_TUNING[s - 1]) for s in range(1, 7)]
                            )
                # stats_only=True when we want TAB/structural stats without constraining
                # (i.e. unconstrained run but still want to collect logit stats)
                _stats_only = (not use_constrained_decoding) and (collect_logit_stats or collect_structural_logit_stats)
                logits_processor = create_constrained_processor(
                    input_ids,
                    input_vocab,
                    output_vocab,
                    mode=constrained_decoding_mode,
                    tuning_batch=tuning_batch,
                    num_frets=num_frets,
                    device=device,
                    collect_stats=collect_logit_stats,
                    collect_structural_stats=collect_structural_logit_stats,
                    stats_only=_stats_only,
                    pitch_mask=constrained_decoding_pitch_mask,
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

            # Collect TAB logit stats from this batch (Exp 5 / Exp B)
            if collect_logit_stats and logits_processor is not None and hasattr(logits_processor, "get_batch_stats"):
                batch_stats = logits_processor.get_batch_stats(sample_offset=global_sample_offset)
                all_logit_stats.extend(batch_stats)

            # Collect structural logit stats from this batch (Exp A)
            if collect_structural_logit_stats and logits_processor is not None and hasattr(logits_processor, "get_batch_structural_stats"):
                batch_struct_stats = logits_processor.get_batch_structural_stats(sample_offset=global_sample_offset)
                all_structural_logit_stats.extend(batch_struct_stats)

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
            global_sample_offset += input_ids.shape[0]

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
    )

    return metrics, (input_ids, targets, predictions), all_logit_stats, all_structural_logit_stats

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

def classify_aligned_note_error(
    target_string: Optional[int],
    target_fret: Optional[int],
    pred_string: Optional[int],
    pred_fret: Optional[int],
    tuning: Optional[List[int]] = None,
) -> Literal["correct", "Tab_3.1", "Tab_3.2"]:
    """
    Classify a time-aligned (matched) note pair into the format-invariant error taxonomy.

    Classes:
      correct  — (string, fret) identical → pitch and fingering both right
      Tab_3.1  — same pitch, different fingering (enharmonic position error)
      Tab_3.2  — different pitch (and necessarily different fingering)

    G (grammar) and T (time-alignment) errors are at the sequence level and
    must be counted before calling this function.

    Args:
        target_string, target_fret: Ground-truth tab position (1-indexed string).
        pred_string, pred_fret: Predicted tab position.
        tuning: Open-string MIDI pitches [s1..s6]. Defaults to GUITAR_TUNING.

    Returns:
        One of "correct", "Tab_3.1", "Tab_3.2".
    """
    if (
        target_string is None
        or target_fret is None
        or pred_string is None
        or pred_fret is None
    ):
        return "Tab_3.2"

    tab_match = (target_string, target_fret) == (pred_string, pred_fret)
    if tab_match:
        return "correct"

    t_tok = f"TAB_{target_string}_{target_fret}"
    p_tok = f"TAB_{pred_string}_{pred_fret}"
    target_pitch = compute_pitch_from_tab_token(t_tok, tuning=tuning)
    pred_pitch = compute_pitch_from_tab_token(p_tok, tuning=tuning)

    if target_pitch == -1 or pred_pitch == -1:
        return "Tab_3.2"

    if target_pitch == pred_pitch:
        return "Tab_3.1"
    return "Tab_3.2"


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
