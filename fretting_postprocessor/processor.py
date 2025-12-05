"""
Post-Processor
==============

Core post-processing algorithms for Fretting-Transformer outputs.
Implements overlap correction and neighbor search algorithms from the paper.

Based on: Fretting-Transformer paper (arXiv:2506.14223v1)
- Section 3.5: Overlap Correction
- Section 4.2: Neighbor Search
"""

from typing import List, Optional, Tuple
from .datatypes import Note
from .config import GuitarConfig
from .sequence import NoteSequence
from .validator import PitchValidator


class PostProcessor:
    """
    Post-processing algorithms for guitar tablature generation.

    This class implements the two main algorithms from the paper:
    1. Overlap Correction: Matches model output with input notes to correct pitch errors
    2. Neighbor Search: Optimizes string-fret positions for playability

    The overlap correction algorithm achieves ~99.92% pitch accuracy,
    and neighbor search refines it to 100%.

    Attributes:
        config: Guitar configuration (tuning, fret range, etc.)
        validator: PitchValidator instance for tablature validation
        window_size: Size of search window for overlap correction (default: 5)

    Example:
        >>> config = GuitarConfig()
        >>> processor = PostProcessor(config)
        >>>
        >>> # Apply overlap correction
        >>> corrected = processor.overlap_correction(model_output, input_sequence)
        >>>
        >>> # Apply neighbor search for further refinement
        >>> refined = processor.neighbor_search(corrected)
    """

    def __init__(self, guitar_config: GuitarConfig, window_size: int = 5):
        """
        Initialize post-processor.

        Args:
            guitar_config: Guitar configuration
            window_size: Size of search window for overlap correction (default: 5)
                       Uses ±window_size notes around target position
        """
        self.config = guitar_config
        self.validator = PitchValidator()
        self.window_size = window_size

        # Scoring weights for overlap correction
        self.pitch_weight = 1000  # Most important factor
        self.time_weight = 10     # Secondary importance
        self.duration_weight = 1  # Least important

    def _find_best_match(
        self,
        model_note: Note,
        candidate_notes: List[Note]
    ) -> Optional[Note]:
        """
        Find the best matching input note for a model output note.

        This implements the matching algorithm from Section 3.5 of the paper.
        Scoring formula: score = (pitch_diff * 1000) + (time_diff * 10) + duration_diff

        Args:
            model_note: Note from model output
            candidate_notes: List of candidate notes from input sequence

        Returns:
            Best matching note, or None if no suitable match found

        Algorithm:
            1. Calculate predicted pitch from model_note's tablature
            2. For each candidate, compute weighted score:
               - Pitch difference (highest weight)
               - Time difference (medium weight)
               - Duration difference (lowest weight)
            3. Select candidate with minimum score
            4. Skip candidates that are already matched

        Example:
            >>> model_note = Note(pitch=45, onset_ticks=100, duration_ticks=480, ...)
            >>> candidates = [
            ...     Note(pitch=45, onset_ticks=96, duration_ticks=480, ...),  # Close match
            ...     Note(pitch=50, onset_ticks=500, duration_ticks=240, ...)  # Far match
            ... ]
            >>> best = processor._find_best_match(model_note, candidates)
            >>> best.pitch
            45  # First candidate selected
        """
        if not candidate_notes:
            return None

        # Calculate predicted pitch from model's tablature
        if model_note.has_tablature():
            try:
                predicted_pitch = model_note.get_pitch_from_tablature(
                    self.config.get_effective_tuning()
                )
            except (IndexError, TypeError):
                # Invalid tablature (e.g., string out of range)
                predicted_pitch = model_note.pitch
        else:
            predicted_pitch = model_note.pitch

        best_match = None
        min_score = float('inf')

        for candidate in candidate_notes:
            # Skip already matched notes
            if candidate.matched:
                continue

            # Calculate matching score components
            pitch_diff = abs(candidate.pitch - predicted_pitch)
            time_diff = abs(candidate.onset_ticks - model_note.onset_ticks)
            duration_diff = abs(candidate.duration_ticks - model_note.duration_ticks)

            # Weighted score (lower is better)
            score = (
                pitch_diff * self.pitch_weight +
                time_diff * self.time_weight +
                duration_diff * self.duration_weight
            )

            if score < min_score:
                min_score = score
                best_match = candidate

        return best_match

    def _create_fallback_note(self, model_note: Note) -> Note:
        """
        Create a fallback note when no matching input note is found.

        This generates a valid note using the model's timing but corrected
        tablature. Used when the overlap correction cannot find a suitable
        match in the input sequence.

        Args:
            model_note: Note from model output

        Returns:
            Note with corrected tablature

        Algorithm:
            1. Use model_note's pitch and timing
            2. Generate valid tablature using pitch_to_string_fret()
            3. Mark as "fallback" source for tracking

        Example:
            >>> model_note = Note(pitch=45, onset_ticks=0, duration_ticks=480,
            ...                   velocity=80, string=None, fret=None)
            >>> fallback = processor._create_fallback_note(model_note)
            >>> fallback.has_tablature()
            True
            >>> fallback.source
            'fallback'
        """
        # Create new note preserving model's timing
        fallback_note = Note(
            pitch=model_note.pitch,
            onset_ticks=model_note.onset_ticks,
            duration_ticks=model_note.duration_ticks,
            velocity=model_note.velocity if model_note.velocity > 0 else 80,
            source="fallback"
        )

        # Generate valid tablature for the pitch
        success = self.validator.correct_note_tablature(fallback_note, self.config)

        if not success:
            # Pitch cannot be played on this guitar
            # This shouldn't happen often, but we handle it gracefully
            # Try to use the closest valid pitch
            min_pitch, max_pitch = self.config.get_pitch_range()

            if fallback_note.pitch < min_pitch:
                fallback_note.pitch = min_pitch
            elif fallback_note.pitch > max_pitch:
                fallback_note.pitch = max_pitch

            # Try again with adjusted pitch
            self.validator.correct_note_tablature(fallback_note, self.config)

        return fallback_note

    def overlap_correction(
        self,
        model_output: NoteSequence,
        input_sequence: NoteSequence
    ) -> NoteSequence:
        """
        Apply overlap correction algorithm to fix pitch errors.

        This implements the algorithm from Section 3.5 of the paper.
        Matches each model output note with input notes in a ±window_size window,
        using the input note's pitch (ground truth) while preserving model's timing.

        Expected improvement: ~97% → ~99.92% pitch accuracy

        Args:
            model_output: Sequence of notes predicted by model (with tablature)
            input_sequence: Ground truth input sequence (MIDI pitches)

        Returns:
            Corrected note sequence with improved pitch accuracy

        Algorithm:
            FOR EACH model_note IN model_output:
                1. Get candidates in ±window_size window from input_sequence
                2. Find best_match using _find_best_match()
                3. IF match found:
                   - Use input pitch (ground truth)
                   - Keep model's timing and preferred string
                   - Recalculate fret for the corrected pitch
                   - Validate and use fallback if needed
                   - Mark input note as matched
                4. ELSE:
                   - Create fallback_note
                5. Add to corrected_notes

        Example:
            >>> # Model output has pitch error
            >>> model_output = NoteSequence([
            ...     Note(pitch=47, onset_ticks=0, duration_ticks=480,  # Wrong pitch!
            ...          velocity=80, string=1, fret=2)
            ... ])
            >>> # Input has correct pitch
            >>> input_sequence = NoteSequence([
            ...     Note(pitch=45, onset_ticks=0, duration_ticks=480,
            ...          velocity=80)  # Correct: A2
            ... ])
            >>> corrected = processor.overlap_correction(model_output, input_sequence)
            >>> corrected.notes[0].pitch
            45  # Corrected to ground truth
        """
        corrected_notes = []
        effective_tuning = self.config.get_effective_tuning()

        for model_note in model_output:
            # Step 1: Get candidate notes in window
            window_notes = input_sequence.get_notes_in_window(
                model_note.onset_ticks,
                self.window_size
            )

            # Step 2: Find best matching input note
            best_match = self._find_best_match(model_note, window_notes)

            if best_match is not None:
                # Step 3a: Match found - use input pitch (ground truth)
                corrected_note = Note(
                    pitch=best_match.pitch,  # Ground truth pitch
                    onset_ticks=model_note.onset_ticks,  # Model timing
                    duration_ticks=model_note.duration_ticks,
                    velocity=best_match.velocity,
                    source="corrected"
                )

                # Try to preserve model's string choice if valid
                if model_note.has_tablature():
                    corrected_note.string = model_note.string

                    # Recalculate fret for the corrected pitch
                    if self.config.is_valid_string(corrected_note.string):
                        corrected_note.fret = (
                            corrected_note.pitch -
                            effective_tuning[corrected_note.string]
                        )

                        # Validate the recalculated tablature
                        if not self.validator.validate_note(corrected_note, self.config):
                            # Invalid - use validator to find valid position
                            self.validator.correct_note_tablature(
                                corrected_note,
                                self.config,
                                preferred_string=model_note.string
                            )
                    else:
                        # Invalid string - correct tablature
                        self.validator.correct_note_tablature(
                            corrected_note,
                            self.config
                        )
                else:
                    # Model didn't have tablature - generate it
                    self.validator.correct_note_tablature(
                        corrected_note,
                        self.config
                    )

                # Mark input note as matched to avoid reuse
                best_match.matched = True
                corrected_notes.append(corrected_note)

            else:
                # Step 3b: No match found - use fallback
                fallback_note = self._create_fallback_note(model_note)
                corrected_notes.append(fallback_note)

        return NoteSequence(corrected_notes, source="overlap_corrected")

    def process(
        self,
        model_output: NoteSequence,
        input_sequence: NoteSequence,
        apply_neighbor_search: bool = True
    ) -> NoteSequence:
        """
        Complete post-processing pipeline.

        Convenience method that applies both overlap correction and optionally
        neighbor search in sequence.

        Args:
            model_output: Model predicted sequence
            input_sequence: Ground truth input sequence
            apply_neighbor_search: Whether to apply neighbor search after
                                 overlap correction (default: True)

        Returns:
            Fully processed note sequence

        Example:
            >>> result = processor.process(model_output, input_sequence)
            >>> # Equivalent to:
            >>> corrected = processor.overlap_correction(model_output, input_sequence)
            >>> result = processor.neighbor_search(corrected)
        """
        # Apply overlap correction
        corrected = self.overlap_correction(model_output, input_sequence)

        # Optionally apply neighbor search
        if apply_neighbor_search:
            return self.neighbor_search(corrected)

        return corrected

    def _get_context_notes(
        self,
        notes: List[Note],
        current_idx: int,
        context_window: int = 3
    ) -> Tuple[List[Note], List[Note]]:
        """
        Get context notes around current position for neighbor search.

        Args:
            notes: List of all notes in sequence
            current_idx: Index of current note
            context_window: Number of notes to look before/after (default: 3)

        Returns:
            Tuple of (previous_notes, following_notes)

        Example:
            >>> notes = [note1, note2, note3, note4, note5]
            >>> prev, next = processor._get_context_notes(notes, 2, context_window=1)
            >>> # prev = [note2], next = [note4]
        """
        prev_start = max(0, current_idx - context_window)
        next_end = min(len(notes), current_idx + context_window + 1)

        previous_notes = notes[prev_start:current_idx]
        following_notes = notes[current_idx + 1:next_end]

        return previous_notes, following_notes

    def _evaluate_position(
        self,
        string: int,
        fret: int,
        previous_notes: List[Note],
        optimize_for: str = "balanced"
    ) -> float:
        """
        Evaluate a specific (string, fret) position for playability.

        Implements the scoring system from Section 4.2 of the paper.
        Lower score is better.

        Args:
            string: String index to evaluate
            fret: Fret number to evaluate
            previous_notes: Previous notes for context
            optimize_for: Optimization strategy ("playability", "position_stability", "balanced")

        Returns:
            Score (lower is better)

        Scoring factors:
            1. String consistency: -20 if same string as recent notes
            2. Position proximity: +5 per fret distance on same string
            3. Playability: +0.5 per fret (prefer lower frets)
            4. Avoid extremes: +10 if fret > 15

        Example:
            >>> score = processor._evaluate_position(
            ...     string=1, fret=5,
            ...     previous_notes=[Note(..., string=1, fret=3)]
            ... )
            >>> # Low score because same string and close fret
        """
        score = 0.0

        # Factor 1: String consistency (prefer same string as neighbors)
        # Look at last 2-3 notes
        recent_notes = previous_notes[-3:] if len(previous_notes) >= 3 else previous_notes

        for prev_note in recent_notes:
            if prev_note.has_tablature() and prev_note.string == string:
                score -= 20  # Reward for string consistency

        # Factor 2: Position proximity (minimize hand movement)
        if previous_notes and previous_notes[-1].has_tablature():
            prev_note = previous_notes[-1]

            if prev_note.string == string:
                # Same string - penalize large fret jumps
                fret_distance = abs(fret - prev_note.fret)
                score += fret_distance * 5

        # Factor 3: Playability (prefer lower frets for easier playing)
        if optimize_for in ["playability", "balanced"]:
            score += fret * 0.5  # Slight preference for lower frets

        # Factor 4: Avoid extreme positions
        if fret > 15:
            score += 10  # Penalize high frets

        return score

    def neighbor_search(
        self,
        corrected_sequence: NoteSequence,
        optimize_for: str = "balanced"
    ) -> NoteSequence:
        """
        Apply neighbor search algorithm for tablature optimization.

        This implements the algorithm from Section 4.2 of the paper.
        For each note, explores all alternative (string, fret) positions that
        produce the same pitch, and selects the best one based on context.

        Expected result: 100% pitch accuracy with optimized playability

        Args:
            corrected_sequence: Sequence after overlap correction
            optimize_for: Optimization strategy:
                - "playability": Prefer lower frets (easier to play)
                - "position_stability": Minimize hand movement
                - "balanced": Balance both factors (default)

        Returns:
            Optimized sequence with improved tablature choices

        Algorithm:
            FOR EACH note IN corrected_sequence:
                1. Get all alternative (string, fret) positions for same pitch
                2. If only one position exists, keep it
                3. Get context (previous and following notes)
                4. Score each alternative position
                5. Select position with lowest score
                6. Create refined note with optimal position

        Example:
            >>> corrected = processor.overlap_correction(model_output, input_sequence)
            >>> refined = processor.neighbor_search(corrected)
            >>> # All notes now have optimal string-fret positions
        """
        notes_list = list(corrected_sequence.notes)
        refined_notes = []

        for i, note in enumerate(notes_list):
            # Skip notes without tablature
            if not note.has_tablature():
                refined_notes.append(note)
                continue

            # Step 1: Get all alternative positions for this pitch
            alternatives = self.config.pitch_to_string_fret(note.pitch)

            if len(alternatives) <= 1:
                # Only one valid position, keep it
                refined_notes.append(note)
                continue

            # Step 2: Get context notes
            previous_notes = refined_notes  # Already processed notes
            _, _ = self._get_context_notes(notes_list, i, context_window=3)

            # Step 3: Score each alternative position
            best_score = float('inf')
            best_position = (note.string, note.fret)

            for string, fret in alternatives:
                score = self._evaluate_position(
                    string, fret,
                    previous_notes,
                    optimize_for
                )

                if score < best_score:
                    best_score = score
                    best_position = (string, fret)

            # Step 4: Create refined note with optimal position
            refined_note = Note(
                pitch=note.pitch,
                onset_ticks=note.onset_ticks,
                duration_ticks=note.duration_ticks,
                velocity=note.velocity,
                string=best_position[0],
                fret=best_position[1],
                source="refined"
            )

            refined_notes.append(refined_note)

        return NoteSequence(refined_notes, source="neighbor_search")
