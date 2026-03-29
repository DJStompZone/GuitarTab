#!/usr/bin/env python3
"""
Demo: MIDI or audio file → predicted guitar tablature.

Supports MIDI (.mid) directly, or audio (.mp3 .wav .flac .ogg .m4a) via Basic Pitch AMT.
For audio input, install: pip install basic-pitch

Usage:
    python demo.py --input song.mid  --checkpoint outputs/combine_data/best_model.pt
    python demo.py --input song.mp3  --checkpoint outputs/combine_data/best_model.pt
    python demo.py --input song.mp3  --checkpoint outputs/combine_data/best_model.pt --max-bars 16 --output tab.txt
"""

import argparse
import os
import tempfile

import mido
import torch
import yaml
from pathlib import Path

from src.tab_dataset import TabDataset
from src.model import FrettingTransformer
from src.metrics import generate_and_compute_accuracy
from src.visualization import render_as_tablature
from src.dataloader import create_dataloader
from inference import decode_tokens_to_events


# Standard guitar tuning: E2, A2, D3, G3, B3, E4
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
MAX_FRET = 20
MAX_WAIT = 480
TGT_TPB = 960  # Target ticks per beat (DadaGP standard)


def pitch_to_tab(pitch):
    """Convert MIDI pitch to (string, fret) using heuristic (highest string first)."""
    for s in range(5, -1, -1):  # strings 6→1 (low E → high E)
        fret = pitch - STANDARD_TUNING[s]
        if 0 <= fret <= MAX_FRET:
            return s + 1, fret  # 1-indexed string
    return None


def midi_to_tokens_txt(midi_path):
    """Convert MIDI file to DadaGP .tokens.txt format string."""
    mid = mido.MidiFile(midi_path)
    src_tpb = mid.ticks_per_beat

    # Merge all tracks and collect note events (skip percussion on channel 9)
    tick_notes = {}
    abs_tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != 9:
            result = pitch_to_tab(msg.note)
            if result:
                tick_notes.setdefault(abs_tick, []).append(result)

    if not tick_notes:
        raise ValueError(f"No playable notes found in {midi_path} (all out of guitar range or percussion only)")

    # Add 2026-03-21: emit new_measure tokens at 4-beat bar boundaries so TabDataset
    # can find bar positions and create segments. Without new_measure, TabDataset
    # returns 0 segments ("No bar positions found").
    # Original: no new_measure tokens → 0 segments → crash
    bar_tgt = TGT_TPB * 4  # 3840 target ticks per bar (4/4 time assumed)
    next_bar_tgt = bar_tgt
    lines = ["artist:demo", "tempo:120", "start", "new_measure"]  # initial bar
    prev_tgt = 0
    for src_tick in sorted(tick_notes):
        tgt_tick = round(src_tick * TGT_TPB / src_tpb)
        delta = tgt_tick - prev_tgt
        while delta > 0:
            chunk = min(delta, MAX_WAIT)
            lines.append(f"wait:{chunk}")
            delta -= chunk
        prev_tgt = tgt_tick
        while next_bar_tgt <= tgt_tick:  # flush bar boundaries after waits, before notes
            lines.append("new_measure")
            next_bar_tgt += bar_tgt
        for s, f in sorted(tick_notes[src_tick]):
            lines.append(f"clean:note:s{s}:f{f}")
    lines.append("end")
    return "\n".join(lines)


AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}


def audio_to_midi_file(audio_path, tmp_midi_path):
    """Convert audio file to MIDI using Basic Pitch, write to tmp_midi_path."""
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        raise ImportError(
            "basic-pitch is required for audio input. Install with: pip install basic-pitch"
        )
    # Add 2026-03-21: ICASSP_2022_MODEL_PATH points to .tflite which crashes on NumPy 2.x.
    # Prefer the .onnx sibling file — onnxruntime is NumPy 2.x compatible.
    # Original: model_path = ICASSP_2022_MODEL_PATH
    model_path = ICASSP_2022_MODEL_PATH
    onnx_candidate = Path(str(ICASSP_2022_MODEL_PATH)).with_suffix('.onnx')
    if onnx_candidate.exists():
        model_path = str(onnx_candidate)
    print("Running Basic Pitch transcription (this may take a moment)...")
    _, midi_data, _ = predict(audio_path, model_path)
    midi_data.write(tmp_midi_path)


def events_to_midi(events, output_path, ticks_per_beat=480, bpm=120):
    """Convert decoded NOTE_ON/NOTE_OFF/TIME_SHIFT events to a MIDI file."""
    tempo = int(60_000_000 / bpm)
    scale = ticks_per_beat / TGT_TPB  # rescale 960-TPB ticks → output TPB (480)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    messages = []  # list of (abs_tick, mido.Message)
    current_tick = 0
    for event in events:
        if event.type == 'TIME_SHIFT':
            current_tick += int(event.delta * scale)
        elif event.type == 'NOTE_ON':
            messages.append((current_tick, mido.Message('note_on',  note=event.pitch, velocity=80, time=0)))
        elif event.type == 'NOTE_OFF':
            messages.append((current_tick, mido.Message('note_off', note=event.pitch, velocity=0,  time=0)))

    messages.sort(key=lambda x: x[0])
    prev_tick = 0
    for abs_tick, msg in messages:
        track.append(msg.copy(time=abs_tick - prev_tick))
        prev_tick = abs_tick
    mid.save(output_path)


def play_midi(midi_path):
    """Play a MIDI file using timidity or fluidsynth via subprocess."""
    import subprocess
    import shutil
    # Original: pygame.mixer approach (requires timidity config)
    # Modify 2026-03-21: use subprocess with timidity or fluidsynth instead
    if shutil.which('timidity'):
        subprocess.run(['timidity', midi_path])
        return
    sf2_candidates = [
        '/usr/share/sounds/sf2/FluidR3_GM.sf2',
        '/usr/share/sounds/sf2/default.sf2',
        '/usr/share/soundfonts/GeneralUser_GS.sf2',
    ]
    if shutil.which('fluidsynth'):
        for sf in sf2_candidates:
            if os.path.exists(sf):
                subprocess.run(['fluidsynth', '-a', 'pulseaudio', '-ni', sf, midi_path])
                return
        print("fluidsynth found but no soundfont. Install: sudo apt install fluid-soundfont-gm")
        return
    print("Cannot play MIDI automatically.")
    print("Install timidity:  sudo apt install timidity")
    print(f"Or open manually: {midi_path}")


def load_model(checkpoint_path, device):
    """Load model from checkpoint, reading model config from sibling config.yaml."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_path = Path(checkpoint_path).parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    model = FrettingTransformer(
        input_vocab_size=760,   # fixed for v1 format
        output_vocab_size=886,  # fixed for v1 format
        model_config=cfg["model"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    model.eval()
    return model, data_cfg


def main():
    parser = argparse.ArgumentParser(description="Demo: MIDI → Guitar Tab")
    parser.add_argument("--input", required=True, help="Path to input MIDI (.mid) or audio (.mp3/.wav/.flac/...)")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--max-bars", type=int, default=8, help="Max bars to render (default: 8)")
    parser.add_argument("--output", default=None, help="Save tab to text file (optional)")
    parser.add_argument("--output-midi", default=None, help="Save predicted tab as MIDI file (optional)")
    parser.add_argument("--play", action="store_true", help="Play the MIDI after generation (requires pygame)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"MIDI file not found: {args.input}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Processing: {args.input}")

    # Step 1: (Audio →) MIDI → temp .tokens.txt
    midi_tmp = None
    if Path(args.input).suffix.lower() in AUDIO_EXTENSIONS:
        midi_tmp_f = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        midi_tmp_f.close()
        midi_tmp = midi_tmp_f.name
        audio_to_midi_file(args.input, midi_tmp)
        midi_path = midi_tmp
    else:
        midi_path = args.input

    tokens_txt = midi_to_tokens_txt(midi_path)
    if midi_tmp:
        os.unlink(midi_tmp)
    note_count = sum(1 for line in tokens_txt.splitlines() if line.startswith("clean:note:"))
    print(f"Converted MIDI: {note_count} notes")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tokens.txt", delete=False) as f:
        f.write(tokens_txt)
        tmp = f.name

    try:
        # Step 2: Load model (reads data config from checkpoint's config.yaml)
        model, data_cfg = load_model(args.checkpoint, device)

        # Step 3: Create dataset from temp file using training config params
        dataset = TabDataset(
            token_files=[tmp],
            max_sequence_length=data_cfg.get("max_sequence_length", 512),
            max_pitch=data_cfg.get("max_pitch", 127),
            max_time_shift=data_cfg.get("max_time_shift", 500),
            num_strings=data_cfg.get("num_strings", 6),
            num_frets=data_cfg.get("num_frets", 21),
            output_format=data_cfg.get("output_format", "v1"),
        )
        print(f"Created {len(dataset)} segment(s)")

        loader = create_dataloader(dataset, batch_size=16, shuffle=False, num_workers=0)

        # Step 4: AR inference
        output_vocab = dataset.output_vocab
        print("Running AR inference...")
        _, (_, _, preds) = generate_and_compute_accuracy(
            model, loader, output_vocab, device, use_teacher_forcing=False
        )

        # Step 5: Decode and render
        all_events = []
        for seg in preds:
            toks = [output_vocab.id_to_token[t.item()] for t in seg]
            all_events.extend(decode_tokens_to_events(toks))

        tab = render_as_tablature(all_events, max_bars=args.max_bars)
        print("\n" + tab)

        if args.output:
            with open(args.output, "w") as f:
                f.write(tab)
            print(f"\nTab saved to: {args.output}")

        if args.output_midi or args.play:
            midi_out = args.output_midi or tempfile.mktemp(suffix=".mid")
            events_to_midi(all_events, midi_out)
            print(f"MIDI saved to: {midi_out}")
            if args.play:
                play_midi(midi_out)
            if not args.output_midi:
                os.unlink(midi_out)

    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    main()
