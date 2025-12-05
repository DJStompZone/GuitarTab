"""
JAMS/MIDI Integration Utilities
================================

提供與現有 GuitarTab pipeline 整合的工具函數。
支援 JAMS → Tokens → Post-processing → MIDI 的完整流程。
"""

import json
import os
from typing import List, Tuple, Optional, Dict, Any
import mido

from .datatypes import Note
from .sequence import NoteSequence
from .config import GuitarConfig
from .parser import TokenParser
from .serializer import TokenSerializer
from .api import FrettingPostProcessor


def jams_to_tokens(jams_path: str) -> Tuple[List[str], List[str], GuitarConfig]:
    """
    從 JAMS 檔案提取 tokens 和 guitar configuration

    參考現有 jams2midi.py 的格式。

    Args:
        jams_path: JAMS 檔案路徑

    Returns:
        Tuple of (input_tokens, output_tokens, guitar_config):
            - input_tokens: NOTE_ON/OFF tokens (ground truth pitches)
            - output_tokens: TAB tokens (如果有 tablature 資訊)
            - guitar_config: GuitarConfig 物件

    Raises:
        FileNotFoundError: 如果檔案不存在
        ValueError: 如果 JAMS 格式不正確
    """
    # Load JAMS file
    if not os.path.exists(jams_path):
        raise FileNotFoundError(f"JAMS file not found: {jams_path}")

    with open(jams_path, 'r') as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    if not annotations:
        raise ValueError(f"No annotations found in JAMS file: {jams_path}")

    # Extract guitar configuration from first annotation
    guitar_config = None
    for ann in annotations:
        if ann.get("namespace") == "tempo":
            continue

        sandbox = ann.get("sandbox", {})
        if "open_tuning" in sandbox:
            # Extract tuning
            # JAMS 中每個 string 有一個 annotation，需要收集所有 strings
            break

    # Collect all strings to build tuning
    string_annotations = []
    for ann in annotations:
        if ann.get("namespace") == "tempo":
            continue
        string_annotations.append(ann)

    # Sort by string index
    string_annotations.sort(key=lambda x: x.get("sandbox", {}).get("string_index", 0))

    # Build tuning tuple
    tuning_list = []
    for ann in string_annotations:
        sandbox = ann.get("sandbox", {})
        open_tuning = sandbox.get("open_tuning", None)
        if open_tuning is not None:
            tuning_list.append(open_tuning)

    if not tuning_list:
        # Default to standard tuning
        tuning = (40, 45, 50, 55, 59, 64)  # E A D G B E
    else:
        tuning = tuple(tuning_list)

    # Get other config parameters
    first_sandbox = string_annotations[0].get("sandbox", {}) if string_annotations else {}
    fret_count = first_sandbox.get("fret_count", 24)
    instrument = first_sandbox.get("instrument", "guitar")

    guitar_config = GuitarConfig(
        tuning=tuning,
        fret_count=fret_count,
        capo_position=0  # JAMS doesn't store capo info
    )

    # Collect all note events
    # events = [(onset_ticks, type, pitch, duration, string, fret, velocity)]
    events = []

    for ann in string_annotations:
        sandbox = ann.get("sandbox", {})
        string_index = sandbox.get("string_index", 1)
        open_tuning = sandbox.get("open_tuning", 0)

        for note_data in ann.get("data", []):
            value = note_data.get("value", {})
            fret = value.get("fret", 0)
            velocity = value.get("velocity", 80)

            onset_ticks = int(note_data.get("time", 0))
            duration_ticks = int(note_data.get("duration", 0))

            pitch = fret + open_tuning

            events.append({
                'onset': onset_ticks,
                'duration': duration_ticks,
                'pitch': pitch,
                'string': string_index - 1,  # Convert to 0-indexed
                'fret': fret,
                'velocity': velocity
            })

    # Sort events by onset time
    events.sort(key=lambda x: x['onset'])

    # Generate input tokens (NOTE_ON/OFF format)
    input_tokens = []
    # Generate output tokens (TAB format)
    output_tokens = []

    # Track time for TIME_SHIFT generation
    current_time = 0

    # Create event list with explicit note-on/note-off
    event_list = []
    for ev in events:
        event_list.append({
            'time': ev['onset'],
            'type': 'note_on',
            'pitch': ev['pitch'],
            'velocity': ev['velocity'],
            'string': ev['string'],
            'fret': ev['fret']
        })
        event_list.append({
            'time': ev['onset'] + ev['duration'],
            'type': 'note_off',
            'pitch': ev['pitch'],
            'velocity': ev['velocity'],
            'string': ev['string'],
            'fret': ev['fret']
        })

    event_list.sort(key=lambda x: (x['time'], x['type'] == 'note_off'))

    # Generate tokens
    for event in event_list:
        event_time = event['time']

        # Add TIME_SHIFT if needed
        if event_time > current_time:
            time_shift = event_time - current_time
            input_tokens.append(f'TIME_SHIFT<{time_shift}>')
            output_tokens.append(f'TIME_SHIFT<{time_shift}>')
            current_time = event_time

        if event['type'] == 'note_on':
            # Input token: NOTE_ON
            input_tokens.append(f"NOTE_ON<{event['pitch']}>")
            # Output token: TAB
            output_tokens.append(f"TAB<{event['string']},{event['fret']}>")

        elif event['type'] == 'note_off':
            # Input token: NOTE_OFF
            input_tokens.append(f"NOTE_OFF<{event['pitch']}>")
            # Output doesn't need note_off for TAB format

    return input_tokens, output_tokens, guitar_config


def tokens_to_midi(tokens: List[str],
                  output_path: str,
                  guitar_config: GuitarConfig,
                  ticks_per_beat: int = 960) -> None:
    """
    將 tokens 轉換為 MIDI 檔案

    參考現有 jams2midi.py 的 MIDI 生成方式。

    Args:
        tokens: TAB tokens (post-processed output)
        output_path: 輸出 MIDI 檔案路徑
        guitar_config: Guitar 配置
        ticks_per_beat: MIDI ticks per beat (預設 960)

    Raises:
        ValueError: 如果 tokens 無效
    """
    # Parse tokens to notes
    parser = TokenParser()

    # 建立假的 input sequence（從 TAB tokens 推斷 pitches）
    events = []
    current_time = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith('TIME_SHIFT<'):
            # Extract time shift
            shift = int(token.split('<')[1].split('>')[0])
            current_time += shift

        elif token.startswith('TAB<'):
            # Extract string and fret
            params = token.split('<')[1].split('>')[0]
            string_str, fret_str = params.split(',')
            string = int(string_str)
            fret = int(fret_str)

            # Calculate pitch from guitar config
            pitch = guitar_config.tuning[string] + fret

            # Assume default duration (quarter note = 480 ticks)
            # In practice, you'd need duration info from tokens
            duration = 480

            events.append({
                'onset': current_time,
                'pitch': pitch,
                'velocity': 80,
                'duration': duration,
                'channel': string  # Use string as MIDI channel
            })

        i += 1

    # Create MIDI file
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Default tempo (120 BPM)
    track.append(mido.MetaMessage(
        'set_tempo',
        tempo=mido.bpm2tempo(120),
        time=0
    ))

    # Convert notes to MIDI messages
    midi_events = []
    for ev in events:
        # Note on
        midi_events.append({
            'time': ev['onset'],
            'type': 'note_on',
            'note': ev['pitch'],
            'velocity': ev['velocity'],
            'channel': ev['channel']
        })
        # Note off
        midi_events.append({
            'time': ev['onset'] + ev['duration'],
            'type': 'note_off',
            'note': ev['pitch'],
            'velocity': ev['velocity'],
            'channel': ev['channel']
        })

    # Sort by time
    midi_events.sort(key=lambda x: (x['time'], x['type'] == 'note_off'))

    # Write MIDI messages with delta times
    current_tick = 0
    for event in midi_events:
        delta = max(event['time'] - current_tick, 0)
        current_tick = event['time']

        if event['type'] == 'note_on':
            msg = mido.Message('note_on',
                              note=event['note'],
                              velocity=event['velocity'],
                              time=delta,
                              channel=event['channel'])
        else:
            msg = mido.Message('note_off',
                              note=event['note'],
                              velocity=event['velocity'],
                              time=delta,
                              channel=event['channel'])

        track.append(msg)

    # Save MIDI file
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    midi.save(output_path)


def process_jams_file(jams_path: str,
                     model_output_tokens: List[str],
                     output_midi_path: str,
                     method: str = 'neighbor_search',
                     verbose: bool = False) -> Dict[str, Any]:
    """
    完整處理 pipeline: JAMS → Post-processing → MIDI

    這是主要的整合函數，執行完整流程：
    1. 從 JAMS 提取 input tokens 和 config
    2. 使用 model output tokens 執行 post-processing
    3. 將修正後的 tokens 轉換為 MIDI

    Args:
        jams_path: 輸入 JAMS 檔案路徑
        model_output_tokens: 模型預測的 TAB tokens
        output_midi_path: 輸出 MIDI 檔案路徑
        method: Post-processing 方法 ('overlap' 或 'neighbor_search')
        verbose: 是否打印詳細資訊

    Returns:
        Dict 包含評估結果和處理資訊

    Raises:
        FileNotFoundError: 如果 JAMS 檔案不存在
        ValueError: 如果處理失敗
    """
    # Step 1: Load JAMS and extract config
    if verbose:
        print(f"Loading JAMS file: {jams_path}")

    input_tokens, ground_truth_tokens, config = jams_to_tokens(jams_path)

    if verbose:
        print(f"  Extracted {len(input_tokens)} input tokens")
        print(f"  Guitar config: tuning={config.tuning}")

    # Step 2: Apply post-processing
    if verbose:
        print(f"Applying post-processing (method={method})...")

    processor = FrettingPostProcessor(config)

    corrected_tokens, evaluation = processor.process_and_evaluate(
        model_output_tokens,
        input_tokens,
        method=method,
        verbose=verbose
    )

    # Step 3: Convert to MIDI
    if verbose:
        print(f"Generating MIDI file: {output_midi_path}")

    tokens_to_midi(corrected_tokens, output_midi_path, config)

    if verbose:
        print(f"Done! MIDI file saved to {output_midi_path}")

    return {
        'evaluation': evaluation,
        'num_input_tokens': len(input_tokens),
        'num_output_tokens': len(corrected_tokens),
        'guitar_config': config,
        'output_midi_path': output_midi_path
    }


def batch_process_jams_directory(jams_dir: str,
                                 output_dir: str,
                                 model_outputs: Dict[str, List[str]],
                                 method: str = 'neighbor_search',
                                 verbose: bool = True) -> List[Dict[str, Any]]:
    """
    批次處理整個目錄的 JAMS 檔案

    Args:
        jams_dir: JAMS 檔案目錄
        output_dir: 輸出 MIDI 檔案目錄
        model_outputs: Dict mapping JAMS filename → model output tokens
        method: Post-processing 方法
        verbose: 是否打印進度

    Returns:
        List of processing results

    Raises:
        FileNotFoundError: 如果目錄不存在
    """
    if not os.path.exists(jams_dir):
        raise FileNotFoundError(f"JAMS directory not found: {jams_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Find all JAMS files
    jams_files = []
    for root, dirs, files in os.walk(jams_dir):
        for file in files:
            if file.endswith('.jams'):
                jams_files.append(os.path.join(root, file))

    if verbose:
        print(f"Found {len(jams_files)} JAMS files")

    results = []

    for jams_path in jams_files:
        filename = os.path.basename(jams_path)
        base_name = os.path.splitext(filename)[0]

        if filename not in model_outputs:
            if verbose:
                print(f"Skipping {filename}: no model output provided")
            continue

        # Output MIDI path
        output_midi_path = os.path.join(output_dir, f"{base_name}_corrected.mid")

        try:
            result = process_jams_file(
                jams_path,
                model_outputs[filename],
                output_midi_path,
                method=method,
                verbose=False  # Don't print for each file
            )

            result['jams_path'] = jams_path
            result['filename'] = filename
            results.append(result)

            if verbose:
                eval_result = result['evaluation']['neighbor_search']
                print(f"✓ {filename}: {eval_result['pitch_accuracy']:.2f}% pitch accuracy")

        except Exception as e:
            if verbose:
                print(f"✗ {filename}: Error - {str(e)}")
            continue

    return results
