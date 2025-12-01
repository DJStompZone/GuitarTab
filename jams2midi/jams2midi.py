# -*- coding: utf-8 -*-

import json, os, sys, shutil
import mido as Mido
from tqdm import tqdm

def jams_to_single_midi(jams_path, midi_output_path):
    """
    Convert a JAMS file (multi-string guitar annotation)
    into a single MIDI file with multiple channels.
    Uses the JAMS time/duration directly as TICKS.
    Supports multiple tempo changes.
    """

    # Load JSON
    with open(jams_path, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]


    # Collect tempo change events (ticks)
    tempo_events = []   # (time_in_ticks, bpm)

    for ann in annotations:
        if ann["namespace"] == "tempo":
            for t in ann["data"]:
                bpm = t.get("value", 120)
                time_tick = int(t.get("time", 0))

                # clamp bpm to avoid invalid tempo
                if bpm < 30 or bpm > 300:
                    bpm = 120

                tempo_events.append((time_tick, bpm))

    # Default tempo
    if len(tempo_events) == 0:
        tempo_events = [(0, 120)]

    # Sort tempo by time
    tempo_events = sorted(tempo_events, key=lambda x: x[0])

    # Gather notes from all strings
    # events = (time, type, pitch, velocity, channel)
    events = []

    for ann in annotations:
        if ann["namespace"] == "tempo":
            continue

        string_index = ann["sandbox"]["string_index"]
        open_tuning = ann["sandbox"]["open_tuning"]
        channel = string_index - 1

        for nt in ann["data"]:
            pitch = nt["value"]["fret"] + open_tuning
            velocity = nt["value"]["velocity"]

            start = int(nt["time"])        # already ticks
            dur = int(nt["duration"])      # already ticks
            end = start + dur

            # note-on
            events.append((start, "on", pitch, velocity, channel))
            # note-off
            events.append((end, "off", pitch, velocity, channel))

    # Sort all note events by time
    events = sorted(events, key=lambda x: x[0])

    # Create MIDI file
    midi = Mido.MidiFile(ticks_per_beat=960)
    track = Mido.MidiTrack()
    midi.tracks.append(track)

    # Write tempo changes
    current_tick = 0
    for t_tick, bpm in tempo_events:
        delta = max(t_tick - current_tick, 0)
        current_tick = t_tick

        track.append(Mido.MetaMessage(
            'set_tempo',
            tempo=Mido.bpm2tempo(bpm),
            time=delta
        ))

    # Write notes
    current_tick = 0
    for t, tp, pitch, vel, ch in events:
        delta = max(t - current_tick, 0)
        current_tick = t

        if tp == "on":
            msg = Mido.Message('note_on', note=pitch, velocity=vel,
                               time=delta, channel=ch)
        else:
            msg = Mido.Message('note_off', note=pitch, velocity=vel,
                               time=delta, channel=ch)

        track.append(msg)

    midi.save(midi_output_path)


if __name__ == "__main__":
    targetDir = sys.argv[1]
    outputDir = sys.argv[2]

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # find all JAMS
    allJams = []
    for root, dirs, files in os.walk(targetDir):
        for file in files:
            if file.endswith(".jams"):
                allJams.append(os.path.join(root, file))

    for jam in tqdm(allJams, desc="Converting JAMS to SINGLE MIDI"):
        song_name = jam.split("/")[-2]
        jams_base = os.path.splitext(os.path.basename(jam))[0]

        midi_name = f"{song_name}__{jams_base}.mid"
        output_path = os.path.join(outputDir, midi_name)

        try:
            jams_to_single_midi(jam, output_path)
        except Exception as e:
            print("Error converting:", jam)
            print(e)
            with open(os.path.join(outputDir, "error.txt"), "a") as f:
                f.write(jam + " : " + str(e) + "\n")
