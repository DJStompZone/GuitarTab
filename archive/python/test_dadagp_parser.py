#!/usr/bin/env python3
"""
Test script for DadaGP parser.
"""

from src.dadagp_parser import parse_dadagp_file_to_events, NoteOnEvent, NoteOffEvent, TimeShiftEvent, TabEvent


def main():
    # Test with the example file
    test_file = "DadaGP-v1.1/H/Haggard/Haggard - Prophecy Fullfilled (Intro).gp3.tokens.txt"

    print(f"Parsing: {test_file}")
    print("=" * 60)

    input_events, output_events = parse_dadagp_file_to_events(test_file)

    print(f"\nInput events: {len(input_events)}")
    print(f"Output events: {len(output_events)}")
    print(f"Extra events in output (TAB tokens): {len(output_events) - len(input_events)}")

    print("\nFirst 20 input events:")
    for i, event in enumerate(input_events[:20]):
        if isinstance(event, NoteOnEvent):
            print(f"  {i:3d}: NOTE_ON  pitch={event.pitch}")
        elif isinstance(event, NoteOffEvent):
            print(f"  {i:3d}: NOTE_OFF pitch={event.pitch}")
        elif isinstance(event, TimeShiftEvent):
            print(f"  {i:3d}: TIME_SHIFT delta={event.delta}")

    print("\nFirst 20 output events:")
    for i, event in enumerate(output_events[:20]):
        if isinstance(event, NoteOnEvent):
            print(f"  {i:3d}: NOTE_ON  pitch={event.pitch}")
        elif isinstance(event, NoteOffEvent):
            print(f"  {i:3d}: NOTE_OFF pitch={event.pitch}")
        elif isinstance(event, TimeShiftEvent):
            print(f"  {i:3d}: TIME_SHIFT delta={event.delta}")
        elif isinstance(event, TabEvent):
            print(f"  {i:3d}: TAB string={event.string} fret={event.fret}")

    print("\n" + "=" * 60)
    print("✓ Parsing successful!")


if __name__ == "__main__":
    main()
