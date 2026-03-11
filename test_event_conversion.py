#!/usr/bin/env python3
"""
Test script to verify event-to-token conversion correctness.
"""

import sys
from src.tab_dataset import build_vocabulary
from src.post_processing import (
    id_to_event,
    ids_to_events,
    event_to_token_string,
    events_to_ids,
    NoteOnEvent,
    NoteOffEvent,
    TimeShiftEvent,
    TabEvent
)


def test_roundtrip_conversion():
    """Test: token_id -> event -> token_string -> token_id"""
    print("="*80)
    print("TEST 1: Roundtrip Conversion (ID -> Event -> Token String -> ID)")
    print("="*80)

    # Build vocabularies
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=127,
        max_time_shift=500,
        num_strings=6,
        num_frets=21
    )

    # Test sample token IDs from output vocab
    test_cases = []

    # Find some NOTE_ON tokens
    note_on_ids = [id for id, token in output_vocab.id_to_token.items()
                   if token.startswith("NOTE_ON_")][:5]
    test_cases.extend(note_on_ids)

    # Find some NOTE_OFF tokens
    note_off_ids = [id for id, token in output_vocab.id_to_token.items()
                    if token.startswith("NOTE_OFF_")][:5]
    test_cases.extend(note_off_ids)

    # Find some TAB tokens
    tab_ids = [id for id, token in output_vocab.id_to_token.items()
               if token.startswith("TAB_")][:5]
    test_cases.extend(tab_ids)

    # Find some TIME_SHIFT tokens
    time_ids = [id for id, token in output_vocab.id_to_token.items()
                if token.startswith("TIME_SHIFT_")][:5]
    test_cases.extend(time_ids)

    success_count = 0
    fail_count = 0

    for token_id in test_cases:
        # Original token string
        original_token = output_vocab.id_to_token[token_id]

        # Convert: ID -> Event
        event = id_to_event(token_id, output_vocab)

        if event is None:
            print(f"❌ FAIL: token_id={token_id} ({original_token}) -> event=None")
            fail_count += 1
            continue

        # Convert: Event -> Token String
        reconstructed_token = event_to_token_string(event)

        # Convert: Token String -> ID
        reconstructed_id = output_vocab.token_to_id.get(reconstructed_token, output_vocab.unk_id)

        # Check if roundtrip is successful
        if reconstructed_id == token_id and reconstructed_token == original_token:
            success_count += 1
        else:
            print(f"❌ FAIL: {original_token} (ID={token_id})")
            print(f"   Event: {event}")
            print(f"   Reconstructed: {reconstructed_token} (ID={reconstructed_id})")
            fail_count += 1

    print(f"\nResults: ✅ {success_count} passed, ❌ {fail_count} failed")
    print()

    return fail_count == 0


def test_event_creation():
    """Test: Create events manually and check token conversion"""
    print("="*80)
    print("TEST 2: Manual Event Creation and Conversion")
    print("="*80)

    input_vocab, output_vocab = build_vocabulary(
        max_pitch=127,
        max_time_shift=500,
        num_strings=6,
        num_frets=21
    )

    # Create test events
    test_events = [
        (NoteOnEvent(pitch=60), "NOTE_ON_60"),
        (NoteOnEvent(pitch=64), "NOTE_ON_64"),
        (TabEvent(string=3, fret=5), "TAB_3_5"),
        (NoteOffEvent(pitch=60), "NOTE_OFF_60"),
        (TimeShiftEvent(delta=240), "TIME_SHIFT_240"),
        (TabEvent(string=1, fret=0), "TAB_1_0"),
        (TabEvent(string=6, fret=21), "TAB_6_21"),
    ]

    success_count = 0
    fail_count = 0

    for event, expected_token in test_events:
        # Convert event to token string
        token_string = event_to_token_string(event)

        # Check if it matches expected
        if token_string != expected_token:
            print(f"❌ FAIL: Event {event} -> '{token_string}' (expected '{expected_token}')")
            fail_count += 1
            continue

        # Check if token exists in vocabulary
        token_id = output_vocab.token_to_id.get(token_string, None)

        if token_id is None:
            print(f"❌ FAIL: Token '{token_string}' NOT in vocabulary")
            fail_count += 1
        else:
            print(f"✅ PASS: {event} -> '{token_string}' (ID={token_id})")
            success_count += 1

    print(f"\nResults: ✅ {success_count} passed, ❌ {fail_count} failed")
    print()

    return fail_count == 0


def test_out_of_range_values():
    """Test: Check behavior with out-of-range values"""
    print("="*80)
    print("TEST 3: Out-of-Range Values")
    print("="*80)

    input_vocab, output_vocab = build_vocabulary(
        max_pitch=88,  # Smaller range
        max_time_shift=100,  # Smaller range
        num_strings=6,
        num_frets=21
    )

    # Create events that might be out of range
    test_events = [
        (NoteOnEvent(pitch=100), "NOTE_ON_100", "Pitch > max_pitch"),
        (TimeShiftEvent(delta=200), "TIME_SHIFT_200", "Delta > max_time_shift"),
        (TabEvent(string=3, fret=22), "TAB_3_22", "Fret > num_frets"),
    ]

    for event, expected_token, description in test_events:
        token_string = event_to_token_string(event)
        token_id = output_vocab.token_to_id.get(token_string, None)

        print(f"\n{description}:")
        print(f"  Event: {event}")
        print(f"  Token: '{token_string}'")

        if token_id is None:
            print(f"  ⚠️  Token NOT in vocabulary (will become UNK={output_vocab.unk_id})")
        else:
            print(f"  ✅ Token in vocabulary (ID={token_id})")

    print()


def test_batch_conversion():
    """Test: Batch conversion events_to_ids"""
    print("="*80)
    print("TEST 4: Batch Conversion (events_to_ids)")
    print("="*80)

    input_vocab, output_vocab = build_vocabulary(
        max_pitch=127,
        max_time_shift=500,
        num_strings=6,
        num_frets=21
    )

    # Create a sequence of events
    events = [
        NoteOnEvent(pitch=60),
        TabEvent(string=3, fret=5),
        NoteOffEvent(pitch=60),
        TimeShiftEvent(delta=240),
        NoteOnEvent(pitch=64),
        TabEvent(string=3, fret=7),
        NoteOffEvent(pitch=64),
    ]

    print("Converting events:")
    for i, event in enumerate(events):
        print(f"  {i}: {event}")

    # Convert to IDs
    ids = events_to_ids(events, output_vocab)

    print(f"\nResulting IDs: {ids}")

    # Check for UNK tokens
    unk_count = sum(1 for id in ids if id == output_vocab.unk_id)

    if unk_count > 0:
        print(f"❌ FAIL: {unk_count} tokens converted to UNK")
        # Show which ones
        for i, (event, id) in enumerate(zip(events, ids)):
            if id == output_vocab.unk_id:
                token = event_to_token_string(event)
                print(f"  Position {i}: {event} -> '{token}' -> UNK")
        return False
    else:
        print(f"✅ PASS: All tokens converted successfully (no UNK)")

        # Verify roundtrip
        print("\nVerifying roundtrip:")
        reconstructed_events = ids_to_events(ids, output_vocab)

        if len(reconstructed_events) != len(events):
            print(f"❌ FAIL: Length mismatch ({len(reconstructed_events)} vs {len(events)})")
            return False

        for i, (orig, recon) in enumerate(zip(events, reconstructed_events)):
            if type(orig) != type(recon):
                print(f"❌ FAIL at position {i}: type mismatch")
                print(f"   Original: {orig}")
                print(f"   Reconstructed: {recon}")
                return False

            # Check attributes
            if isinstance(orig, NoteOnEvent) and orig.pitch != recon.pitch:
                print(f"❌ FAIL at position {i}: pitch mismatch ({orig.pitch} vs {recon.pitch})")
                return False
            elif isinstance(orig, TabEvent) and (orig.string != recon.string or orig.fret != recon.fret):
                print(f"❌ FAIL at position {i}: tab mismatch")
                return False

        print("✅ PASS: Roundtrip successful")
        return True


def main():
    print("\n" + "="*80)
    print("EVENT-TO-TOKEN CONVERSION TEST SUITE")
    print("="*80 + "\n")

    results = []

    results.append(("Roundtrip Conversion", test_roundtrip_conversion()))
    results.append(("Manual Event Creation", test_event_creation()))
    test_out_of_range_values()  # Informational test
    results.append(("Batch Conversion", test_batch_conversion()))

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
