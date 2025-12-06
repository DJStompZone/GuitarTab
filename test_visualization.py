"""
Test script for visualization functions.

This script demonstrates and tests the tablature and note notation rendering
for both DadaGP raw tokens and parsed events.
"""

import sys
from pathlib import Path
import random

from src.dadagp_parser import parse_dadagp_file, parse_dadagp_file_to_events
from src.visualization import (
    render_dadagp_tokens_as_tablature,
    render_dadagp_tokens_as_notes,
    render_as_tablature,
    render_as_notes
)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def test_single_file(file_path, max_bars=3):
    """Test visualization on a single file."""
    print_section(f"Testing: {Path(file_path).name}")

    # Parse file both ways
    raw_tokens = parse_dadagp_file(str(file_path))
    input_events, output_events, _ = parse_dadagp_file_to_events(str(file_path))

    print(f"Parsed {len(raw_tokens):,} raw tokens")
    print(f"Generated {len(input_events):,} input events")
    print(f"Generated {len(output_events):,} output events")

    # Test 1: Tablature from DadaGP tokens
    print_section("1. Tablature from DadaGP Tokens")
    tab_from_tokens = render_dadagp_tokens_as_tablature(
        raw_tokens,
        max_bars=max_bars,
        bars_per_row=1,
        chars_per_beat=8
    )
    print(tab_from_tokens)

    # Test 2: Tablature from parsed events
    print_section("2. Tablature from Parsed Events")
    tab_from_events = render_as_tablature(
        output_events,
        max_bars=max_bars,
        bars_per_row=1,
        chars_per_beat=8
    )
    print(tab_from_events)

    # Test 3: Notes from DadaGP tokens
    print_section("3. Note Notation from DadaGP Tokens")
    notes_from_tokens = render_dadagp_tokens_as_notes(
        raw_tokens,
        max_bars=max_bars,
        bars_per_row=1,
        chars_per_beat=8
    )
    print(notes_from_tokens)

    # Test 4: Notes from parsed events
    print_section("4. Note Notation from Parsed Events")
    notes_from_events = render_as_notes(
        input_events,
        max_bars=max_bars,
        bars_per_row=1,
        chars_per_beat=8
    )
    print(notes_from_events)

    # Test 5: Multiple bars per row (tablature)
    print_section("5. Tablature with 3 Bars Per Row")
    tab_multibar = render_as_tablature(
        output_events,
        max_bars=6,
        bars_per_row=3,
        chars_per_beat=8
    )
    print(tab_multibar)

    # Test 6: Compact view
    print_section("6. Compact View (6 bars per row, 6 chars/beat)")
    tab_compact = render_as_tablature(
        output_events,
        max_bars=12,
        bars_per_row=6,
        chars_per_beat=6
    )
    print(tab_compact)

    # Test 7: Expanded view
    print_section("7. Expanded View (2 bars per row, 12 chars/beat)")
    tab_expanded = render_as_tablature(
        output_events,
        max_bars=2,
        bars_per_row=2,
        chars_per_beat=12
    )
    print(tab_expanded)

    # Test 8: Side-by-side comparison
    print_section("8. Side-by-Side Comparison (First 2 bars)")
    print("TABLATURE COMPARISON:")
    print("-" * 80)
    print("\nFrom DadaGP Tokens:")
    print(render_dadagp_tokens_as_tablature(raw_tokens, max_bars=2, bars_per_row=2))
    print("\nFrom Parsed Events:")
    print(render_as_tablature(output_events, max_bars=2, bars_per_row=2))

    print("\n" + "-" * 80)
    print("NOTE NOTATION COMPARISON:")
    print("-" * 80)
    print("\nFrom DadaGP Tokens:")
    print(render_dadagp_tokens_as_notes(raw_tokens, max_bars=2, bars_per_row=2))
    print("\nFrom Parsed Events:")
    print(render_as_notes(input_events, max_bars=2, bars_per_row=2))


def test_multiple_files(num_files=3, max_bars=2):
    """Test visualization on multiple random files."""
    print_section(f"Testing on {num_files} Random Files")

    # Find token files
    dadagp_dir = Path('DadaGP-v1.1')
    if not dadagp_dir.exists():
        print(f"Error: {dadagp_dir} not found")
        return

    token_files = list(dadagp_dir.rglob('*.tokens.txt'))
    if not token_files:
        print("No token files found")
        return

    print(f"Found {len(token_files):,} token files\n")

    # Sample random files
    sample_files = random.sample(token_files, min(num_files, len(token_files)))

    for i, file in enumerate(sample_files, 1):
        print(f"\n[{i}/{num_files}] {file.relative_to(dadagp_dir)}")
        print("-" * 80)

        try:
            _, output_events, _ = parse_dadagp_file_to_events(str(file))
            tab = render_as_tablature(output_events, max_bars=max_bars, bars_per_row=max_bars)
            print(tab)
        except Exception as e:
            print(f"Error: {e}")


def test_format_validation():
    """Test that output format is correct."""
    print_section("Format Validation Tests")

    # Find a sample file
    dadagp_dir = Path('DadaGP-v1.1')
    token_files = list(dadagp_dir.rglob('*.tokens.txt'))

    if not token_files:
        print("No token files found")
        return

    sample_file = token_files[0]
    print(f"Using: {sample_file.name}\n")

    raw_tokens = parse_dadagp_file(str(sample_file))
    input_events, output_events, _ = parse_dadagp_file_to_events(str(sample_file))

    # Test 1: Check output is string
    result = render_as_tablature(output_events, max_bars=2)
    assert isinstance(result, str), "Output should be a string"
    print("✓ Output is a string")

    # Test 2: Check contains string lines (multiples of 6)
    lines = result.split('\n')
    string_lines = [l for l in lines if l.startswith(('e|', 'B|', 'G|', 'D|', 'A|', 'E|'))]
    assert len(string_lines) > 0 and len(string_lines) % 6 == 0, f"Expected multiple of 6 string lines, got {len(string_lines)}"
    print(f"✓ Contains {len(string_lines)} string lines ({len(string_lines)//6} rows)")

    # Test 3: Check bar markers present
    bar_marker_lines = [l for l in lines if 'Bar' in l]
    assert len(bar_marker_lines) > 0, "Expected bar markers"
    print("✓ Contains bar markers")

    # Test 4: Check all strings have same length content
    string_contents = [l.split('|')[1] for l in string_lines]
    lengths = [len(s) for s in string_contents]
    assert len(set(lengths)) == 1, f"All strings should have same length, got {lengths}"
    print(f"✓ All string lines have equal length ({lengths[0]} chars)")

    # Test 5: Check bars_per_row parameter works
    result_1bar = render_as_tablature(output_events, max_bars=3, bars_per_row=1)
    result_3bar = render_as_tablature(output_events, max_bars=3, bars_per_row=3)

    # With 3 bars per row, the line should be longer (but still 6 lines)
    lines_1bar = result_1bar.split('\n')
    lines_3bar = result_3bar.split('\n')

    string_lines_1bar = [l for l in lines_1bar if l.startswith('e|')]
    string_lines_3bar = [l for l in lines_3bar if l.startswith('e|')]

    # With bars_per_row=3, should have 1 row with 1 'e|' line
    # With bars_per_row=1, should have 3 rows with 3 'e|' lines (one per row)
    assert len(string_lines_1bar) == 3, f"Expected 3 'e|' lines (3 rows × 1 bar each), got {len(string_lines_1bar)}"
    assert len(string_lines_3bar) == 1, f"Expected 1 'e|' line (1 row × 3 bars), got {len(string_lines_3bar)}"

    # Check that the 3-bar-per-row line is longer (has bar separators)
    if string_lines_3bar and string_lines_1bar:
        len_3bar = len(string_lines_3bar[0])
        len_1bar = len(string_lines_1bar[0])
        assert len_3bar > len_1bar * 2, f"Expected 3-bar line to be much longer, got {len_3bar} vs {len_1bar}"

    print("✓ bars_per_row parameter works correctly")

    # Test 6: Check note notation format
    note_result = render_as_notes(input_events, max_bars=2)
    note_lines = note_result.split('\n')
    note_content_lines = [l for l in note_lines if l.strip().startswith('|')]
    assert len(note_content_lines) > 0, "Expected note content lines"
    print("✓ Note notation format is valid")

    print("\n" + "=" * 80)
    print("All validation tests passed!")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VISUALIZATION TEST SUITE")
    print("=" * 80)

    # Find DadaGP directory
    dadagp_dir = Path('DadaGP-v1.1')
    if not dadagp_dir.exists():
        print(f"\nError: {dadagp_dir} not found")
        print("Please ensure DadaGP-v1.1 is in the current directory")
        return 1

    # Find token files
    token_files = list(dadagp_dir.rglob('*.tokens.txt'))
    if not token_files:
        print(f"\nError: No .tokens.txt files found in {dadagp_dir}")
        return 1

    print(f"\nFound {len(token_files):,} token files")

    # Select a sample file for detailed testing
    sample_file = random.choice(token_files)

    # Run tests
    try:
        # Test 1: Single file with all visualization types
        test_single_file(sample_file, max_bars=3)

        # Test 2: Multiple files quick view
        test_multiple_files(num_files=3, max_bars=2)

        # Test 3: Format validation
        test_format_validation()

        print_section("All Tests Completed Successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
