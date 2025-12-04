# GuitarTab Exploration Notebooks

This directory contains Jupyter notebooks for exploring and analyzing the DadaGP dataset using token files.

## Notebooks

### 1. `01_dataset_overview.ipynb`
**Purpose**: Get an overview of the DadaGP dataset without loading all files

**Contents**:
- Count GP files and token files in DadaGP-v1.1
- Sample random files for analysis
- Parse raw tokens (simple string parsing, no dependencies)
- Analyze token distributions (notes, waits, measures, effects)
- Aggregate statistics across multiple files

**Key features**:
- ✅ No dependencies on `src/` modules (except final notebooks)
- ✅ Simple string parsing of token files
- ✅ Fast - doesn't load all files, samples randomly
- ✅ Shows token patterns and distributions

---

### 2. `02_token_analysis.ipynb`
**Purpose**: Analyze tokens using the `dadagp_parser` module

**Contents**:
- Parse DadaGP token files into events (NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB)
- Compare input vs output sequences
- Analyze token distributions:
  - Pitch ranges and frequencies
  - Time shift patterns
  - TAB token distributions (strings, frets, combinations)
- **Sequence alignment verification** - checks if input and output are properly aligned
- Multi-file aggregate statistics

**Dependencies**: `src.dadagp_parser`

**Key insights**:
- Shows how DadaGP tokens transform into model-ready events
- Verifies the paired sequence approach (input + TAB tokens = output)
- Provides detailed statistics for understanding the data

---

### 3. `03_token_visualization.ipynb`
**Purpose**: Visualize tokens as human-readable scores

**Contents**:
- **Six-string tablature notation** - shows fret numbers on guitar strings
- **Note name notation** - shows note names (C4, E3, etc.) with rhythm
- Both formats include:
  - Proper bar lines (measures)
  - Beat markers
  - Correct timing/rhythm
- Interactive functions to visualize any file
- Batch visualization for multiple files
- Export visualizations to text files

**Dependencies**: `src.dadagp_parser`, `src.visualization`

**Example output**:

```
Guitar Tablature (Standard Tuning: EADGBE)
================================================================================

Bar 1:
e|0-----3--5-----------7--------|
B|----1-------3-----5-----------|
G|--------0-------2-------4-----|
D|------------------------------|
A|------------------------------|
E|------------------------------|
 |   1       2       3       4  |
```

---

### 4. `04_consistency_check.ipynb`
**Purpose**: Verify consistency between raw DadaGP tokens and parsed events

**Contents**:
- Parse files with both methods (raw tokens + events)
- Verify note counts match (raw → NOTE_ON → TAB)
- Verify timing matches (wait tokens → TIME_SHIFT events)
- Verify note details (string, fret, pitch calculations)
- Check sequence alignment (output = input + TAB tokens)
- Verify metadata extraction (tempo, downtune)
- Multi-file consistency check across samples
- Summary report with pass/fail status

**Dependencies**: `src.dadagp_parser`

**Key checks**:
1. ✓ Note count consistency
2. ✓ Timing consistency (total ticks)
3. ✓ Pitch calculation (string + fret → MIDI pitch)
4. ✓ Sequence alignment (input events match output events minus TAB)
5. ✓ Metadata preservation

**Why this matters**:
This notebook ensures that the transformation from DadaGP tokens to model-ready events is **consistent and lossless**. It verifies that the paired sequence approach (v3 encoding from the paper) is correctly implemented.

---

## Getting Started

### 1. Activate your environment
```bash
conda activate MuiscFinal
```

### 2. Install Jupyter (if not already installed)
```bash
pip install jupyter
```

### 3. Launch Jupyter
```bash
cd notebooks
jupyter notebook
```

### 4. Run notebooks in order
- **01_dataset_overview.ipynb** - Start here to understand the data
- **02_token_analysis.ipynb** - Parse and analyze token events
- **03_token_visualization.ipynb** - Visualize as tablature and notes
- **04_consistency_check.ipynb** - Verify parsing consistency

---

## Data Flow

```
DadaGP Token Files (.tokens.txt)
         ↓
   Raw tokens: leads:note:s1:f0, wait:480, etc.
         ↓
   dadagp_parser module
         ↓
   Events: NOTE_ON, NOTE_OFF, TIME_SHIFT, TAB
         ↓
   visualization module
         ↓
   Human-readable tablature / note notation
```

---

## Dependencies

| Notebook | Dependencies |
|----------|--------------|
| 01_dataset_overview.ipynb | ✅ None (stdlib only) |
| 02_token_analysis.ipynb | `src.dadagp_parser` |
| 03_token_visualization.ipynb | `src.dadagp_parser`, `src.visualization` |
| 04_consistency_check.ipynb | `src.dadagp_parser` |

---

## Key Findings

From running these notebooks:

1. **Dataset size**: ~26,000 DadaGP files with token files
2. **Token format**: DadaGP uses human-readable tokens (leads:note:s1:f0)
3. **Event transformation**: Tokens → NOTE_ON/OFF/TIME_SHIFT/TAB events
4. **Sequence alignment**: Output = Input + TAB tokens (properly aligned)
5. **String usage**: Lower strings (E, A, D) used more than higher strings
6. **Fret distribution**: Open strings (fret 0) and low frets (1-5) most common
7. **Pitch range**: Typical guitar range (E2-E6, MIDI 40-88)

---

## Visualization Examples

### Tablature View
Shows exactly which fret to play on which string:
```
e|0--3--5--7--|
B|1--3--5-----|
G|0--2--4-----|
```

### Note View
Shows the actual pitches being played:
```
Bar 1:
  ||-------|-------|-------|-------|
  Notes:
    @  0: E4 (dur: 8)
    @  8: G4 (dur: 8)
```

---

## Visualization Module

The visualization functions are available in `../src/visualization.py`:

```python
from src.visualization import render_as_tablature, render_as_notes
from src.dadagp_parser import parse_dadagp_file_to_events

# Parse file
input_events, output_events = parse_dadagp_file_to_events('file.tokens.txt')

# Render as tablature (6 strings)
print(render_as_tablature(output_events, max_bars=4))

# Render as notes (pitch names)
print(render_as_notes(input_events, max_bars=4))
```

---

## Next Steps

After exploring the data with these notebooks:

1. ✅ **Data format** is clear: paired sequences with TAB tokens
2. ✅ **Tokenization** is working: dadagp_parser produces aligned events
3. ✅ **Visualization** confirms tokens represent real music correctly

**Remaining work**:
- Implement T5 model architecture (encoder-decoder)
- Create training script using the dadagp_parser
- Add data augmentation (capo/tuning variations)
- Implement evaluation metrics (pitch accuracy, tab accuracy, difficulty score)
- Add post-processing for predictions

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Make sure you're running from the `notebooks/` directory and the parent dir is in the path (notebooks do this automatically)

**Issue**: No token files found
**Solution**: Check that `../DadaGP-v1.1` exists and contains `.tokens.txt` files

**Issue**: Visualization looks wrong
**Solution**: Check the `ticks_per_beat` parameter (default 480) - some files may use different timing

---

## Contributing

To add new visualizations or analysis:
1. Copy an existing notebook as a template
2. Import from `src.dadagp_parser` and `src.visualization`
3. Follow the same structure (markdown explanations + code cells)
4. Update this README with your new notebook
