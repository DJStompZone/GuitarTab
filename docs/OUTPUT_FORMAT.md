# Output Format Configuration

This document describes the `output_format` configuration option for controlling the sequence format during training.

## Overview

The `output_format` parameter controls the structure of both **input (encoder)** and **output (decoder)** sequences in the Seq2Seq model.

| Format | Input Sequence | Output Sequence |
|--------|----------------|-----------------|
| **v1** | NOTE_ON, NOTE_OFF, TIME_SHIFT | NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT |
| **v2** | NOTE_ON, NOTE_OFF, TIME_SHIFT | TAB, TIME_SHIFT |
| **v3** | NOTE_ON, TIME_SHIFT | NOTE_ON, TAB, TIME_SHIFT |

## Format Details

### v1: Full Format (Default)

```
Input:  [NOTE_ON_64, NOTE_OFF_64, TIME_SHIFT_240, ...]
Output: [NOTE_ON_64, TAB_3_5, NOTE_OFF_64, TIME_SHIFT_240, ...]
```

- **Input vocabulary size**: 760 tokens
- **Output vocabulary size**: 886 tokens
- **Use case**: When you want the model to learn the full relationship between pitch, duration, and fingering

### v2: Simplified Output

```
Input:  [NOTE_ON_64, NOTE_OFF_64, TIME_SHIFT_240, ...]
Output: [TAB_3_5, TIME_SHIFT_240, ...]
```

- **Input vocabulary size**: 760 tokens
- **Output vocabulary size**: 630 tokens (29% reduction from v1)
- **Output sequence**: Contains only TAB and TIME_SHIFT tokens (50-60% shorter)
- **Use case**: When you want a more compact representation focused purely on fingering decisions

### v3: No NOTE_OFF

```
Input:  [NOTE_ON_64, TIME_SHIFT_240, ...]
Output: [NOTE_ON_64, TAB_3_5, TIME_SHIFT_240, ...]
```

- **Input vocabulary size**: 632 tokens (17% reduction from v1)
- **Output vocabulary size**: 758 tokens (14% reduction from v1)
- **Use case**: When you want to simplify the model by removing note-off events (assuming notes end at time shifts)

## Usage

### Via Command Line (Hydra)

```bash
# Use v1 format (default)
python train.py

# Use v2 format
python train.py data.output_format=v2

# Use v3 format
python train.py data.output_format=v3
```

### Via Configuration File

Edit `configs/data/dadagp.yaml`:

```yaml
# Output format for input/output sequences
# v1: input = NOTE_ON, NOTE_OFF, TIME_SHIFT; output = NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT
# v2: input = NOTE_ON, NOTE_OFF, TIME_SHIFT; output = TAB, TIME_SHIFT
# v3: input = NOTE_ON, TIME_SHIFT; output = NOTE_ON, TAB, TIME_SHIFT (no NOTE_OFF)
output_format: "v3"
```

## Console Output

When training starts, the console will display the active format:

```
Building vocabularies...
Output format: v1 (NOTE_ON, TAB, NOTE_OFF, TIME_SHIFT)
Input vocab size: 760
Output vocab size: 886
```

or

```
Building vocabularies...
Output format: v2 (TAB, TIME_SHIFT)
Input vocab size: 760
Output vocab size: 630
```

or

```
Building vocabularies...
Output format: v3 (NOTE_ON, TAB, TIME_SHIFT)
Input vocab size: 632
Output vocab size: 758
```

## Comparison

| Metric | v1 | v2 | v3 |
|--------|-----|-----|-----|
| Input Vocab Size | 760 | 760 | 632 |
| Output Vocab Size | 886 | 630 | 758 |
| Input Sequence Length (single note) | 3 tokens | 3 tokens | 2 tokens |
| Output Sequence Length (single note) | 4 tokens | 2 tokens | 3 tokens |
| Has NOTE_OFF in input | Yes | Yes | No |
| Has NOTE_OFF in output | Yes | No | No |
| Has NOTE_ON in output | Yes | No | Yes |

## Technical Details

The format is implemented in the following files:

- `configs/data/dadagp.yaml` - Configuration definition
- `src/dadagp_parser.py` - Event generation logic (`dadagp_to_events()`)
- `src/tab_dataset.py` - Vocabulary building (`build_vocabulary()`)
- `src/dataloader.py` - Dataset creation (`create_dataset()`)
- `train.py` - Configuration passing to dataset

## Notes

- v1 and v2 use the **same input sequence** (NOTE_ON, NOTE_OFF, TIME_SHIFT)
- v3 removes NOTE_OFF from both input and output
- The v2 format removes pitch information from the output (no NOTE_ON/OFF)
- The v3 format keeps pitch information (NOTE_ON) but removes duration information (NOTE_OFF)
- Models trained with different formats are **not compatible** - ensure consistent format between training and inference
