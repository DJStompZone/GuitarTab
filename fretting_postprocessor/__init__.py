"""
Fretting Post-Processor
========================

Post-processing module for Fretting-Transformer model outputs.
Implements overlap correction and neighbor search algorithms to achieve 100% pitch accuracy.

Based on: Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription
(arXiv:2506.14223v1)
"""

__version__ = "1.0.0"
__author__ = "Based on Fretting-Transformer paper"

# Core configuration
from .config import (
    GuitarConfig,
    STANDARD_TUNING,
    DROP_D_TUNING,
    HALF_STEP_DOWN,
    FULL_STEP_DOWN,
)

# Data types
from .datatypes import (
    TokenType,
    Token,
    Note,
)

# Sequence container
from .sequence import NoteSequence

# Parser and Serializer
from .parser import TokenParser
from .serializer import TokenSerializer

# Validator
from .validator import PitchValidator, validate_sequence, calculate_pitch_accuracy

# Processor
from .processor import PostProcessor

# Evaluator
from .evaluator import PostProcessingEvaluator

# Main API
try:
    from .api import (
        FrettingPostProcessor,
        process_tokens_quick,
        evaluate_quick
    )
    _HAS_API = True
except ImportError:
    _HAS_API = False
    FrettingPostProcessor = None
    process_tokens_quick = None
    evaluate_quick = None

# Utils (JAMS/MIDI integration)
try:
    from .utils import (
        jams_to_tokens,
        tokens_to_midi,
        process_jams_file,
        batch_process_jams_directory
    )
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False
    jams_to_tokens = None
    tokens_to_midi = None
    process_jams_file = None
    batch_process_jams_directory = None

__all__ = [
    # Configuration
    "GuitarConfig",
    "STANDARD_TUNING",
    "DROP_D_TUNING",
    "HALF_STEP_DOWN",
    "FULL_STEP_DOWN",

    # Data types
    "TokenType",
    "Token",
    "Note",

    # Sequence
    "NoteSequence",

    # Parser and Serializer
    "TokenParser",
    "TokenSerializer",

    # Validator
    "PitchValidator",
    "validate_sequence",
    "calculate_pitch_accuracy",

    # Processor
    "PostProcessor",

    # Evaluator
    "PostProcessingEvaluator",
]

# Add API to __all__ if available
if _HAS_API:
    __all__.extend([
        "FrettingPostProcessor",
        "process_tokens_quick",
        "evaluate_quick"
    ])

# Add Utils to __all__ if available
if _HAS_UTILS:
    __all__.extend([
        "jams_to_tokens",
        "tokens_to_midi",
        "process_jams_file",
        "batch_process_jams_directory"
    ])
