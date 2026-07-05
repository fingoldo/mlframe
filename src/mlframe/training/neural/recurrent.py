from __future__ import annotations

"""
Recurrent neural network models for sequence classification and regression.

This module provides PyTorch Lightning-based recurrent models with support for:
- FEATURES_ONLY: Pure MLP on tabular features
- SEQUENCE_ONLY: RNN (LSTM/GRU/RNN/Transformer) on raw time series
- HYBRID: Both sequence and tabular features combined

Classes:
    RNNType: Supported sequence encoder architectures
    InputMode: Input data modes
    RecurrentConfig: Configuration dataclass
    RecurrentDataset: PyTorch Dataset for sequences
    RecurrentDataModule: Lightning DataModule for sequences
    RecurrentTorchModel: Lightning module for training
    RecurrentClassifierWrapper: Sklearn-compatible classifier wrapper
    RecurrentRegressorWrapper: Sklearn-compatible regressor wrapper
    AttentionPooling: Attention mechanism for RNN outputs
    PositionalEncoding: Sinusoidal positional encoding for Transformer
    TransformerSequenceEncoder: Transformer encoder for sequences
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import TYPE_CHECKING

try:
    import xxhash as _xxhash  # noqa: F401  module-top for hot cache-key path
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

if TYPE_CHECKING:
    pass


__all__ = [
    # Enums
    "RNNType",
    "InputMode",
    # Configuration
    "RecurrentConfig",
    # Dataset/DataModule
    "RecurrentDataset",
    "RecurrentDataModule",
    "recurrent_collate_fn",
    # Model Components
    "AttentionPooling",
    "PositionalEncoding",
    "TransformerSequenceEncoder",
    "MLPHead",
    # Lightning Module
    "RecurrentTorchModel",
    # Sklearn Wrappers
    "RecurrentClassifierWrapper",
    "RecurrentRegressorWrapper",
    # Utilities
    "extract_sequences",
    "extract_sequences_chunked",
]


# ----------------------------------------------------------------------------------------------------------------------------
# Late re-exports + shared utility
# ----------------------------------------------------------------------------------------------------------------------------


from .base import _ensure_numpy  # noqa: E402,F401  shared with _recurrent_data
from ._recurrent_config import RNNType, InputMode, RecurrentConfig  # noqa: E402,F401
from ._recurrent_data import RecurrentDataset, recurrent_collate_fn, RecurrentDataModule  # noqa: E402,F401
from ._recurrent_arch import AttentionPooling, PositionalEncoding, TransformerSequenceEncoder, MLPHead  # noqa: E402,F401

# ----------------------------------------------------------------------------------------------------------------------------
# Lightning Module
# ----------------------------------------------------------------------------------------------------------------------------


# Wave 103 (2026-05-21): RecurrentTorchModel class (~390 lines) moved to
# sibling file _recurrent_torch_model.py to drop this file below the 1k
# monolith threshold. Re-exported below so existing callers
# (`from mlframe.training.neural.recurrent import RecurrentTorchModel`)
# keep working.
from ._recurrent_torch_model import RecurrentTorchModel  # noqa: F401, E402

# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn Wrappers + EarlyStopping monitor-direction helper -- carved to
# ``recurrent_dataset_helpers.py`` to keep this facade under the 1000-LOC
# budget. Re-exported below so existing
# ``from .recurrent import RecurrentClassifierWrapper`` (and _RecurrentWrapperBase /
# _monitor_mode / _DEFAULT_SEQ_INPUT_SIZE) callers keep working.
# ----------------------------------------------------------------------------------------------------------------------------

from .recurrent_dataset_helpers import (  # noqa: E402, F401
    _DEFAULT_SEQ_INPUT_SIZE,
    _MONITOR_MIN_KEYS,
    _MONITOR_MAX_KEYS,
    _monitor_mode,
    _RecurrentWrapperBase,
    RecurrentClassifierWrapper,
    RecurrentRegressorWrapper,
)
# ----------------------------------------------------------------------------------------------------------------------------
# Utility Functions -- carved to ``_recurrent_sequences.py`` to keep this
# facade under the 1000-LOC budget. Re-exported below so existing
# ``from .recurrent import extract_sequences`` callers keep working.
# ----------------------------------------------------------------------------------------------------------------------------

from ._recurrent_sequences import (  # noqa: E402, F401
    extract_sequences,
    extract_sequences_chunked,
)
