"""
Neural network models for mlframe training.

This module provides PyTorch Lightning-based neural network models:
- Base infrastructure (estimators, callbacks, utilities)
- Flat/tabular models (MLP)
- Recurrent/sequence models (LSTM, GRU, RNN, Transformer)

All classes maintain sklearn-compatible API through wrappers.
"""

# Base infrastructure
from .base import (
    # Estimators
    PytorchLightningEstimator,
    PytorchLightningRegressor,
    PytorchLightningClassifier,
    # Callbacks
    NetworkGraphLoggingCallback,
    AggregatingValidationCallback,
    BestEpochModelCheckpoint,
    PeriodicLearningRateFinder,
    # Utilities
    MetricSpec,
    to_tensor_any,
    to_numpy_safe,
    custom_collate_fn,
)

# Data handling
from .data import (
    TorchDataset,
    TorchDataModule,
)

# Flat/tabular models (MLP)
from .flat import (
    MLPTorchModel,
    MLPNeuronsByLayerArchitecture,
    generate_mlp,
    get_valid_num_groups,
)

# Recurrent/sequence models
from .recurrent import (
    RNNType,
    InputMode,
    RecurrentConfig,
    RecurrentDataset,
    RecurrentDataModule,
    RecurrentTorchModel,
    RecurrentClassifierWrapper,
    RecurrentRegressorWrapper,
    AttentionPooling,
    PositionalEncoding,
    TransformerSequenceEncoder,
    MLPHead,
    recurrent_collate_fn,
    extract_sequences,
    extract_sequences_chunked,
)

__all__ = [
    # Base - Estimators
    "PytorchLightningEstimator",
    "PytorchLightningRegressor",
    "PytorchLightningClassifier",
    # Base - Callbacks
    "NetworkGraphLoggingCallback",
    "AggregatingValidationCallback",
    "BestEpochModelCheckpoint",
    "PeriodicLearningRateFinder",
    # Base - Utilities
    "MetricSpec",
    "to_tensor_any",
    "to_numpy_safe",
    "custom_collate_fn",
    # Data
    "TorchDataset",
    "TorchDataModule",
    # Flat
    "MLPTorchModel",
    "MLPNeuronsByLayerArchitecture",
    "generate_mlp",
    "get_valid_num_groups",
    # Recurrent
    "RNNType",
    "InputMode",
    "RecurrentConfig",
    "RecurrentDataset",
    "RecurrentDataModule",
    "RecurrentTorchModel",
    "RecurrentClassifierWrapper",
    "RecurrentRegressorWrapper",
    "AttentionPooling",
    "PositionalEncoding",
    "TransformerSequenceEncoder",
    "MLPHead",
    "recurrent_collate_fn",
    "extract_sequences",
    "extract_sequences_chunked",
]
