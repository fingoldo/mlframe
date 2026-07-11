"""
Neural network models for mlframe training.

This module provides PyTorch Lightning-based neural network models:
- Base infrastructure (estimators, callbacks, utilities)
- Flat/tabular models (MLP)
- Recurrent/sequence models (LSTM, GRU, RNN, Transformer)

All classes maintain sklearn-compatible API through wrappers.
"""

from __future__ import annotations

# Base infrastructure
from .base import (
    # Estimators
    PytorchLightningEstimator,
    PytorchLightningRegressor,
    PytorchLightningClassifier,
    # Callbacks
    NetworkGraphLoggingCallback,
    AggregatingValidationCallback,
    MonotonicDeclineStopCallback,
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

# Fixed-mask structured-sparsity layer (standalone, not part of the Lightning estimator infra)
from .fixed_sparse_linear import FixedSparseLinear

# Correlation-ordered 1D-CNN over tabular features (standalone, not part of the Lightning estimator infra)
from .tabular_1dcnn import Tabular1DCNNClassifier, Tabular1DCNNRegressor, correlation_order_features

# Trunk-into-every-block residual MLP (standalone, not part of the Lightning estimator infra)
from .trunk_residual_mlp import TrunkResidualMLPRegressor

# Field-grouped sub-MLP encoders (standalone, not part of the Lightning estimator infra; see its own
# module docstring for an honest-negative note: the hypothesized generalization win did not reproduce).
from .field_grouped_mlp import FieldGroupedMLPRegressor

# Group-aware causal attention mask (simultaneous-events-within-a-group support for TransformerSequenceEncoder)
from .group_causal_attention_mask import group_causal_attention_mask

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
    "MonotonicDeclineStopCallback",
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
    # Fixed-mask structured-sparsity layer
    "FixedSparseLinear",
    # Correlation-ordered 1D-CNN over tabular features
    "Tabular1DCNNRegressor",
    "Tabular1DCNNClassifier",
    "correlation_order_features",
    # Trunk-into-every-block residual MLP
    "TrunkResidualMLPRegressor",
    # Field-grouped sub-MLP encoders (honest negative -- see module docstring)
    "FieldGroupedMLPRegressor",
    # Group-aware causal attention mask
    "group_causal_attention_mask",
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
