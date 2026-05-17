"""Recurrent model configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class RNNType(str, Enum):
    """Supported sequence encoder architectures."""

    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"
    TRANSFORMER = "transformer"


class InputMode(str, Enum):
    """Input data modes for the classifier/regressor."""

    SEQUENCE_ONLY = "sequence"  # raw time series only
    FEATURES_ONLY = "features"  # tabular features only
    HYBRID = "hybrid"  # both sequence + features


# ----------------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------------


@dataclass
class RecurrentConfig:
    """
    Configuration for recurrent models (classification and regression).

    Attributes:
        input_mode: which inputs to use (SEQUENCE_ONLY, FEATURES_ONLY, HYBRID)
        rnn_type: RNN architecture (LSTM, GRU, RNN, TRANSFORMER); ignored if FEATURES_ONLY
        hidden_size: RNN hidden state size
        num_layers: number of RNN layers
        bidirectional: whether to use bidirectional RNN
        use_attention: whether to use attention pooling (vs last hidden)
        n_heads: number of attention heads (Transformer only)
        dim_feedforward: feedforward dimension in Transformer
        mlp_hidden_sizes: tuple of MLP hidden layer sizes
        dropout: dropout probability
        learning_rate: learning rate for optimizer
        weight_decay: L2 regularization weight
        batch_size: training batch size
        max_epochs: maximum training epochs
        early_stopping_patience: epochs to wait before early stopping
        gradient_clip_val: gradient clipping value
        precision: training precision ("16-mixed" for AMP, "32-true" for full)
        early_stopping_monitor: metric to monitor ("val_loss" or "val_auprc")
        use_stratified_sampler: use weighted sampling for imbalanced data
        accelerator: device to use ("auto", "gpu", "cpu")
        num_workers: DataLoader workers
        scale_features: whether to StandardScaler auxiliary features
        num_classes: number of output classes (for classification)
    """

    # Input mode
    input_mode: InputMode = InputMode.HYBRID

    # Sequence encoder architecture (ignored if input_mode=FEATURES_ONLY)
    rnn_type: RNNType = RNNType.LSTM
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True  # for RNN/LSTM/GRU only
    use_attention: bool = True  # for RNN/LSTM/GRU only

    # Transformer-specific (only used if rnn_type=TRANSFORMER)
    n_heads: int = 4
    dim_feedforward: int = 256

    # MLP head
    mlp_hidden_sizes: tuple[int, ...] = (256, 128)
    dropout: float = 0.3

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 30
    gradient_clip_val: float = 1.0
    precision: str = "16-mixed"
    early_stopping_monitor: str = "val_loss"
    use_stratified_sampler: bool = True

    # Hardware
    accelerator: str = "auto"
    num_workers: int = 0  # 0 for Windows compatibility

    # Preprocessing
    scale_features: bool = True

    # Sequence preprocessing strategy (audit 2026-05-17, finding C9).
    #
    # ``"none"`` (default, post-audit): pass sequences through unchanged.
    #     Magnitude is preserved -- correct when the absolute scale of
    #     channels carries discriminative information (the common case
    #     for generic ML usage).
    # ``"per_sequence_zscore"``: z-score each channel within each
    #     sequence independently. Correct when each sequence is a
    #     standalone time-series whose absolute scale is meaningless
    #     (e.g. one stock's price series) -- BUT destroys magnitude
    #     information across sequences. Pre-audit this was the implicit
    #     hardcoded behaviour, which was wrong for the typical
    #     classification / regression case.
    # ``"astronomy_mjd_delta"``: legacy compat for astronomy datasets
    #     where column 0 is an MJD timestamp delta-encoded and scaled
    #     by 10 (an in-house dataset's convention). Provided so the
    #     existing astronomy pipeline doesn't break after the default
    #     flip; new users should not pick this.
    sequence_preprocessing: str = "none"

    # Output
    num_classes: int = 2  # for classification; ignored for regression

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
