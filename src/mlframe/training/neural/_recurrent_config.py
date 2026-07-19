"""Recurrent model configuration types."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def _default_num_workers() -> int:
    """F-71 (2026-05-31): OS-aware DataLoader workers default.

    Windows uses spawn semantics for DataLoader workers, which costs
    ~150-300 ms per worker spawn each epoch. Combined with our F-46
    persistent_workers + F-51 share_memory_ + F-52 prefetch_factor the
    spawn cost amortizes after the first epoch, but a user running
    smoke tests / short fits on Windows would still feel it. Stay at 0
    on Windows; default to 2 on Linux + macOS where fork semantics make
    workers nearly free and the per-item ``torch.as_tensor(
    sequences[idx])`` CPU cost in RecurrentDataset.__getitem__ benefits
    from parallelism (1.3-1.7x train throughput per Agent B audit P1).
    """
    return 0 if os.name == "nt" else 2


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
    # F-71 (2026-05-31): num_workers default is now OS-aware via
    # _default_num_workers() -- 2 on Linux/macOS (fork is cheap), 0 on
    # Windows (spawn semantics costs ~150-300 ms per worker per epoch).
    # Combined with F-46 persistent_workers, F-51 share_memory_, and F-52
    # prefetch_factor=4, Linux users now get 1.3-1.7x train throughput
    # by default without further config. Windows users still need to
    # explicitly opt in via RecurrentConfig(num_workers=2) to amortize
    # the spawn cost across enough epochs.
    num_workers: int = field(default_factory=_default_num_workers)
    # Overrides the auto-detected ``torch.cuda.is_available() and accelerator in (...)`` default
    # (F-47) when set. Some driver/CUDA-toolkit combos crash during CachingHostAllocator/CUDAEvent
    # teardown when pinned memory is in play (a fatal std::terminate from a tensor destructor, not
    # a catchable Python exception) -- GPU detection alone can't distinguish a healthy pinned-
    # memory path from a broken one, so an explicit ``pin_memory=False`` is the only way out.
    pin_memory: Optional[bool] = None
    # F-52: prefetch_factor only used when num_workers > 0. Default of 2 is
    # fine for in-RAM eager tensors; for recurrent + variable-len padding
    # (pad-on-the-fly via recurrent_collate_fn), 4 smooths GPU starvation
    # during long-pad batches. Lift: 5-10% on heterogeneous seq-len data.
    prefetch_factor: int = 4

    # Preprocessing
    scale_features: bool = True

    # Sequence preprocessing strategy.
    #
    # ``"none"`` (default): pass sequences through unchanged. Magnitude
    #     is preserved, correct when the absolute scale of channels
    #     carries discriminative information (the common case for
    #     generic ML usage).
    # ``"per_sequence_zscore"``: z-score each channel within each
    #     sequence independently. Correct when each sequence is a
    #     standalone time-series whose absolute scale is meaningless
    #     (e.g. one stock's price series) but destroys magnitude
    #     information across sequences.
    # ``"astronomy_mjd_delta"``: legacy compat for astronomy datasets
    #     where column 0 is an MJD timestamp delta-encoded and scaled
    #     by an in-house dataset's convention. Provided for backwards
    #     compatibility; new users should not pick this.
    sequence_preprocessing: str = "none"

    # F-62 (2026-05-31): Lookahead meta-optimizer (Zhang 2019). Off by
    # default. When True, wraps the AdamW optimizer; +0.4-0.6% on tabular
    # MLP per RealMLP-TD 2024 ablations.
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

    # F-69 (2026-05-31): Mixup augmentation. Off by default. When True,
    # training_step mixes aux_features + labels by a Beta(alpha, alpha)
    # interpolation; pure SEQUENCE_ONLY mode (no aux_features) is a no-op.
    # Sequence-aware mixup (mixing padded sequences) is deferred to F-70.
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # F-68 (2026-05-31): exponential moving average of weights via
    # Lightning's WeightAveraging callback. Mirrors MLP's F-28; off by
    # default. When True, Lightning auto-swaps the EMA weights into the
    # live model on on_train_end -- predict() then uses the EMA copy
    # transparently (no save/load changes). +0.4% on tabular per
    # RealMLP-TD ablations. Cheaper than SWA (no LR warm-restart phase).
    # ema_decay default 0.999 mirrors torch.optim.swa_utils default.
    # Falls back to SWA-as-EMA shim when WeightAveraging is unavailable
    # (Lightning < 2.5).
    use_ema: bool = False
    ema_decay: float = 0.999

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
