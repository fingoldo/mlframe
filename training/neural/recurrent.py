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

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Any, Union

import numpy as np
import lightning as L
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lightning import LightningDataModule

if TYPE_CHECKING:
    import polars as pl_df
    import pandas as pd


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
# Utilities
# ----------------------------------------------------------------------------------------------------------------------------


def _ensure_numpy(arr: np.ndarray | object, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert DataFrame/Series/array-like to numpy array."""
    if arr is None:
        return None
    if hasattr(arr, "to_numpy"):  # Polars DataFrame/Series
        return arr.to_numpy().astype(dtype)
    if hasattr(arr, "values"):  # Pandas DataFrame/Series
        return arr.values.astype(dtype)
    return np.asarray(arr, dtype=dtype)


# ----------------------------------------------------------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------------------------------------------------------


class RNNType(str, Enum):
    """Supported sequence encoder architectures."""

    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"
    TRANSFORMER = "transformer"


class InputMode(str, Enum):
    """Input data modes for the classifier/regressor."""

    SEQUENCE_ONLY = "sequence"  # Raw time series only
    FEATURES_ONLY = "features"  # Tabular features only
    HYBRID = "hybrid"  # Both sequence + features


# ----------------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------------


@dataclass
class RecurrentConfig:
    """
    Configuration for recurrent models (classification and regression).

    Attributes:
        input_mode: Which inputs to use (SEQUENCE_ONLY, FEATURES_ONLY, HYBRID)
        rnn_type: RNN architecture (LSTM, GRU, RNN, TRANSFORMER) - ignored if FEATURES_ONLY
        hidden_size: RNN hidden state size
        num_layers: Number of RNN layers
        bidirectional: Whether to use bidirectional RNN
        use_attention: Whether to use attention pooling (vs last hidden)
        n_heads: Number of attention heads (Transformer only)
        dim_feedforward: Feedforward dimension in transformer
        mlp_hidden_sizes: Tuple of MLP hidden layer sizes
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs to wait before early stopping
        gradient_clip_val: Gradient clipping value
        precision: Training precision ("16-mixed" for AMP, "32-true" for full)
        early_stopping_monitor: Metric to monitor ("val_loss" or "val_auprc")
        use_stratified_sampler: Use weighted sampling for imbalanced data
        accelerator: Device to use ("auto", "gpu", "cpu")
        num_workers: DataLoader workers
        scale_features: Whether to StandardScaler auxiliary features
        num_classes: Number of output classes (for classification)
    """

    # Input mode
    input_mode: InputMode = InputMode.HYBRID

    # Sequence Encoder Architecture (ignored if input_mode=FEATURES_ONLY)
    rnn_type: RNNType = RNNType.LSTM
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True  # For RNN/LSTM/GRU only
    use_attention: bool = True  # For RNN/LSTM/GRU only

    # Transformer-specific (only used if rnn_type=TRANSFORMER)
    n_heads: int = 4
    dim_feedforward: int = 256

    # MLP Head
    mlp_hidden_sizes: Tuple[int, ...] = (256, 128)
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

    # Output
    num_classes: int = 2  # For classification; ignored for regression

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")


# ----------------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentDataset(Dataset):
    """
    Dataset for variable-length sequences with optional auxiliary features.

    Handles variable-length sequences and optional tabular features.
    Supports both classification (integer labels) and regression (float labels).
    """

    __slots__ = ("sequences", "aux_features", "labels", "sample_weights", "_has_sequences", "_is_regression")

    def __init__(
        self,
        sequences: List[np.ndarray] | None,
        aux_features: np.ndarray | None,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
        is_regression: bool = False,
    ) -> None:
        """
        Initialize dataset.

        Args:
            sequences: List of (seq_len, n_features) arrays, or None
            aux_features: (n_samples, n_features) array, or None
            labels: (n_samples,) array of labels
            sample_weights: (n_samples,) array of weights, or None
            is_regression: Whether this is a regression task (affects label dtype)
        """
        self.sequences = sequences
        self._has_sequences = sequences is not None
        self._is_regression = is_regression

        # Convert labels to appropriate dtype
        labels_np = np.asarray(labels)
        if is_regression:
            self.labels = torch.as_tensor(labels_np, dtype=torch.float32)
        else:
            self.labels = torch.as_tensor(labels_np, dtype=torch.long)

        # Pre-convert fixed-size arrays to tensors
        aux_np = _ensure_numpy(aux_features)
        self.aux_features = torch.as_tensor(aux_np, dtype=torch.float32) if aux_np is not None else None

        weights_np = _ensure_numpy(sample_weights)
        self.sample_weights = torch.as_tensor(weights_np, dtype=torch.float32) if weights_np is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        item: Dict[str, torch.Tensor] = {"labels": self.labels[idx]}

        if self._has_sequences:
            item["sequence"] = torch.as_tensor(self.sequences[idx], dtype=torch.float32)

        if self.aux_features is not None:
            item["aux_features"] = self.aux_features[idx]

        if self.sample_weights is not None:
            item["sample_weights"] = self.sample_weights[idx]

        return item


def recurrent_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function handling variable-length sequences.

    Pads sequences to max length in batch.

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Collated batch dict with:
        - sequences: (batch, max_len, n_features) padded tensor
        - lengths: (batch,) original lengths
        - aux_features: (batch, n_features) if present
        - labels: (batch,)
        - sample_weights: (batch,) if present
    """
    result: Dict[str, torch.Tensor] = {}

    # Labels (always present)
    result["labels"] = torch.stack([item["labels"] for item in batch])

    # Sequences (variable length)
    if "sequence" in batch[0]:
        sequences = [item["sequence"] for item in batch]
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        result["sequences"] = padded
        result["lengths"] = lengths

    # Auxiliary features
    if "aux_features" in batch[0]:
        result["aux_features"] = torch.stack([item["aux_features"] for item in batch])

    # Sample weights
    if "sample_weights" in batch[0]:
        result["sample_weights"] = torch.stack([item["sample_weights"] for item in batch])

    return result


# ----------------------------------------------------------------------------------------------------------------------------
# RecurrentDataModule
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentDataModule(LightningDataModule):
    """
    Lightning DataModule for recurrent models with sequence data.

    Handles train/val/test/predict stages with proper sequence handling.
    """

    def __init__(
        self,
        train_sequences: Optional[List[np.ndarray]] = None,
        train_features: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
        train_sample_weight: Optional[np.ndarray] = None,
        val_sequences: Optional[List[np.ndarray]] = None,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_sample_weight: Optional[np.ndarray] = None,
        test_sequences: Optional[List[np.ndarray]] = None,
        test_features: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        batch_size: int = 256,
        num_workers: int = 0,
        is_regression: bool = False,
        use_stratified_sampler: bool = True,
    ):
        """
        Initialize DataModule.

        Args:
            train_sequences: Training sequences (list of variable-length arrays)
            train_features: Training tabular features
            train_labels: Training labels
            train_sample_weight: Training sample weights
            val_sequences: Validation sequences
            val_features: Validation tabular features
            val_labels: Validation labels
            val_sample_weight: Validation sample weights
            test_sequences: Test sequences
            test_features: Test tabular features
            test_labels: Test labels
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            is_regression: Whether this is a regression task
            use_stratified_sampler: Use weighted sampling for imbalanced data
        """
        super().__init__()
        self.train_sequences = train_sequences
        self.train_features = train_features
        self.train_labels = train_labels
        self.train_sample_weight = train_sample_weight
        self.val_sequences = val_sequences
        self.val_features = val_features
        self.val_labels = val_labels
        self.val_sample_weight = val_sample_weight
        self.test_sequences = test_sequences
        self.test_features = test_features
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_regression = is_regression
        self.use_stratified_sampler = use_stratified_sampler

        # For dynamic prediction
        self.predict_sequences = None
        self.predict_features = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        pass  # Data is already provided in __init__

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        dataset = RecurrentDataset(
            sequences=self.train_sequences,
            aux_features=self.train_features,
            labels=self.train_labels,
            sample_weights=self.train_sample_weight,
            is_regression=self.is_regression,
        )

        sampler = None
        shuffle = True

        # Use stratified sampling for imbalanced classification data
        if self.use_stratified_sampler and not self.is_regression:
            labels = dataset.labels.numpy()
            class_counts = np.bincount(labels)
            if len(class_counts) > 1 and all(c > 0 for c in class_counts):
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[labels]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
                )
                shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        if self.val_labels is None:
            raise RuntimeError("Validation labels not provided")

        dataset = RecurrentDataset(
            sequences=self.val_sequences,
            aux_features=self.val_features,
            labels=self.val_labels,
            sample_weights=self.val_sample_weight,
            is_regression=self.is_regression,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        if self.test_labels is None:
            raise RuntimeError("Test labels not provided")

        dataset = RecurrentDataset(
            sequences=self.test_sequences,
            aux_features=self.test_features,
            labels=self.test_labels,
            is_regression=self.is_regression,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction DataLoader."""
        if self.predict_sequences is None and self.predict_features is None:
            raise RuntimeError("Call setup_predict() first")

        n_samples = len(self.predict_sequences) if self.predict_sequences else len(self.predict_features)
        dummy_labels = np.zeros(n_samples, dtype=np.float32 if self.is_regression else np.int64)

        dataset = RecurrentDataset(
            sequences=self.predict_sequences,
            aux_features=self.predict_features,
            labels=dummy_labels,
            is_regression=self.is_regression,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def setup_predict(
        self,
        sequences: Optional[List[np.ndarray]] = None,
        features: Optional[np.ndarray] = None,
    ):
        """Setup data for prediction."""
        self.predict_sequences = sequences
        self.predict_features = features


# ----------------------------------------------------------------------------------------------------------------------------
# Model Components
# ----------------------------------------------------------------------------------------------------------------------------


class AttentionPooling(nn.Module):
    """
    Attention mechanism to aggregate variable-length RNN outputs.

    Learns importance weights for each timestep, producing a fixed-size
    context vector regardless of sequence length.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        rnn_output: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            rnn_output: (batch, seq_len, hidden) RNN outputs
            lengths: (batch,) original sequence lengths

        Returns:
            Context vector (batch, hidden)
        """
        batch_size, max_len, hidden_size = rnn_output.size()
        device = rnn_output.device

        # Compute attention scores
        scores = self.attention(rnn_output).squeeze(-1)  # (batch, seq_len)

        # Create mask for padded positions
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over valid positions
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)

        return context


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds position information to input embeddings since Transformers
    have no inherent notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer encoder for variable-length sequences.

    Projects input to hidden_size, applies positional encoding,
    then runs through TransformerEncoder layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project input to hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for classification (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.hidden_size = hidden_size

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode sequences with Transformer.

        Args:
            sequences: (batch, seq_len, input_size) padded sequences
            lengths: (batch,) original sequence lengths

        Returns:
            Context vector (batch, hidden_size) from CLS token
        """
        batch_size, max_len, _ = sequences.size()
        device = sequences.device

        # Project to hidden size
        x = self.input_projection(sequences)  # (batch, seq_len, hidden_size)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_size)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for padded positions (True = ignore)
        seq_positions = torch.arange(max_len + 1, device=device).unsqueeze(0)
        padding_mask = seq_positions > lengths.unsqueeze(1)

        # Run through transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Return CLS token representation
        return x[:, 0, :]  # (batch, hidden_size)


class MLPHead(nn.Module):
    """
    MLP classification/regression head.

    Used by all input modes for final output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, ...],
        output_size: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        layers: List[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits/predictions."""
        return self.mlp(x)


# ----------------------------------------------------------------------------------------------------------------------------
# Lightning Module
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentTorchModel(L.LightningModule):
    """
    PyTorch Lightning module for sequence classification/regression.

    Supports three input modes:
    - SEQUENCE_ONLY: RNN on raw time series
    - FEATURES_ONLY: MLP on tabular features
    - HYBRID: Both combined
    """

    def __init__(
        self,
        config: RecurrentConfig,
        seq_input_size: int = 4,
        aux_input_size: int = 0,
        class_weight: torch.Tensor | None = None,
        is_regression: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weight"])
        self.config = config
        self.is_regression = is_regression

        # Register class_weight as buffer so it moves with model to device
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

        # Build components based on input mode
        self._build_model(seq_input_size, aux_input_size)
        self._setup_metrics()
        self._setup_loss_functions()

    def _build_model(self, seq_input_size: int, aux_input_size: int) -> None:
        """Construct model components based on input mode."""
        mlp_input_size = 0
        self._use_transformer = False

        if self.config.input_mode != InputMode.FEATURES_ONLY:
            if self.config.rnn_type == RNNType.TRANSFORMER:
                # Build Transformer encoder
                self._use_transformer = True
                self.transformer_encoder = TransformerSequenceEncoder(
                    input_size=seq_input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    n_heads=self.config.n_heads,
                    dim_feedforward=self.config.dim_feedforward,
                    dropout=self.config.dropout,
                )
                mlp_input_size += self.config.hidden_size
            else:
                # Build RNN (LSTM/GRU/RNN)
                rnn_class = {
                    RNNType.LSTM: nn.LSTM,
                    RNNType.GRU: nn.GRU,
                    RNNType.RNN: nn.RNN,
                }[self.config.rnn_type]

                self.rnn = rnn_class(
                    input_size=seq_input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    batch_first=True,
                    bidirectional=self.config.bidirectional,
                    dropout=self.config.dropout if self.config.num_layers > 1 else 0,
                )

                rnn_output_size = self.config.hidden_size * (2 if self.config.bidirectional else 1)

                if self.config.use_attention:
                    self.attention = AttentionPooling(rnn_output_size)

                mlp_input_size += rnn_output_size

        if self.config.input_mode != InputMode.SEQUENCE_ONLY:
            mlp_input_size += aux_input_size

        # Build MLP head
        output_size = 1 if self.is_regression else self.config.num_classes
        self.mlp_head = MLPHead(
            input_size=mlp_input_size,
            hidden_sizes=self.config.mlp_hidden_sizes,
            output_size=output_size,
            dropout=self.config.dropout,
        )

    def _setup_metrics(self) -> None:
        """Initialize torchmetrics for proper metric aggregation."""
        try:
            from torchmetrics import AUROC, Accuracy, AveragePrecision, MeanSquaredError, R2Score

            if self.is_regression:
                self.train_mse = MeanSquaredError()
                self.val_mse = MeanSquaredError()
                self.val_r2 = R2Score()
            else:
                self.train_acc = Accuracy(task="multiclass" if self.config.num_classes > 2 else "binary",
                                          num_classes=self.config.num_classes if self.config.num_classes > 2 else None)
                self.val_acc = Accuracy(task="multiclass" if self.config.num_classes > 2 else "binary",
                                        num_classes=self.config.num_classes if self.config.num_classes > 2 else None)
                if self.config.num_classes == 2:
                    self.val_auroc = AUROC(task="binary")
                    self.val_auprc = AveragePrecision(task="binary")
            self._has_metrics = True
        except ImportError:
            self._has_metrics = False
            warnings.warn("torchmetrics not installed, skipping metric logging")

    def _setup_loss_functions(self) -> None:
        """Pre-create loss functions."""
        if self.is_regression:
            self._loss_fn_unreduced = nn.MSELoss(reduction="none")
            self._loss_fn_mean = nn.MSELoss(reduction="mean")
        else:
            self._loss_fn_unreduced = nn.CrossEntropyLoss(weight=self.class_weight, reduction="none")
            self._loss_fn_mean = nn.CrossEntropyLoss(weight=self.class_weight, reduction="mean")

    def forward(
        self,
        sequences: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: (batch, seq_len, n_features) padded sequences
            lengths: (batch,) original sequence lengths
            aux_features: (batch, n_features) tabular features

        Returns:
            Logits (batch, num_classes) for classification or (batch, 1) for regression
        """
        features_list: List[torch.Tensor] = []

        # Process sequences if needed
        if self.config.input_mode != InputMode.FEATURES_ONLY:
            if sequences is None or lengths is None:
                raise ValueError("sequences and lengths required for this input mode")
            rnn_out = self._encode_sequences(sequences, lengths)
            features_list.append(rnn_out)

        # Add auxiliary features if needed
        if self.config.input_mode != InputMode.SEQUENCE_ONLY:
            if aux_features is None:
                raise ValueError("aux_features required for this input mode")
            features_list.append(aux_features)

        # Concatenate and predict
        if len(features_list) > 1:
            combined = torch.cat(features_list, dim=1)
        else:
            combined = features_list[0]

        return self.mlp_head(combined)

    def _encode_sequences(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sequences through RNN or Transformer."""
        if self._use_transformer:
            return self.transformer_encoder(sequences, lengths)

        # RNN path: Pack for efficient processing
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Pool to fixed size
        if self.config.use_attention:
            return self.attention(rnn_out, lengths)
        else:
            return self._get_last_hidden(rnn_out, lengths)

    @staticmethod
    def _get_last_hidden(rnn_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Extract last valid hidden state for each sequence."""
        device = rnn_out.device
        batch_size = rnn_out.size(0)
        hidden_size = rnn_out.size(2)

        last_indices = (lengths - 1).to(device).view(-1, 1, 1).expand(-1, 1, hidden_size)
        return rnn_out.gather(1, last_indices).squeeze(1)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        logits = self._forward_batch(batch)
        loss = self._compute_weighted_loss(
            logits,
            batch["labels"],
            batch.get("sample_weights"),
        )
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self._has_metrics:
            if self.is_regression:
                preds = logits.squeeze(-1)
                self.train_mse(preds, batch["labels"])
                self.log("train_mse", self.train_mse, prog_bar=True, on_step=False, on_epoch=True)
            else:
                probs = torch.softmax(logits, dim=1)
                if self.config.num_classes == 2:
                    preds = (probs[:, 1] >= 0.5).long()
                else:
                    preds = probs.argmax(dim=1)
                self.train_acc(preds, batch["labels"])
                self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step."""
        logits = self._forward_batch(batch)
        loss = self._compute_weighted_loss(logits, batch["labels"], None)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self._has_metrics:
            if self.is_regression:
                preds = logits.squeeze(-1)
                self.val_mse(preds, batch["labels"])
                self.val_r2(preds, batch["labels"])
                self.log("val_mse", self.val_mse, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_r2", self.val_r2, prog_bar=True, on_step=False, on_epoch=True)
            else:
                probs = torch.softmax(logits, dim=1)
                if self.config.num_classes == 2:
                    preds = (probs[:, 1] >= 0.5).long()
                    self.val_acc(preds, batch["labels"])
                    self.val_auroc(probs[:, 1], batch["labels"])
                    self.val_auprc(probs[:, 1], batch["labels"])
                    self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
                    self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)
                    self.log("val_auprc", self.val_auprc, on_step=False, on_epoch=True)
                else:
                    preds = probs.argmax(dim=1)
                    self.val_acc(preds, batch["labels"])
                    self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        logits = self._forward_batch(batch)
        if self.is_regression:
            return logits.squeeze(-1)
        return torch.softmax(logits, dim=1)

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        """Helper to forward a batch dict."""
        return self(
            sequences=batch.get("sequences"),
            lengths=batch.get("lengths"),
            aux_features=batch.get("aux_features"),
        )

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute loss with optional sample weights."""
        if self.is_regression:
            preds = logits.squeeze(-1)
            if sample_weights is not None:
                losses = self._loss_fn_unreduced(preds, labels)
                return (losses * sample_weights).mean()
            return self._loss_fn_mean(preds, labels)
        else:
            if sample_weights is not None:
                losses = self._loss_fn_unreduced(logits, labels)
                return (losses * sample_weights).mean()
            return self._loss_fn_mean(logits, labels)

    def configure_optimizers(self):
        """Configure AdamW with OneCycleLR scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.trainer and self.trainer.estimated_stepping_batches:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn Wrappers - Base
# ----------------------------------------------------------------------------------------------------------------------------


class _RecurrentWrapperBase(BaseEstimator):
    """
    Base class for sklearn-compatible recurrent model wrappers.

    Provides common functionality for both classification and regression.
    """

    _is_regression: bool = False

    def __init__(
        self,
        config: RecurrentConfig | None = None,
        random_state: int = 42,
    ) -> None:
        self.config = config or RecurrentConfig()
        self.random_state = random_state
        self.model: RecurrentTorchModel | None = None
        self.trainer: L.Trainer | None = None
        self._aux_input_size: int = 0
        self._seq_input_size: int = 4
        self._feature_scaler: StandardScaler | None = None
        self._prediction_cache: Dict[int, np.ndarray] = {}

    def _validate_inputs(
        self,
        features: np.ndarray | None,
        sequences: List[np.ndarray] | None,
    ) -> None:
        """Validate inputs match the configured input mode."""
        mode = self.config.input_mode

        if mode == InputMode.FEATURES_ONLY and features is None:
            raise ValueError("features required for FEATURES_ONLY mode")
        if mode == InputMode.SEQUENCE_ONLY and sequences is None:
            raise ValueError("sequences required for SEQUENCE_ONLY mode")
        if mode == InputMode.HYBRID and (features is None or sequences is None):
            raise ValueError("both features and sequences required for HYBRID mode")

    def _create_dataset(
        self,
        sequences: List[np.ndarray] | None,
        features: np.ndarray | None,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> RecurrentDataset:
        """Create dataset with proper preprocessing."""
        processed_seqs = None
        if sequences is not None:
            if len(sequences) > 10_000:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    processed_seqs = list(executor.map(self._preprocess_sequence, sequences))
            else:
                processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        scaled_features = _ensure_numpy(features)
        if scaled_features is not None and self._feature_scaler is not None:
            scaled_features = self._feature_scaler.transform(scaled_features).astype(np.float32)

        return RecurrentDataset(
            sequences=processed_seqs,
            aux_features=scaled_features,
            labels=labels,
            sample_weights=sample_weights,
            is_regression=self._is_regression,
        )

    def _create_eval_dataset(self, eval_set: tuple) -> RecurrentDataset:
        """Create validation dataset from eval_set tuple."""
        if len(eval_set) == 2:
            features, labels = eval_set
            return self._create_dataset(None, features, labels)
        elif len(eval_set) == 3:
            sequences, features, labels = eval_set
            return self._create_dataset(sequences, features, labels)
        else:
            raise ValueError(f"eval_set must have 2 or 3 elements, got {len(eval_set)}")

    _DELTA_MJD_SCALE: float = 10.0
    _STD_EPSILON: float = 1e-8

    @staticmethod
    def _preprocess_sequence(seq: np.ndarray) -> np.ndarray:
        """Preprocess a single sequence with proper normalization."""
        result = seq.astype(np.float32)
        n_cols = result.shape[1]

        # Column 0: Delta encode and scale
        if n_cols > 0:
            delta = np.zeros(len(result), dtype=np.float32)
            delta[1:] = np.diff(seq[:, 0])
            result[:, 0] = delta / _RecurrentWrapperBase._DELTA_MJD_SCALE

        # Column 1+: Z-score normalize
        for col_idx in range(1, n_cols):
            col = seq[:, col_idx]
            col_mean = np.mean(col)
            col_std = np.std(col)
            if col_std > _RecurrentWrapperBase._STD_EPSILON:
                result[:, col_idx] = (col - col_mean) / col_std
            else:
                result[:, col_idx] = 0.0

        return result

    def _create_dataloader(
        self,
        dataset: RecurrentDataset,
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader:
        """Create DataLoader with proper collate function."""
        sampler = None

        if shuffle and self.config.use_stratified_sampler and not self._is_regression:
            labels = dataset.labels.numpy()
            class_counts = np.bincount(labels)
            if len(class_counts) > 1 and all(c > 0 for c in class_counts):
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[labels]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
                )
                shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
        )

    def _create_model(
        self,
        seq_input_size: int,
        aux_input_size: int,
        class_weight: Dict[int, float] | None,
    ) -> RecurrentTorchModel:
        """Create model instance."""
        weight_tensor = None
        if class_weight and not self._is_regression:
            weight_tensor = torch.tensor(
                [class_weight.get(i, 1.0) for i in range(self.config.num_classes)],
                dtype=torch.float32,
            )

        return RecurrentTorchModel(
            config=self.config,
            seq_input_size=seq_input_size,
            aux_input_size=aux_input_size,
            class_weight=weight_tensor,
            is_regression=self._is_regression,
        )

    def _create_trainer(self, has_validation: bool, plot: bool) -> Tuple[L.Trainer, Any]:
        """Create Lightning Trainer with callbacks."""
        callbacks: list = []
        checkpoint_callback = None

        if has_validation:
            monitor = self.config.early_stopping_monitor
            mode = "min" if "loss" in monitor or "mse" in monitor else "max"

            callbacks.append(
                L.pytorch.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=self.config.early_stopping_patience,
                    mode=mode,
                )
            )
            checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                save_last=False,
            )
            callbacks.append(checkpoint_callback)

        trainer = L.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=self.config.accelerator,
            precision=self.config.precision,
            callbacks=callbacks,
            gradient_clip_val=self.config.gradient_clip_val,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=plot,
            deterministic="warn",
        )
        return trainer, checkpoint_callback

    def _clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()

    @staticmethod
    def _compute_cache_key(
        features: np.ndarray | None,
        sequences: List[np.ndarray] | None,
    ) -> int:
        """Compute cache key from input arrays."""
        parts: list = []

        if features is not None:
            parts.append(features.shape)
            parts.append(features.dtype.str)
            if features.size > 0:
                flat = features.ravel()
                indices = [0, len(flat) // 2, -1] if len(flat) > 2 else list(range(len(flat)))
                parts.append(tuple(float(flat[i]) for i in indices))

        if sequences is not None:
            parts.append(len(sequences))
            if sequences:
                sample_indices = [0, len(sequences) // 2, -1] if len(sequences) > 2 else list(range(len(sequences)))
                for idx in sample_indices:
                    seq = sequences[idx]
                    parts.append(seq.shape)
                    if seq.size > 0:
                        flat = seq.ravel()
                        parts.append((float(flat[0]), float(flat[-1])))

        return hash(tuple(map(str, parts)))

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")

        state = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "random_state": self.random_state,
            "aux_input_size": self._aux_input_size,
            "seq_input_size": self._seq_input_size,
            "feature_scaler": self._feature_scaler,
            "is_regression": self._is_regression,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "_RecurrentWrapperBase":
        """Load model from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        wrapper = cls(config=state["config"], random_state=state["random_state"])
        wrapper._aux_input_size = state.get("aux_input_size", 0)
        wrapper._seq_input_size = state.get("seq_input_size", 4)
        wrapper._feature_scaler = state.get("feature_scaler", None)

        wrapper.model = RecurrentTorchModel(
            config=state["config"],
            seq_input_size=wrapper._seq_input_size,
            aux_input_size=wrapper._aux_input_size,
            is_regression=state.get("is_regression", wrapper._is_regression),
        )
        wrapper.model.load_state_dict(state["model_state_dict"])
        wrapper.model.eval()

        return wrapper


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn Wrappers - Classifier
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentClassifierWrapper(_RecurrentWrapperBase, ClassifierMixin):
    """
    Sklearn-compatible wrapper for RecurrentTorchModel (classification).

    Provides fit/predict/predict_proba interface.
    """

    _estimator_type = "classifier"
    _is_regression = False

    def fit(
        self,
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sequences: List[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        class_weight: Dict[int, float] | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
    ) -> "RecurrentClassifierWrapper":
        """
        Train the model.

        Args:
            features: (n_samples, n_features) tabular features
            labels: (n_samples,) labels
            sample_weight: (n_samples,) per-sample weights
            sequences: List of (seq_len, n_features) arrays
            eval_set: Validation data tuple
            class_weight: Class weights dict
            plot: Whether to enable logging
            plot_file: Path for logs (unused, for compatibility)

        Returns:
            self for method chaining
        """
        if labels is None:
            raise ValueError("labels is required")

        self._validate_inputs(features, sequences)
        self._clear_cache()

        if self.config.scale_features and features is not None:
            self._feature_scaler = StandardScaler()
            self._feature_scaler.fit(features)

        L.seed_everything(self.random_state, workers=True)

        train_dataset = self._create_dataset(sequences, features, labels, sample_weight)
        val_dataset = self._create_eval_dataset(eval_set) if eval_set else None

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None

        self._aux_input_size = features.shape[1] if features is not None else 0
        self._seq_input_size = sequences[0].shape[1] if sequences is not None and len(sequences) > 0 else 4

        self.model = self._create_model(
            seq_input_size=self._seq_input_size,
            aux_input_size=self._aux_input_size,
            class_weight=class_weight,
        )

        self.trainer, checkpoint_callback = self._create_trainer(val_loader is not None, plot)
        self.trainer.fit(self.model, train_loader, val_loader)

        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            try:
                self.model = RecurrentTorchModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    config=self.config,
                    seq_input_size=self._seq_input_size,
                    aux_input_size=self._aux_input_size,
                    is_regression=False,
                    weights_only=False,
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, using final model: {e}")

        return self

    def predict_proba(
        self,
        features: np.ndarray | None = None,
        sequences: List[np.ndarray] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, n_features) arrays
            batch_size: Override batch size for prediction

        Returns:
            (n_samples, num_classes) array of probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self._validate_inputs(features, sequences)

        cache_key = self._compute_cache_key(features, sequences)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        n_samples = len(sequences) if sequences is not None else len(features)

        dataset = self._create_dataset(
            sequences,
            features,
            labels=np.zeros(n_samples, dtype=np.int64),
        )
        loader = self._create_dataloader(dataset, shuffle=False, batch_size=batch_size)

        predict_trainer = L.Trainer(
            accelerator=self.config.accelerator,
            precision=self.config.precision,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        predictions = predict_trainer.predict(self.model, loader)

        result = torch.cat(predictions, dim=0).float().cpu().numpy().astype(np.float32)

        self._prediction_cache[cache_key] = result
        return result

    def predict(
        self,
        features: np.ndarray | None = None,
        sequences: List[np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, n_features) arrays

        Returns:
            (n_samples,) array of predictions
        """
        proba = self.predict_proba(features, sequences)
        return proba.argmax(axis=1).astype(np.int8)


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn Wrappers - Regressor
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentRegressorWrapper(_RecurrentWrapperBase, RegressorMixin):
    """
    Sklearn-compatible wrapper for RecurrentTorchModel (regression).

    Provides fit/predict interface.
    """

    _estimator_type = "regressor"
    _is_regression = True

    def __init__(
        self,
        config: RecurrentConfig | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(config, random_state)
        # Override early_stopping_monitor for regression
        if self.config.early_stopping_monitor == "val_auprc":
            self.config.early_stopping_monitor = "val_loss"

    def fit(
        self,
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sequences: List[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
    ) -> "RecurrentRegressorWrapper":
        """
        Train the model.

        Args:
            features: (n_samples, n_features) tabular features
            labels: (n_samples,) continuous target values
            sample_weight: (n_samples,) per-sample weights
            sequences: List of (seq_len, n_features) arrays
            eval_set: Validation data tuple
            plot: Whether to enable logging
            plot_file: Path for logs (unused, for compatibility)

        Returns:
            self for method chaining
        """
        if labels is None:
            raise ValueError("labels is required")

        self._validate_inputs(features, sequences)
        self._clear_cache()

        if self.config.scale_features and features is not None:
            self._feature_scaler = StandardScaler()
            self._feature_scaler.fit(features)

        L.seed_everything(self.random_state, workers=True)

        train_dataset = self._create_dataset(sequences, features, labels, sample_weight)
        val_dataset = self._create_eval_dataset(eval_set) if eval_set else None

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None

        self._aux_input_size = features.shape[1] if features is not None else 0
        self._seq_input_size = sequences[0].shape[1] if sequences is not None and len(sequences) > 0 else 4

        self.model = self._create_model(
            seq_input_size=self._seq_input_size,
            aux_input_size=self._aux_input_size,
            class_weight=None,  # No class weights for regression
        )

        self.trainer, checkpoint_callback = self._create_trainer(val_loader is not None, plot)
        self.trainer.fit(self.model, train_loader, val_loader)

        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            try:
                self.model = RecurrentTorchModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    config=self.config,
                    seq_input_size=self._seq_input_size,
                    aux_input_size=self._aux_input_size,
                    is_regression=True,
                    weights_only=False,
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, using final model: {e}")

        return self

    def predict(
        self,
        features: np.ndarray | None = None,
        sequences: List[np.ndarray] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Predict continuous values.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, n_features) arrays
            batch_size: Override batch size for prediction

        Returns:
            (n_samples,) array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self._validate_inputs(features, sequences)

        cache_key = self._compute_cache_key(features, sequences)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        n_samples = len(sequences) if sequences is not None else len(features)

        dataset = self._create_dataset(
            sequences,
            features,
            labels=np.zeros(n_samples, dtype=np.float32),
        )
        loader = self._create_dataloader(dataset, shuffle=False, batch_size=batch_size)

        predict_trainer = L.Trainer(
            accelerator=self.config.accelerator,
            precision=self.config.precision,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        predictions = predict_trainer.predict(self.model, loader)

        result = torch.cat(predictions, dim=0).float().cpu().numpy().astype(np.float32)

        self._prediction_cache[cache_key] = result
        return result


# ----------------------------------------------------------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------------------------------------------------------


def extract_sequences(
    df: "pl_df.DataFrame",
    indices: np.ndarray | List[int] | None = None,
    columns: Tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> List[np.ndarray]:
    """
    Extract raw time series from Polars DataFrame with list columns.

    Args:
        df: DataFrame with list columns
        indices: Optional subset of row indices to extract
        columns: Column names to stack into sequences

    Returns:
        List of (seq_len, n_columns) float32 arrays
    """
    if indices is not None:
        df = df[indices]

    n_rows = len(df)
    n_cols = len(columns)

    col_data = [df[col].to_list() for col in columns]

    result: List[np.ndarray] = []
    for i in range(n_rows):
        seq_len = len(col_data[0][i])
        arr = np.empty((seq_len, n_cols), dtype=np.float32)
        for j in range(n_cols):
            arr[:, j] = col_data[j][i]
        result.append(arr)

    return result


def extract_sequences_chunked(
    df: "pl_df.DataFrame",
    indices: np.ndarray | List[int] | None = None,
    chunk_size: int = 100_000,
    columns: Tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> List[np.ndarray]:
    """
    Memory-efficient sequence extraction for large datasets.

    Args:
        df: DataFrame with list columns
        indices: Optional subset of row indices
        chunk_size: Number of rows per chunk
        columns: Column names to extract

    Returns:
        List of (seq_len, n_columns) float32 arrays
    """
    if indices is not None:
        indices = np.asarray(indices)
    else:
        indices = np.arange(len(df))

    sequences: List[np.ndarray] = []

    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        chunk_seqs = extract_sequences(df, chunk_indices.tolist(), columns)
        sequences.extend(chunk_seqs)

    return sequences
