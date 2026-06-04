"""``RecurrentTorchModel`` -- the PyTorch Lightning recurrent model.

Wave 103 (2026-05-21): split out from ``training/neural/recurrent.py`` to
keep that file below the 1k-line monolith threshold. Behaviour preserved
bit-for-bit; the class is re-exported from ``recurrent`` so existing
imports continue to work.
"""
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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Dict, Any

import numpy as np
import lightning as L
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
from concurrent.futures import ThreadPoolExecutor

try:
    import xxhash as _xxhash  # noqa: F401  module-top for hot cache-key path
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False
import hashlib as _hashlib

if TYPE_CHECKING:
    import polars as pl_df


# Default number of channels per timestep when callers don't supply
# sequences (FEATURES_ONLY mode keeps the RNN branch dormant). Kept as a
# module-level constant so the save/load round-trip and the wrapper
# initialisation agree on the same fallback.
_DEFAULT_SEQ_INPUT_SIZE: int = 4


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
    # NOTE: the sklearn wrappers (RecurrentClassifierWrapper/RegressorWrapper) and
    # sequence utilities (extract_sequences/_chunked) are defined in recurrent.py /
    # _recurrent_sequences.py and exported from there, not from this impl module.
]


# ----------------------------------------------------------------------------------------------------------------------------
# Late re-exports + shared utility
# ----------------------------------------------------------------------------------------------------------------------------


from .base import _ensure_numpy  # noqa: E402,F401  shared with _recurrent_data
from ._recurrent_config import RNNType, InputMode, RecurrentConfig  # noqa: E402,F401
from ._recurrent_data import RecurrentDataset, recurrent_collate_fn, RecurrentDataModule  # noqa: E402,F401
from ._recurrent_arch import AttentionPooling, PositionalEncoding, TransformerSequenceEncoder, MLPHead  # noqa: E402,F401


# Substring-match on the monitor name was buggy: "val_log_likelihood" contains
# "loss" -> wrong "min" direction (likelihood is max-better). Explicit table
# of base-metric -> direction, matched on whitespace/underscore-delimited
# tokens of the trailing component.
_MONITOR_MIN_KEYS: frozenset[str] = frozenset({
    "loss", "mse", "mae", "rmse", "mape", "smape", "logloss", "log_loss",
    "huber", "kl", "kl_div", "perplexity", "nll", "error", "err",
})
_MONITOR_MAX_KEYS: frozenset[str] = frozenset({
    "acc", "accuracy", "f1", "auroc", "auc", "roc_auc", "pr_auc", "ap",
    "precision", "recall", "iou", "r2", "score", "likelihood", "log_likelihood",
    "ndcg", "map", "mrr",
})



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
        seq_input_size: int = _DEFAULT_SEQ_INPUT_SIZE,
        aux_input_size: int = 0,
        class_weight: torch.Tensor | None = None,
        is_regression: bool = False,
        task_type: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weight"])
        self.config = config
        self.is_regression = is_regression
        # task_type selects loss + activation:
        #   None / "binary" / "multiclass" -> CrossEntropyLoss + softmax (default)
        #   "multilabel"                   -> BCEWithLogitsLoss + sigmoid (each output binary)
        # No effect when is_regression=True (always MSE + identity).
        self.task_type = task_type

        # Register class_weight as buffer so it moves with model to device
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

        self._build_model(seq_input_size, aux_input_size)
        self._setup_metrics()
        self._setup_loss_functions()

    def _build_model(self, seq_input_size: int, aux_input_size: int) -> None:
        """Construct model components based on input mode."""
        mlp_input_size = 0
        self._use_transformer = False

        if self.config.input_mode != InputMode.FEATURES_ONLY:
            if self.config.rnn_type == RNNType.TRANSFORMER:
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
            elif self.task_type == "multilabel":
                # Per-label torchmetrics for log-only; loss computed via BCEWithLogitsLoss.
                # Suite computes its own multi-target metrics downstream from sigmoid probs.
                K = self.config.num_classes
                self.train_acc = Accuracy(task="multilabel", num_labels=K)
                self.val_acc = Accuracy(task="multilabel", num_labels=K)
                self.val_auroc = AUROC(task="multilabel", num_labels=K, average="macro")
                self.val_auprc = AveragePrecision(task="multilabel", num_labels=K, average="macro")
                self._has_metrics = True
                return
            else:
                K = self.config.num_classes
                if K > 2:
                    task_str = "multiclass"
                    self.train_acc = Accuracy(task=task_str, num_classes=K)
                    self.val_acc = Accuracy(task=task_str, num_classes=K)
                    self.val_auroc = AUROC(task=task_str, num_classes=K, average="macro")
                    self.val_auprc = AveragePrecision(task=task_str, num_classes=K, average="macro")
                else:
                    self.train_acc = Accuracy(task="binary")
                    self.val_acc = Accuracy(task="binary")
                    self.val_auroc = AUROC(task="binary")
                    self.val_auprc = AveragePrecision(task="binary")
            self._has_metrics = True
        except ImportError:
            self._has_metrics = False
            warnings.warn("torchmetrics not installed, skipping metric logging", stacklevel=2)

    def _setup_loss_functions(self) -> None:
        """Pre-instantiate loss functions for the active task.

        - regression: MSE
        - multilabel (``task_type='multilabel'``): per-label BCEWithLogitsLoss (each output independent binary; 2-D y of shape (N, K))
        - binary / multiclass (default): CrossEntropyLoss + softmax
        """
        if self.is_regression:
            self._loss_fn_unreduced = nn.MSELoss(reduction="none")
            self._loss_fn_mean = nn.MSELoss(reduction="mean")
        elif self.task_type == "multilabel":
            # Per-label sigmoid via BCEWithLogitsLoss (numerically stable).
            self._loss_fn_unreduced = nn.BCEWithLogitsLoss(reduction="none")
            self._loss_fn_mean = nn.BCEWithLogitsLoss(reduction="mean")
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
        features_list: list[torch.Tensor] = []

        if self.config.input_mode != InputMode.FEATURES_ONLY:
            if sequences is None or lengths is None:
                raise ValueError("sequences and lengths required for this input mode")
            rnn_out = self._encode_sequences(sequences, lengths)
            features_list.append(rnn_out)

        if self.config.input_mode != InputMode.SEQUENCE_ONLY:
            if aux_features is None:
                raise ValueError("aux_features required for this input mode")
            features_list.append(aux_features)

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

        # RNN path: pack_padded_sequence skips compute on padded steps; enforce_sorted=False lets us pass unsorted lengths.
        # pack_padded_sequence dispatches length-sort on CPU even when the data is on GPU.
        # F-53 (2026-05-31): use non_blocking=True so the host->device async stream can overlap; skip the .cpu() altogether
        # when lengths is already CPU (recurrent_collate_fn emits lengths as CPU torch.long; Lightning typically does not
        # auto-migrate them, but we defensively handle both). Per-forward saving: ~10-50 us on Pascal.
        # Guard: pack_padded_sequence raises RuntimeError when any length == 0 (zero-row sequence). Treat as length-1
        # padded row so the call stays valid; the caller's downstream attention/last-hidden gather will still work.
        lengths_cpu = lengths if lengths.device.type == "cpu" else lengths.detach().cpu()
        if (lengths_cpu <= 0).any():
            lengths_cpu = torch.clamp(lengths_cpu, min=1)
        packed = pack_padded_sequence(
            sequences,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Downstream attention/last-hidden expect lengths >= 1 (last_idx = lengths - 1 must be >=0).
        # Pass the clamped tensor on the original device to keep gather indices valid for zero-len rows.
        safe_lengths = lengths.clamp(min=1) if (lengths <= 0).any() else lengths

        if self.config.use_attention:
            return self.attention(rnn_out, safe_lengths)
        else:
            return self._get_last_hidden(
                rnn_out, safe_lengths,
                bidirectional=self.config.bidirectional,
            )

    @staticmethod
    def _get_last_hidden(
        rnn_out: torch.Tensor,
        lengths: torch.Tensor,
        bidirectional: bool = False,
    ) -> torch.Tensor:
        """Extract the last-valid-step hidden state(s) for each sequence.

        F-J fix (2026-05-31, audit-found silent correctness bug):

        For UNIDIRECTIONAL RNN, ``rnn_out[b, lengths[b]-1, :]`` is the
        forward-RNN summary of positions [0..lengths[b]-1] -- correct.

        For BIDIRECTIONAL RNN (rnn_out shape (B, T, 2*hidden) with
        rnn_out[..., :hidden] = forward channel and rnn_out[..., hidden:]
        = backward channel), the previous implementation gathered
        ``rnn_out[b, lengths[b]-1, :]`` for BOTH halves. The forward
        half is correct (summary of [0..length-1]) but the backward half
        is only the backward RNN's hidden state AT POSITION length-1,
        which after pack_padded_sequence has processed positions
        [length-1, length-2, ..., 0] in reverse means h_bwd[length-1]
        only summarises [length-1..length-1] -- just the LAST token,
        not the full sequence.

        The semantically correct backward summary lives at h_bwd[0]
        (the backward RNN's final hidden state in run-order, which is
        the summary of [0..length-1]). So the fix gathers:
          fwd_last  = rnn_out[b, length-1, :hidden]   (forward summary)
          bwd_first = rnn_out[b, 0,        hidden:]   (backward summary)
        and concatenates them along the channel axis.

        Pre-fix, every bidirectional + use_attention=False recurrent
        ensemble member was silently feeding the MLPHead a half-padding
        embedding. Estimated R^2 drop 0.1-0.3 on sequence tasks where
        the backward-channel summary actually carries signal (typical
        on tabular sequences with bidirectional defaults).
        """
        device = rnn_out.device
        total_channels = rnn_out.size(2)
        last_indices = (lengths - 1).to(device).view(-1, 1, 1)

        if not bidirectional:
            # Unchanged unidirectional path: gather the whole channel block
            # at position length-1.
            last_indices_full = last_indices.expand(-1, 1, total_channels)
            return rnn_out.gather(1, last_indices_full).squeeze(1)

        # Bidirectional: split into fwd / bwd halves, gather each at the
        # semantically correct position, then re-concatenate.
        assert total_channels % 2 == 0, (
            f"bidirectional rnn_out must have even channel count; got {total_channels}"
        )
        hidden_size = total_channels // 2
        fwd = rnn_out[:, :, :hidden_size]   # (B, T, H)
        bwd = rnn_out[:, :, hidden_size:]   # (B, T, H)

        # fwd_last_idx: (B, 1, H) gather at position lengths-1.
        fwd_last_idx = last_indices.expand(-1, 1, hidden_size)
        fwd_summary = fwd.gather(1, fwd_last_idx).squeeze(1)  # (B, H)

        # bwd_first: position 0 across the batch -- single slice is fine,
        # no gather needed.
        bwd_summary = bwd[:, 0, :]                              # (B, H)

        return torch.cat([fwd_summary, bwd_summary], dim=1)     # (B, 2H)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # F-69/F-70 (2026-05-31): Mixup augmentation covering all 3 input
        # modes: pure SEQUENCE_ONLY (mix padded sequences + lengths),
        # FEATURES_ONLY (mix aux_features), HYBRID (mix BOTH with the
        # same lam + idx so per-sample identity is preserved across
        # modalities). The padded-sequence path uses
        # ``mixup_sequence_batch`` which sets lengths = max(l_a, l_b)
        # so pack_padded_sequence drives the RNN over the union of the
        # two original sequences.
        _mixup_enabled = (
            getattr(self.config, "use_mixup", False)
            and self.training
            and batch["labels"].shape[0] >= 2
        )
        _mixup_active = False
        if _mixup_enabled:
            from ._mixup import mixup_batch, mixup_sequence_batch
            _alpha = getattr(self.config, "mixup_alpha", 0.2)
            batch = dict(batch)  # shallow copy so we don't mutate the loader's dict
            if batch.get("sequences") is not None:
                # F-70 sequence-aware path: SEQUENCE_ONLY or HYBRID
                seq_mixed, lens_mixed, aux_mixed, _y_a, _y_b, _lam = mixup_sequence_batch(
                    batch["sequences"], batch["lengths"], batch["labels"],
                    alpha=_alpha,
                    aux_features=batch.get("aux_features"),
                )
                batch["sequences"] = seq_mixed
                batch["lengths"] = lens_mixed
                if aux_mixed is not None:
                    batch["aux_features"] = aux_mixed
                _mixup_active = True
            elif batch.get("aux_features") is not None:
                # F-69 FEATURES_ONLY path
                aux_mixed, _y_a, _y_b, _lam = mixup_batch(
                    batch["aux_features"], batch["labels"], alpha=_alpha,
                )
                batch["aux_features"] = aux_mixed
                _mixup_active = True

        logits = self._forward_batch(batch)
        if _mixup_active:
            # Same task-specific loss split as MLP F-68: integer labels
            # use the two-target convex form, float labels mix targets.
            if batch["labels"].dtype in (torch.int32, torch.int64):
                loss_a = self._compute_weighted_loss(logits, _y_a, batch.get("sample_weights"))
                loss_b = self._compute_weighted_loss(logits, _y_b, batch.get("sample_weights"))
                loss = _lam * loss_a + (1.0 - _lam) * loss_b
            else:
                mixed_labels = _lam * _y_a + (1.0 - _lam) * _y_b
                loss = self._compute_weighted_loss(logits, mixed_labels, batch.get("sample_weights"))
        else:
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
            elif self.task_type == "multilabel":
                # Per-label sigmoid + 0.5-thresholded preds; both preds and target shape (N, K).
                preds = (torch.sigmoid(logits) >= 0.5).long()
                self.train_acc(preds, batch["labels"].long())
                self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
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
                self.log("val_mse", self.val_mse, prog_bar=True, on_step=False, on_epoch=True)
                # R2Score returns NaN when fewer than 2 samples are observed (variance undefined).
                # The metric accumulates across batches inside the epoch but the per-step update
                # also needs >=2 elements to keep total_sum_squares finite, otherwise the logged
                # val_r2 turns NaN and any downstream EarlyStopping monitoring it stalls silently.
                if preds.numel() >= 2:
                    self.val_r2(preds, batch["labels"])
                    self.log("val_r2", self.val_r2, prog_bar=True, on_step=False, on_epoch=True)
            elif self.task_type == "multilabel":
                # Both preds and target shape (N, K): thresholded preds for accuracy, raw sigmoid probs for AUROC / AUPRC.
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                self.val_acc(preds, batch["labels"].long())
                self.val_auroc(probs, batch["labels"].long())
                self.val_auprc(probs, batch["labels"].long())
                self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_auprc", self.val_auprc, on_step=False, on_epoch=True)
            else:
                probs = torch.softmax(logits, dim=1)
                if self.config.num_classes == 2:
                    preds = (probs[:, 1] >= 0.5).long()
                    self.val_acc(preds, batch["labels"])
                    self.val_auroc(probs[:, 1], batch["labels"])
                    self.val_auprc(probs[:, 1], batch["labels"])
                    self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
                else:
                    # Multiclass: argmax for class preds; pass full (N, K) probs to torchmetrics MulticlassAUROC (expects per-class scores).
                    preds = probs.argmax(dim=1)
                    self.val_acc(preds, batch["labels"])
                    self.val_auroc(probs, batch["labels"])
                    self.val_auprc(probs, batch["labels"])
                    self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
                    self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)
                    self.log("val_auprc", self.val_auprc, on_step=False, on_epoch=True)

    def predict_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Prediction step.

        - regression: squeeze last dim, return raw values
        - multilabel: per-label sigmoid (each output independent in [0, 1])
        - binary / multiclass: softmax over K outputs (sums to 1 per row)
        """
        logits = self._forward_batch(batch)
        if self.is_regression:
            return logits.squeeze(-1)
        if self.task_type == "multilabel":
            return torch.sigmoid(logits)
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
        """Compute loss with optional sample weights.

        Multilabel ``BCEWithLogitsLoss`` requires labels of dtype float32 (matching logits).
        The DataModule prepares 2-D labels for multilabel; cast here defensively in case caller passes int dtypes.
        """
        # Weighted reduction MUST divide by sum-of-weights, not N -- the
        # former is the unbiased weighted mean; the latter inflates the
        # loss by ``N / sum(w)`` whenever weights aren't uniform-mean-1.
        # Pre-fix all three branches used ``.mean()`` which divided by
        # ``N`` (or ``N*K`` for multilabel); with ``weights=[10, 0, 0]``
        # and ``losses=[1, 1, 1]`` that produced 10/3 в‰€ 3.33 instead of
        # the only-non-zero-weight sample's 1.0.
        _w_eps = torch.tensor(1e-12, dtype=logits.dtype, device=logits.device)
        if self.is_regression:
            preds = logits.squeeze(-1)
            if sample_weights is not None:
                losses = self._loss_fn_unreduced(preds, labels)
                w_sum = sample_weights.sum().clamp(min=_w_eps)
                return (losses * sample_weights).sum() / w_sum
            return self._loss_fn_mean(preds, labels)
        elif self.task_type == "multilabel":
            # BCEWithLogitsLoss expects float labels (same dtype as logits).
            labels_f = labels.to(logits.dtype)
            if sample_weights is not None:
                losses = self._loss_fn_unreduced(logits, labels_f)
                # losses shape: (N, K); broadcast sample_weights (N,) -> (N, 1).
                # Per-label mean across K, then weighted mean across N.
                per_sample = losses.mean(dim=-1)
                w_sum = sample_weights.sum().clamp(min=_w_eps)
                return (per_sample * sample_weights).sum() / w_sum
            return self._loss_fn_mean(logits, labels_f)
        else:
            if sample_weights is not None:
                losses = self._loss_fn_unreduced(logits, labels)
                w_sum = sample_weights.sum().clamp(min=_w_eps)
                return (losses * sample_weights).sum() / w_sum
            return self._loss_fn_mean(logits, labels)

    def configure_optimizers(self):
        """Configure AdamW with OneCycleLR scheduler.

        F-44 (2026-05-31): opt into fused AdamW kernel on CUDA. Skip under
        fp16-mixed -- Lightning AMP plugin can't reconcile fused-AdamW
        internal unscaling with the gradient-clipping pass that runs under
        a live GradScaler. bf16-mixed and fp32 are safe.
        """
        precision = "32-true"
        try:
            if self.trainer is not None:
                precision = str(self.trainer.precision)
        except Exception:
            pass
        # F-44 fused-AdamW + grad clipping under ANY mixed precision is
        # broken by Lightning's AMP plugin (not just 16-mixed): the plugin
        # routes through `_clip_gradients` which rejects fused optimizers
        # because they unscale internally. bf16-mixed has no GradScaler
        # but the same plugin path still fires the rejection. fp32 / no-AMP
        # is the only safe precision for fused. Observed 2026-05-31 with
        # CPU+EMA on a CUDA-available host: Lightning auto-promotes
        # 16-mixed -> bf16-mixed on CPU, then crashes on the first step.
        _safe_for_fused = (
            torch.cuda.is_available()
            and "mixed" not in precision
        )
        _fused_kwarg = {"fused": True} if _safe_for_fused else {}
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            **_fused_kwarg,
        )
        # F-62 (2026-05-31): optional Lookahead meta-optimizer wrap. Off by
        # default for the recurrent path; users can opt in via
        # RecurrentConfig.use_lookahead. Composes with F-44 fused AdamW.
        if getattr(self.config, "use_lookahead", False):
            from ._lookahead_optimizer import Lookahead
            optimizer = Lookahead(
                optimizer,
                k=getattr(self.config, "lookahead_k", 5),
                alpha=getattr(self.config, "lookahead_alpha", 0.5),
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

    def on_train_end(self) -> None:
        """F-B fix (2026-05-31, audit follow-up): if the optimizer is a
        Lookahead wrapper (F-62), commit slow weights to fast so predict()
        uses the anchor (per Zhang 2019 evaluation contract) rather than
        the mid-cycle fast exploration head. Without this hook a fit
        stopping on a non-k-th step leaves the live params at fast.
        """
        try:
            from ._lookahead_optimizer import Lookahead
            opts = self.optimizers()
            opt_iter = opts if isinstance(opts, (list, tuple)) else [opts]
            for opt in opt_iter:
                base = getattr(opt, "optimizer", opt)
                if isinstance(base, Lookahead):
                    base.commit_slow_to_fast()
        except Exception as _lh_err:
            logger.debug(
                "F-B: Lookahead commit_slow_to_fast skipped (%s)", _lh_err
            )
