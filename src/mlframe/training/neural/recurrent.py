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


def _monitor_mode(monitor: str) -> str:
    """Return 'min' or 'max' for a metric name without substring traps.

    Splits on '_' / '/' and inspects suffix tokens against an explicit
    min/max table. Falls back to 'max' (the historical default for non-loss
    names); unknown names emit a warning so the caller can extend the table
    rather than silently misdirect EarlyStopping.
    """
    _tokens = [t for t in monitor.replace("/", "_").split("_") if t]
    # Walk suffix-first so 'val_log_likelihood' -> 'likelihood' wins over 'log'.
    for tok in reversed(_tokens):
        if tok in _MONITOR_MIN_KEYS:
            return "min"
        if tok in _MONITOR_MAX_KEYS:
            return "max"
    logger.warning("_monitor_mode: unknown monitor %r; defaulting to 'max'", monitor)
    return "max"


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
        self._seq_input_size: int = _DEFAULT_SEQ_INPUT_SIZE
        self._feature_scaler: StandardScaler | None = None
        self._prediction_cache: dict[bytes, np.ndarray] = {}

    def _validate_inputs(
        self,
        features: np.ndarray | None,
        sequences: list[np.ndarray] | None,
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
        sequences: list[np.ndarray] | None,
        features: np.ndarray | None,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> RecurrentDataset:
        """Create dataset with proper preprocessing."""
        processed_seqs = None
        if sequences is not None:
            mode = getattr(self.config, "sequence_preprocessing", "none")
            if mode == "none":
                # Fast-path: cast to float32 and pass through unchanged.
                processed_seqs = [np.asarray(s, dtype=np.float32) for s in sequences]
            else:
                _preprocess = lambda s: _RecurrentWrapperBase._preprocess_sequence(s, mode=mode)
                # Threshold tuned for >100k sequences: ThreadPool overhead (thread spin-up + GIL
                # contention on numpy ops that release the GIL) only pays back at ~100k+ sequences.
                # Below that the synchronous loop is faster because the numpy std/mean kernels run
                # in C; the executor's per-task scheduling cost dominates.
                if len(sequences) > 100_000:
                    with ThreadPoolExecutor() as executor:
                        processed_seqs = list(executor.map(_preprocess, sequences))
                else:
                    processed_seqs = [_preprocess(seq) for seq in sequences]

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

    # Domain-specific scale only applied when the caller opts into
    # ``sequence_preprocessing="astronomy_mjd_delta"``. The generic
    # ``"none"`` and ``"per_sequence_zscore"`` modes never reference it.
    _DELTA_MJD_SCALE: float = 10.0
    _STD_EPSILON: float = 1e-8

    @staticmethod
    def _preprocess_sequence(seq: np.ndarray, mode: str = "none") -> np.ndarray:
        """Preprocess a single sequence.

        ``mode``:
          * ``"none"``: cast to float32 and return; magnitude preserved.
          * ``"per_sequence_zscore"``: independent z-score per channel per
            sequence. Destroys cross-sequence magnitude information; use
            only when each sequence is a standalone series whose
            absolute scale is irrelevant.
          * ``"astronomy_mjd_delta"``: legacy compat for in-house
            astronomy datasets. Column 0 is delta-encoded then scaled
            by ``_DELTA_MJD_SCALE``; columns 1+ get per-sequence
            z-score. Provided so existing astronomy callers don't break
            after the default flip.
        """
        result = seq.astype(np.float32)
        if mode == "none":
            return result

        n_cols = result.shape[1]
        if mode == "astronomy_mjd_delta":
            if n_cols > 0:
                delta = np.zeros(len(result), dtype=np.float32)
                delta[1:] = np.diff(seq[:, 0])
                result[:, 0] = delta / _RecurrentWrapperBase._DELTA_MJD_SCALE
            start_col = 1
        elif mode == "per_sequence_zscore":
            start_col = 0
        else:
            raise ValueError(
                f"_preprocess_sequence: unknown mode {mode!r}; expected one of "
                "{'none', 'per_sequence_zscore', 'astronomy_mjd_delta'}"
            )

        for col_idx in range(start_col, n_cols):
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

        # Stratified sampler: skip for multilabel (np.bincount fails on 2-D)
        # and for regression (continuous y has no class structure).
        # ``shape[1] >= 2`` (not just ``ndim == 2``): single-label targets
        # delivered as a 1-column 2-D array are still SINGLE-label; treating
        # them as multilabel suppresses the stratified sampler for what is
        # actually a stratifiable single-label classification dataset.
        _is_multilabel_ds = (dataset.labels.ndim == 2 and dataset.labels.shape[1] >= 2)
        if shuffle and self.config.use_stratified_sampler and not self._is_regression and not _is_multilabel_ds:
            labels = dataset.labels.numpy()
            # np.bincount needs non-negative contiguous integer labels and
            # silently misreports for non-contiguous label sets ({0,5} ->
            # length-6 array with zeros); use np.unique for correctness.
            unique_labels, class_counts = np.unique(labels, return_counts=True)
            if len(unique_labels) > 1 and (class_counts > 0).all():
                label_to_weight = {int(lbl): 1.0 / int(cnt) for lbl, cnt in zip(unique_labels, class_counts)}
                sample_weights = np.array([label_to_weight[int(lbl)] for lbl in labels], dtype=np.float64)
                # Seeded generator: WeightedRandomSampler without an explicit generator pulls from
                # the GLOBAL torch RNG, which means parallel processes / re-instantiated wrappers
                # produce different sample orders across runs. self.random_state pins the order.
                _gen_sampler = torch.Generator()
                _gen_sampler.manual_seed(int(self.random_state))
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
                    generator=_gen_sampler,
                )
                shuffle = False

        # DataLoader shuffle=True also uses the global torch RNG unless an explicit generator
        # is passed. Pin to self.random_state so two wrapper instances with the same seed produce
        # identical batch sequences (required by sklearn clone() round-trip reproducibility).
        _gen_dl = None
        if shuffle:
            _gen_dl = torch.Generator()
            _gen_dl.manual_seed(int(self.random_state))

        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
            generator=_gen_dl,
        )

    def _create_model(
        self,
        seq_input_size: int,
        aux_input_size: int,
        class_weight: dict[int, float] | None,
    ) -> RecurrentTorchModel:
        """Create model instance."""
        weight_tensor = None
        if class_weight and not self._is_regression:
            weight_tensor = torch.tensor(
                [class_weight.get(i, 1.0) for i in range(self.config.num_classes)],
                dtype=torch.float32,
            )

        # Thread task_type='multilabel' when fit() detected 2-D y; LightningModule switches to BCE loss + sigmoid output.
        _task_type = "multilabel" if getattr(self, "_is_multilabel", False) else None
        return RecurrentTorchModel(
            config=self.config,
            seq_input_size=seq_input_size,
            aux_input_size=aux_input_size,
            class_weight=weight_tensor,
            is_regression=self._is_regression,
            task_type=_task_type,
        )

    def _auto_precision(self) -> str:
        from ._recurrent_perf import auto_precision
        return auto_precision(self.config.precision)

    def _maybe_enable_cudnn_rnn_autotune(self) -> None:
        from ._recurrent_perf import maybe_enable_cudnn_rnn_autotune
        maybe_enable_cudnn_rnn_autotune(self.config.rnn_type)

    def _create_trainer(self, has_validation: bool, plot: bool) -> tuple[L.Trainer, Any]:
        """Create Lightning Trainer with callbacks."""
        callbacks: list = []
        checkpoint_callback = None

        if has_validation:
            monitor = self.config.early_stopping_monitor
            mode = _monitor_mode(monitor)

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

        # F-68 (2026-05-31): exponential moving average of weights. Mirrors
        # MLP's F-28. Off by default; enable via RecurrentConfig.use_ema.
        # When on, Lightning's WeightAveraging callback maintains an EMA
        # copy alongside the live model + swaps it in on on_train_end so
        # downstream predict() uses the averaged copy transparently. Falls
        # back to SWA-as-EMA (with constant swa_lrs = config.learning_rate
        # so no LR-restart phase) when WeightAveraging is unavailable in
        # the installed Lightning (<2.5).
        if getattr(self.config, "use_ema", False):
            from torch.optim.swa_utils import get_ema_avg_fn
            _ema_decay = getattr(self.config, "ema_decay", 0.999)
            try:
                from lightning.pytorch.callbacks import WeightAveraging
                callbacks.append(
                    WeightAveraging(avg_fn=get_ema_avg_fn(decay=_ema_decay))
                )
            except ImportError:
                from lightning.pytorch.callbacks import StochasticWeightAveraging
                callbacks.append(
                    StochasticWeightAveraging(
                        swa_lrs=self.config.learning_rate,
                        swa_epoch_start=0.5,
                        avg_fn=get_ema_avg_fn(decay=_ema_decay),
                    )
                )

        # CUDA-broken-host guard: probe a tiny allocation before Lightning
        # opens its CUDA strategy. On hosts with CUDA libs but a broken
        # runtime (CURAND init failure, illegal memory access on first
        # ``model_to_device``), the bare ``accelerator=self.config.accelerator``
        # path crashes during ``Trainer.fit``. Resolving via ``safe_accelerator``
        # downgrades ``cuda`` / ``gpu`` / ``auto`` to ``cpu`` when the probe
        # fails so the recurrent ensemble member finishes its fit instead of
        # poisoning the whole composite ensemble.
        from ._base_tensor_helpers import safe_accelerator
        trainer = L.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=safe_accelerator(self.config.accelerator),
            precision=self._auto_precision(),
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
        sequences: list[np.ndarray] | None,
    ) -> bytes:
        """Compute content-hash cache key from input arrays.

        Sampling 3 scalars + shape and hashing the tuple-of-str was
        collision-prone: any two predict-batches that agreed on (shape,
        dtype, first/middle/last value) returned the cached prediction of
        the first, which silently mis-predicted on near-duplicate inputs.

        Hash full ``tobytes()`` payload (xxhash if available, blake2b
        otherwise) keyed on shape+dtype to make sub-cell changes always
        invalidate the cache.
        """
        # Module-top imports (hashlib + optional xxhash) keep the predict-hot
        # path import-free; the previous try/except ImportError ran on EVERY
        # predict call. xxhash is ~5x faster than blake2b on tobytes payloads.
        if _HAS_XXHASH:
            _hasher = _xxhash.xxh3_128()
        else:
            _hasher = _hashlib.blake2b(digest_size=16)
        _update = _hasher.update
        _digest = _hasher.digest

        if features is not None:
            _update(b"FEAT")
            _update(str(features.shape).encode())
            _update(features.dtype.str.encode())
            _arr = np.ascontiguousarray(features)
            _update(_arr.tobytes())
        if sequences is not None:
            _update(b"SEQ")
            _update(str(len(sequences)).encode())
            for seq in sequences:
                _arr = np.ascontiguousarray(seq)
                _update(str(seq.shape).encode())
                _update(seq.dtype.str.encode())
                _update(_arr.tobytes())
        return _digest()

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Config is serialised as a primitive dict (Enum values -> .value
        strings) so the file loads under ``torch.load(weights_only=True)``;
        the prior format pickled the ``RecurrentConfig`` dataclass directly,
        which weights_only=True rejects as an unsafe global. feature_scaler
        likewise pickled to disk; we store its key attributes only.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")

        from dataclasses import fields

        _cfg = self.config
        config_dict: dict[str, Any] = {}
        for f in fields(_cfg):
            v = getattr(_cfg, f.name)
            if isinstance(v, Enum):
                v = v.value
            config_dict[f.name] = v

        scaler_dict: dict[str, Any] | None = None
        if self._feature_scaler is not None:
            scaler_dict = {
                "mean_": np.asarray(self._feature_scaler.mean_) if self._feature_scaler.mean_ is not None else None,
                "scale_": np.asarray(self._feature_scaler.scale_) if self._feature_scaler.scale_ is not None else None,
                "var_": np.asarray(self._feature_scaler.var_) if self._feature_scaler.var_ is not None else None,
                "n_features_in_": int(self._feature_scaler.n_features_in_),
                "with_mean": bool(self._feature_scaler.with_mean),
                "with_std": bool(self._feature_scaler.with_std),
            }

        state = {
            "config_dict": config_dict,
            "model_state_dict": self.model.state_dict(),
            "random_state": self.random_state,
            "aux_input_size": self._aux_input_size,
            "seq_input_size": self._seq_input_size,
            "feature_scaler_dict": scaler_dict,
            "is_regression": self._is_regression,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> _RecurrentWrapperBase:
        """Load model from disk (weights_only=True safe)."""
        state = torch.load(path, map_location="cpu", weights_only=True)

        # Reconstruct config from dict; coerce Enum string values back.
        _raw = state["config_dict"]
        _cfg_kwargs = dict(_raw)
        if isinstance(_cfg_kwargs.get("input_mode"), str):
            _cfg_kwargs["input_mode"] = InputMode(_cfg_kwargs["input_mode"])
        if isinstance(_cfg_kwargs.get("rnn_type"), str):
            _cfg_kwargs["rnn_type"] = RNNType(_cfg_kwargs["rnn_type"])
        config = RecurrentConfig(**_cfg_kwargs)

        wrapper = cls(config=config, random_state=state["random_state"])
        wrapper._aux_input_size = state.get("aux_input_size", 0)
        wrapper._seq_input_size = state.get("seq_input_size", _DEFAULT_SEQ_INPUT_SIZE)

        _sd = state.get("feature_scaler_dict")
        if _sd is not None:
            scaler = StandardScaler(with_mean=_sd["with_mean"], with_std=_sd["with_std"])
            if _sd["mean_"] is not None:
                scaler.mean_ = np.asarray(_sd["mean_"])
            if _sd["scale_"] is not None:
                scaler.scale_ = np.asarray(_sd["scale_"])
            if _sd["var_"] is not None:
                scaler.var_ = np.asarray(_sd["var_"])
            scaler.n_features_in_ = int(_sd["n_features_in_"])
            wrapper._feature_scaler = scaler

        wrapper.model = RecurrentTorchModel(
            config=config,
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
        sequences: list[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        class_weight: dict[int, float] | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
    ) -> RecurrentClassifierWrapper:
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

        # Detect multilabel from 2-D y: switches model to BCEWithLogitsLoss + sigmoid output.
        # Multilabel torchmetrics are skipped here (metrics come from the suite's downstream evaluation pipeline).
        # ``shape[1] >= 2`` lower bound: a single-label target delivered as
        # a 1-column 2-D array is still single-label - same gotcha as the
        # flat MLPClassifier in base.py (see commit + comment there); the
        # consequence here would be a misconfigured BCEWithLogitsLoss with
        # a num_classes=1 output head for what is actually multi-class
        # classification.
        self._is_multilabel = bool(
            hasattr(labels, "ndim") and labels.ndim == 2 and np.asarray(labels).shape[1] >= 2
        )
        if self._is_multilabel:
            self._n_labels = int(np.asarray(labels).shape[1])
            # Override config.num_classes to match label count so the MLP head builds the right number of output units.
            self.config.num_classes = self._n_labels

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
        self._seq_input_size = (
            sequences[0].shape[1]
            if sequences is not None and len(sequences) > 0
            else _DEFAULT_SEQ_INPUT_SIZE
        )

        self.model = self._create_model(
            seq_input_size=self._seq_input_size,
            aux_input_size=self._aux_input_size,
            class_weight=class_weight,
        )

        self.trainer, checkpoint_callback = self._create_trainer(val_loader is not None, plot)
        self._maybe_enable_cudnn_rnn_autotune()
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
            except Exception:
                # Wave 41 (2026-05-20): checkpoint-fallback to final-epoch model is a
                # quality regression source; preserve traceback for triage.
                logger.warning("Failed to load checkpoint, using final model", exc_info=True)

        return self

    def predict_proba(
        self,
        features: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
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
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("Model not trained. Call fit() first.")

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
            precision=self._auto_precision(),
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
        sequences: list[np.ndarray] | None = None,
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
        # Wave 40 (2026-05-20): argmax returns 0..(n_classes-1); int8 wraps class 128 -> -128
        # silently mis-classifying any class id > 127. Use the range-aware ladder shared by
        # extractors.intize_targets() instead of the hardcoded int8.
        classes = proba.argmax(axis=1)
        cmax = int(classes.max()) if classes.size else 0
        for _dt in (np.int8, np.int16, np.int32, np.int64):
            if cmax <= np.iinfo(_dt).max:
                return classes.astype(_dt)
        return classes


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
        sequences: list[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
    ) -> RecurrentRegressorWrapper:
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
        self._seq_input_size = (
            sequences[0].shape[1]
            if sequences is not None and len(sequences) > 0
            else _DEFAULT_SEQ_INPUT_SIZE
        )

        self.model = self._create_model(
            seq_input_size=self._seq_input_size,
            aux_input_size=self._aux_input_size,
            class_weight=None,  # No class weights for regression
        )

        self.trainer, checkpoint_callback = self._create_trainer(val_loader is not None, plot)
        self._maybe_enable_cudnn_rnn_autotune()
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
            except Exception:
                # Wave 41 (2026-05-20): checkpoint-fallback to final-epoch model is a
                # quality regression source; preserve traceback for triage.
                logger.warning("Failed to load checkpoint, using final model", exc_info=True)

        return self

    def predict(
        self,
        features: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
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
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("Model not trained. Call fit() first.")

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
            precision=self._auto_precision(),
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
    df: pl_df.DataFrame,
    indices: np.ndarray | list[int] | None = None,
    columns: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> list[np.ndarray]:
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

    # Convert each column's per-row list to a numpy array once. The previous
    # implementation did n_rows * n_cols Python list-lookups in a comprehension
    # then column_stack'd per row, materialising n_rows separate small (k, n_cols)
    # arrays via Python-level loops. Casting per column first lets each row's
    # stack use ndarray slicing rather than nested-list indexing.
    col_arrays: list[list[np.ndarray]] = [
        [np.asarray(v, dtype=np.float32) for v in df[col].to_list()]
        for col in columns
    ]

    result: list[np.ndarray] = [
        np.stack([col_arrays[j][i] for j in range(len(columns))], axis=-1)
        for i in range(n_rows)
    ]

    return result


def extract_sequences_chunked(
    df: pl_df.DataFrame,
    indices: np.ndarray | list[int] | None = None,
    chunk_size: int = 100_000,
    columns: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> list[np.ndarray]:
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

    sequences: list[np.ndarray] = []

    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        chunk_seqs = extract_sequences(df, chunk_indices.tolist(), columns)
        sequences.extend(chunk_seqs)

    return sequences
