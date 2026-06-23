from __future__ import annotations

"""
Sklearn-compatible recurrent model wrappers + EarlyStopping monitor-direction
helper, carved verbatim out of ``recurrent.py`` to keep that facade under the
1000-LOC budget (sibling re-export pattern; see mlframe/CLAUDE.md "Monolith
split"). The classes here (``_RecurrentWrapperBase`` /
``RecurrentClassifierWrapper`` / ``RecurrentRegressorWrapper``) are re-exported
from ``recurrent.py`` so existing
``from mlframe.training.neural.recurrent import RecurrentClassifierWrapper``
callers keep working. The LightningModule (``RecurrentTorchModel``) and the
Dataset/collate (``RecurrentDataset`` / ``recurrent_collate_fn``) live in their
own siblings and are imported here.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------------

import copy

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import lightning as L
import torch
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from concurrent.futures import ThreadPoolExecutor

from .base import _ensure_numpy
from ._recurrent_cat_embeddings import _RecurrentCatEmbeddingMixin
from ._recurrent_config import RNNType, InputMode, RecurrentConfig
from ._recurrent_data import RecurrentDataset, recurrent_collate_fn
from ._recurrent_torch_model import RecurrentTorchModel


# Default number of channels per timestep when callers don't supply
# sequences (FEATURES_ONLY mode keeps the RNN branch dormant). Kept as a
# module-level constant so the save/load round-trip and the wrapper
# initialisation agree on the same fallback.
_DEFAULT_SEQ_INPUT_SIZE: int = 4


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
# Sklearn Wrappers - Base
# ----------------------------------------------------------------------------------------------------------------------------


class _RecurrentWrapperBase(_RecurrentCatEmbeddingMixin, BaseEstimator):
    """
    Base class for sklearn-compatible recurrent model wrappers.

    Provides common functionality for both classification and regression.
    """

    _is_regression: bool = False

    def __init__(
        self,
        config: RecurrentConfig | None = None,
        random_state: int = 42,
        use_learnable_cat_embeddings: bool = True,
        categorical_embed_dim: int | None = None,
    ) -> None:
        # Store ``config`` VERBATIM (None stays None) so sklearn clone() round-trips the exact constructor arg. fit() never mutates this; it
        # mutates a fit-local deepcopy resolved via ``_cfg`` instead, so the shared module-level default RecurrentConfig() is never mutated.
        self.config = config
        self.random_state = random_state
        # ``use_learnable_cat_embeddings`` (default True): when fit() receives ``cat_features`` and the input_mode keeps a tabular block
        # (HYBRID / FEATURES_ONLY), factorize those raw cat columns to integer codes at the fit boundary and learn an ``nn.Embedding`` per cat
        # (trained end-to-end) on the TABULAR features, mirroring the flat MLP. SEQUENCE_ONLY has no tabular block so the path no-ops. Set False
        # to fall back to the strategy's CatBoostEncoder path (cats target-encoded upstream; the factorizer no-ops). ``categorical_embed_dim``:
        # fixed per-cat embedding width; None uses the fastai heuristic.
        self.use_learnable_cat_embeddings = use_learnable_cat_embeddings
        self.categorical_embed_dim = categorical_embed_dim
        self.model: RecurrentTorchModel | None = None
        self.trainer: L.Trainer | None = None
        self._aux_input_size: int = 0
        self._seq_input_size: int = _DEFAULT_SEQ_INPUT_SIZE
        self._feature_scaler: StandardScaler | None = None
        self._prediction_cache: dict[bytes, np.ndarray] = {}
        # Categorical factorization state (tabular block only; sequences never carry cats). Populated by ``_factorize_cats_fit`` and replayed by
        # ``_apply_cat_codes`` at predict. ``_cat_cardinalities_`` None == no learnable-cat embedding active for this fit.
        self._cat_code_maps_: dict | None = None
        self._cat_cols_: list | None = None
        self._cat_cardinalities_: list[int] | None = None
        self._n_cat_features_: int = 0

    @property
    def _cfg(self) -> RecurrentConfig:
        """Effective config: the fit-local deepcopy once resolved, else a fresh fallback from the verbatim ``self.config`` (never the shared default).

        Reads route here so fit-time mutations (num_classes, early_stopping_monitor) land on an instance-owned copy, leaving ``self.config``
        (the verbatim clone-source) and the module-level default RecurrentConfig() untouched. predict() before fit() resolves a fresh copy too.
        """
        resolved = getattr(self, "_cfg_resolved", None)
        if resolved is None:
            resolved = copy.deepcopy(self.config) if self.config is not None else RecurrentConfig()
            self._cfg_resolved = resolved
        return resolved

    def __getstate__(self) -> dict:
        # ``trainer`` is a live ``lightning.pytorch.Trainer`` (references a WarningCache the
        # mlframe save_load ``_SafeUnpickler`` allowlist blocks) and ``_prediction_cache``
        # holds per-call numpy arrays keyed by content hash -- both are runtime-only. Drop
        # them on serialise (the cache rebuilds lazily on the next predict, the trainer on
        # the next fit) and move the torch model to CPU so no CUDA tensor reaches the pickle.
        state = self.__dict__.copy()
        state["trainer"] = None
        state["_prediction_cache"] = {}
        model = state.get("model")
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.trainer = None
        self._prediction_cache = {}

    def _validate_inputs(
        self,
        features: np.ndarray | None,
        sequences: list[np.ndarray] | None,
    ) -> None:
        """Validate inputs match the configured input mode."""
        mode = self._cfg.input_mode

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
            mode = getattr(self._cfg, "sequence_preprocessing", "none")
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
            n_cat = self._n_cat_features_ if self._cat_cardinalities_ else 0
            if n_cat > 0:
                # Scale ONLY the trailing numeric block; the leading cat-code columns pass through unscaled so they stay valid embedding indices.
                scaled_features = np.ascontiguousarray(scaled_features, dtype=np.float32)
                if scaled_features.shape[1] > n_cat:
                    scaled_features[:, n_cat:] = self._feature_scaler.transform(scaled_features[:, n_cat:]).astype(np.float32)
            else:
                scaled_features = self._feature_scaler.transform(scaled_features).astype(np.float32)

        return RecurrentDataset(
            sequences=processed_seqs,
            aux_features=scaled_features,
            labels=labels,
            sample_weights=sample_weights,
            is_regression=self._is_regression,
        )

    def _create_eval_dataset(self, eval_set: tuple) -> RecurrentDataset:
        """Create validation dataset from eval_set tuple. Replays the fit-time cat factorization on the val tabular block (no-op when none)."""
        if len(eval_set) == 2:
            features, labels = eval_set
            return self._create_dataset(None, self._apply_cat_codes(features), labels)
        elif len(eval_set) == 3:
            sequences, features, labels = eval_set
            return self._create_dataset(sequences, self._apply_cat_codes(features), labels)
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
        if shuffle and self._cfg.use_stratified_sampler and not self._is_regression and not _is_multilabel_ds:
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

        # F-47 mirror: pin_memory tracks the trainer accelerator, not just
        # host CUDA availability. Pinning when the trainer runs on CPU (user
        # set ``accelerator="cpu"`` on a CUDA box for debugging) emits
        # "pin_memory is set as true but no accelerator is found" per batch.
        _pin_memory = (
            torch.cuda.is_available()
            and self._cfg.accelerator in ("auto", "gpu", "cuda")
        )
        return DataLoader(
            dataset,
            batch_size=batch_size or self._cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self._cfg.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=_pin_memory,
            persistent_workers=self._cfg.num_workers > 0,
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
                [class_weight.get(i, 1.0) for i in range(self._cfg.num_classes)],
                dtype=torch.float32,
            )

        # Thread task_type='multilabel' when fit() detected 2-D y; LightningModule switches to BCE loss + sigmoid output.
        _task_type = "multilabel" if getattr(self, "_is_multilabel", False) else None
        # Thread the fit-time tabular cat cardinalities so the model prepends a CategoricalEmbedding on the aux block. None when no cats were
        # factorized (or SEQUENCE_ONLY, where there is no aux block). The embedding lives BEFORE the shared MLPHead output, target-type-agnostic.
        _cat_cards = list(self._cat_cardinalities_) if self._cat_cardinalities_ else None
        return RecurrentTorchModel(
            config=self._cfg,
            seq_input_size=seq_input_size,
            aux_input_size=aux_input_size,
            class_weight=weight_tensor,
            is_regression=self._is_regression,
            task_type=_task_type,
            aux_categorical_cardinalities=_cat_cards,
            aux_categorical_embed_dim=self.categorical_embed_dim,
        )

    def _auto_precision(self) -> str:
        from ._recurrent_perf import auto_precision
        return auto_precision(self._cfg.precision)

    def _maybe_enable_cudnn_rnn_autotune(self) -> None:
        from ._recurrent_perf import maybe_enable_cudnn_rnn_autotune
        maybe_enable_cudnn_rnn_autotune(self._cfg.rnn_type)

    def _create_trainer(self, has_validation: bool, plot: bool) -> tuple[L.Trainer, Any]:
        """Create Lightning Trainer with callbacks."""
        callbacks: list = []
        checkpoint_callback = None

        if has_validation:
            monitor = self._cfg.early_stopping_monitor
            mode = _monitor_mode(monitor)

            callbacks.append(
                L.pytorch.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=self._cfg.early_stopping_patience,
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
        if getattr(self._cfg, "use_ema", False):
            from torch.optim.swa_utils import get_ema_avg_fn
            _ema_decay = getattr(self._cfg, "ema_decay", 0.999)
            try:
                from lightning.pytorch.callbacks import WeightAveraging
                callbacks.append(
                    WeightAveraging(avg_fn=get_ema_avg_fn(decay=_ema_decay))
                )
            except ImportError:
                from lightning.pytorch.callbacks import StochasticWeightAveraging
                callbacks.append(
                    StochasticWeightAveraging(
                        swa_lrs=self._cfg.learning_rate,
                        swa_epoch_start=0.5,
                        avg_fn=get_ema_avg_fn(decay=_ema_decay),
                    )
                )

        # CUDA-broken-host guard: probe a tiny allocation before Lightning
        # opens its CUDA strategy. On hosts with CUDA libs but a broken
        # runtime (CURAND init failure, illegal memory access on first
        # ``model_to_device``), the bare ``accelerator=self._cfg.accelerator``
        # path crashes during ``Trainer.fit``. Resolving via ``safe_accelerator``
        # downgrades ``cuda`` / ``gpu`` / ``auto`` to ``cpu`` when the probe
        # fails so the recurrent ensemble member finishes its fit instead of
        # poisoning the whole composite ensemble.
        from ._base_tensor_helpers import safe_accelerator
        trainer = L.Trainer(
            max_epochs=self._cfg.max_epochs,
            accelerator=safe_accelerator(self._cfg.accelerator),
            precision=self._auto_precision(),
            callbacks=callbacks,
            gradient_clip_val=self._cfg.gradient_clip_val,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=plot,
            deterministic="warn",
        )
        return trainer, checkpoint_callback

    def _clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()

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

        # Serialise the EFFECTIVE (fit-resolved) config, not the verbatim ``self.config`` -- fit-time mutations (num_classes, monitor) must survive load.
        _cfg = self._cfg
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
            # Learnable-cat-embedding state for the tabular block. ``cat_code_maps`` keys/values are plain Python (str/int) so they survive
            # ``torch.load(weights_only=True)``; the cardinalities drive the model's aux CategoricalEmbedding reconstruction at load.
            "use_learnable_cat_embeddings": bool(self.use_learnable_cat_embeddings),
            "categorical_embed_dim": self.categorical_embed_dim,
            "cat_code_maps": self._cat_code_maps_,
            "cat_cols": self._cat_cols_,
            "cat_cardinalities": self._cat_cardinalities_,
            "n_cat_features": self._n_cat_features_,
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

        wrapper = cls(
            config=config,
            random_state=state["random_state"],
            use_learnable_cat_embeddings=state.get("use_learnable_cat_embeddings", True),
            categorical_embed_dim=state.get("categorical_embed_dim"),
        )
        wrapper._aux_input_size = state.get("aux_input_size", 0)
        wrapper._seq_input_size = state.get("seq_input_size", _DEFAULT_SEQ_INPUT_SIZE)
        wrapper._cat_code_maps_ = state.get("cat_code_maps")
        wrapper._cat_cols_ = state.get("cat_cols")
        wrapper._cat_cardinalities_ = state.get("cat_cardinalities")
        wrapper._n_cat_features_ = state.get("n_cat_features", 0)

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
            aux_categorical_cardinalities=list(wrapper._cat_cardinalities_) if wrapper._cat_cardinalities_ else None,
            aux_categorical_embed_dim=wrapper.categorical_embed_dim,
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
        cat_features: list[str] | None = None,
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
            cat_features: Tabular categorical column names to factorize + learn entity embeddings for (HYBRID / FEATURES_ONLY). No-op in
                SEQUENCE_ONLY (no tabular block) or when ``use_learnable_cat_embeddings`` is False.

        Returns:
            self for method chaining
        """
        if labels is None:
            raise ValueError("labels is required")

        # Resolve a fresh fit-local config copy: every mutation below lands here, leaving the verbatim ``self.config`` (clone source) untouched.
        self._cfg_resolved = copy.deepcopy(self.config) if self.config is not None else RecurrentConfig()

        # Record the label set so predict/predict_proba map argmax positions back to the ORIGINAL labels (sklearn ClassifierMixin contract).
        # Single-label targets are also ENCODED to contiguous 0..k-1 positions before training: the CrossEntropy head indexes by position, so a
        # raw label like 9 with a 3-output head would be out-of-bounds. predict() inverts the mapping via ``classes_``.
        _labels_arr = np.asarray(labels)
        _is_single_label = not (hasattr(labels, "ndim") and _labels_arr.ndim == 2 and _labels_arr.shape[1] >= 2)
        if _is_single_label:
            self.classes_ = np.unique(_labels_arr)
            labels = np.searchsorted(self.classes_, _labels_arr).astype(np.int64)
            # Size the output head to the observed class count so the CrossEntropy positions are always in range.
            self._cfg.num_classes = int(self.classes_.shape[0])

        # Factorize tabular cat columns to int codes (reordered leading) BEFORE the scaler fit + dataset build, so the learnable aux
        # CategoricalEmbedding can index them and the scaler skips the code columns. Scopes to ``features`` only; sequences are untouched.
        features = self._factorize_cats_fit(features, cat_features)

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
            self._cfg.num_classes = self._n_labels

        self._validate_inputs(features, sequences)
        self._clear_cache()

        if self._cfg.scale_features and features is not None:
            # Numeric-only scaler fit: skips the leading cat-code columns (scaling embedding indices would corrupt them).
            self._scaler_fit_numeric_only(features)

        L.seed_everything(self.random_state, workers=True)

        train_dataset = self._create_dataset(sequences, features, labels, sample_weight)
        # Encode the eval_set labels through the same ``classes_`` mapping so the val CrossEntropy head sees in-range 0..k-1 positions too.
        if eval_set is not None and _is_single_label:
            eval_set = (*eval_set[:-1], np.searchsorted(self.classes_, np.asarray(eval_set[-1])).astype(np.int64))
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
                    config=self._cfg,
                    seq_input_size=self._seq_input_size,
                    aux_input_size=self._aux_input_size,
                    is_regression=False,
                    aux_categorical_cardinalities=list(self._cat_cardinalities_) if self._cat_cardinalities_ else None,
                    aux_categorical_embed_dim=self.categorical_embed_dim,
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

        # Replay the fit-time cat factorization + coerce to float32 ndarray BEFORE the cache key (the key reads ``.dtype``, a DataFrame lacks it)
        # and BEFORE dataset construction. No-op when no cats were factorized; ``None`` (SEQUENCE_ONLY) passes through.
        features = self._prepare_predict_features(features)

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
            accelerator=self._cfg.accelerator,
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
        positions = proba.argmax(axis=1)
        # Map argmax POSITIONS back through ``classes_`` so the returned labels are the original ones (sklearn ClassifierMixin contract); a raw
        # argmax would return 0..(k-1) positional indices, mislabelling any non-0..k-1 label set and breaking cross_val_predict / CalibratedClassifierCV / Stacking.
        classes_ = getattr(self, "classes_", None)
        if classes_ is None:
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("Model not trained. Call fit() first.")
        return np.asarray(classes_)[positions]


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

    def fit(
        self,
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
        cat_features: list[str] | None = None,
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
            cat_features: Tabular categorical column names to factorize + learn entity embeddings for (HYBRID / FEATURES_ONLY). No-op in
                SEQUENCE_ONLY (no tabular block) or when ``use_learnable_cat_embeddings`` is False.

        Returns:
            self for method chaining
        """
        if labels is None:
            raise ValueError("labels is required")

        # Resolve a fresh fit-local config copy: mutations below land here, leaving the verbatim ``self.config`` (clone source) untouched.
        self._cfg_resolved = copy.deepcopy(self.config) if self.config is not None else RecurrentConfig()
        # Regression has no AUPRC; redirect the classification default monitor to val_loss on the fit-local copy (never the verbatim config).
        if self._cfg.early_stopping_monitor == "val_auprc":
            self._cfg.early_stopping_monitor = "val_loss"

        # Factorize tabular cat columns to int codes (reordered leading) BEFORE the scaler fit + dataset build (sequences untouched).
        features = self._factorize_cats_fit(features, cat_features)

        self._validate_inputs(features, sequences)
        self._clear_cache()

        if self._cfg.scale_features and features is not None:
            self._scaler_fit_numeric_only(features)

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
                    config=self._cfg,
                    seq_input_size=self._seq_input_size,
                    aux_input_size=self._aux_input_size,
                    is_regression=True,
                    aux_categorical_cardinalities=list(self._cat_cardinalities_) if self._cat_cardinalities_ else None,
                    aux_categorical_embed_dim=self.categorical_embed_dim,
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

        # Replay the fit-time cat factorization + coerce to float32 ndarray BEFORE the cache key (the key reads ``.dtype``, a DataFrame lacks it)
        # and BEFORE dataset construction. No-op when no cats were factorized; ``None`` (SEQUENCE_ONLY) passes through.
        features = self._prepare_predict_features(features)

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
            accelerator=self._cfg.accelerator,
            precision=self._auto_precision(),
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        predictions = predict_trainer.predict(self.model, loader)

        result = torch.cat(predictions, dim=0).float().cpu().numpy().astype(np.float32)

        self._prediction_cache[cache_key] = result
        return result
