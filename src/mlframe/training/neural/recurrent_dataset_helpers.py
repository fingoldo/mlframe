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

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------------

import copy
import os

from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import lightning as L
import torch
from sklearn.base import BaseEstimator
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

# LRU cap for _RecurrentWrapperBase._prediction_cache, mirroring the flat MLP's
# _CUDA_GRAPH_PREDICT_CACHE_MAX pattern (_flat_torch_module/_flat_torch_predict_accel.py). Without a cap,
# a permutation-importance-style loop issuing many distinct predict()/predict_proba() calls per fit
# accumulates one full-size cached array per distinct call for the whole lifetime of the (possibly
# long-lived) fitted estimator -- unbounded host RAM growth.
_PREDICTION_CACHE_MAX = max(1, int(os.environ.get("MLFRAME_RECURRENT_PREDICTION_CACHE_MAX", "16")))


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
        self._prediction_cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
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
        state["_prediction_cache"] = OrderedDict()
        model = state.get("model")
        if model is not None:
            try:
                model.cpu()
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in recurrent_dataset_helpers.py:160: %s", e)
                pass
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.trainer = None
        self._prediction_cache = OrderedDict()

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

    def _aux_features_emptied_by_selection(self, features) -> bool:
        """True when feature selection (MRMR / RFECV / constant-column drop) removed every aux column for a mode that consumes them.

        A 0-column tabular frame is unfittable here exactly as it is for the booster path: ``_scaler_fit_numeric_only`` hands an ``(n, 0)``
        array to ``StandardScaler.fit`` which raises ``ValueError: Found array with 0 sample(s) (shape=(0, 0))``. SEQUENCE_ONLY carries its
        signal in ``sequences`` and runs fine with no aux features, so the guard only fires for FEATURES_ONLY / HYBRID. Mirrors the suite-level
        0-feature skip in ``_training_loop._train_model_with_fallback`` so the caller's ``model is None -> skip`` degradation handles it
        instead of the fit aborting (or being logged as an exception traceback) when an upstream selector empties every column.
        """
        if self._cfg.input_mode == InputMode.SEQUENCE_ONLY:
            return False
        return features is not None and hasattr(features, "shape") and len(getattr(features, "shape", ())) == 2 and features.shape[1] == 0

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
                def _preprocess(s):
                    """Apply ``_preprocess_sequence`` under ``mode`` to a single sequence; module-level target for the optional ThreadPoolExecutor fan-out below."""
                    return _RecurrentWrapperBase._preprocess_sequence(s, mode=mode)
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

    def _create_eval_dataset(self, eval_set: tuple, sample_weight: np.ndarray | None = None) -> RecurrentDataset:
        """Create validation dataset from eval_set tuple. Replays the fit-time cat factorization on the val tabular block (no-op when none).

        ``sample_weight`` (the wrapper's ``eval_sample_weight`` fit() kwarg) is threaded straight into the
        val ``RecurrentDataset`` so ``validation_step``'s ``batch.get("sample_weights")`` actually has
        something to read.
        """
        if len(eval_set) == 2:
            features, labels = eval_set
            return self._create_dataset(None, self._apply_cat_codes(features), labels, sample_weight)
        elif len(eval_set) == 3:
            sequences, features, labels = eval_set
            return self._create_dataset(sequences, self._apply_cat_codes(features), labels, sample_weight)
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
            raise ValueError(f"_preprocess_sequence: unknown mode {mode!r}; expected one of " "{'none', 'per_sequence_zscore', 'astronomy_mjd_delta'}")

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
        _is_multilabel_ds = dataset.labels.ndim == 2 and dataset.labels.shape[1] >= 2
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
                    weights=cast(Any, sample_weights),
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
        # An explicit RecurrentConfig(pin_memory=...) always wins over the auto-detected default;
        # MLFRAME_MLP_PIN_MEMORY (if set) wins over auto-detect but not over that explicit value --
        # some driver/CUDA-toolkit combos crash during pinned-memory teardown, which GPU
        # auto-detection can't see, and this env var is the one-shot escape hatch for a whole run.
        if self._cfg.pin_memory is not None:
            _pin_memory = self._cfg.pin_memory
        else:
            from mlframe.training.mlp_runtime_defaults import pin_memory_env_override

            _env_pin = pin_memory_env_override()
            _pin_memory = _env_pin if _env_pin is not None else (torch.cuda.is_available() and self._cfg.accelerator in ("auto", "gpu", "cuda"))
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
        """Resolve the configured precision string to the actual value Lightning's ``Trainer`` should use (e.g. mixed-precision auto-detection)."""
        from ._recurrent_perf import auto_precision
        return auto_precision(self._cfg.precision)

    def _maybe_enable_cudnn_rnn_autotune(self) -> "bool | None":
        """Turn on cuDNN's RNN autotuner when the configured RNN type benefits from it; no-op otherwise.

        Returns the prior ``torch.backends.cudnn.benchmark`` value (or ``None`` if unchanged) -- the
        caller MUST restore it in a ``finally`` around ``trainer.fit()`` so this process-global flag
        doesn't leak into unrelated models trained later in the same process.
        """
        from ._recurrent_perf import maybe_enable_cudnn_rnn_autotune
        return maybe_enable_cudnn_rnn_autotune(self._cfg.rnn_type)

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
            from torch.optim.swa_utils import get_ema_avg_fn  # type: ignore[attr-defined]
            _ema_decay = getattr(self._cfg, "ema_decay", 0.999)
            try:
                from lightning.pytorch.callbacks import WeightAveraging  # type: ignore[attr-defined]

                callbacks.append(WeightAveraging(avg_fn=get_ema_avg_fn(decay=_ema_decay)))
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
            precision=cast(Any, self._auto_precision()),
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

    def _cache_get(self, cache_key: bytes) -> "np.ndarray | None":
        """LRU-aware cache lookup: a hit moves the entry to the most-recently-used end."""
        if cache_key not in self._prediction_cache:
            return None
        self._prediction_cache.move_to_end(cache_key)
        return self._prediction_cache[cache_key]

    def _cache_put(self, cache_key: bytes, result: np.ndarray) -> None:
        """Insert a prediction result, evicting the least-recently-used entry once over the cap."""
        self._prediction_cache[cache_key] = result
        self._prediction_cache.move_to_end(cache_key)
        while len(self._prediction_cache) > _PREDICTION_CACHE_MAX:
            self._prediction_cache.popitem(last=False)

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


# X_EFFICIENCY_ARCHITECTURE-1 fix (mrmr_audit_2026-07-22): RecurrentClassifierWrapper /
# RecurrentRegressorWrapper carved out into _recurrent_wrappers.py to clear the repo's enforced hard
# 1000-LOC CI gate (this file was 1049 lines). Re-exported here so existing callers keep working.
from ._recurrent_wrappers import RecurrentClassifierWrapper, RecurrentRegressorWrapper  # noqa: F401
