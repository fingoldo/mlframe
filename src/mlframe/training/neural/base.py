"""
Base infrastructure for PyTorch Lightning models in mlframe.

This module provides:
- sklearn-compatible estimator wrappers (PytorchLightningEstimator, Regressor, Classifier)
- Callbacks (BestEpochModelCheckpoint, AggregatingValidationCallback, etc.)
- Utilities (MetricSpec, to_tensor_any, to_numpy_safe)
"""

from __future__ import annotations


# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# Silence the trio of INFO bullets Lightning emits on every trainer init ("GPU available: True ... TPU/IPU/HPU available: False ...
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]"). With composite-target discovery + multi-model suites these emit 5+ lines per fit,
# drowning the real signal. Useful messages from the same loggers (``Time limit reached``, ``Metric val_MSE improved``, ``Loading
# best model``) are PRESERVED via a substring filter rather than blanket WARNING bump.
class _LightningRankZeroNoiseFilter(logging.Filter):
    """Drop the device-availability bullets that Lightning emits on every trainer init; let everything else through."""

    _PATTERNS = (
        "GPU available",
        "TPU available",
        "IPU available",
        "HPU available",
        "LOCAL_RANK:",
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)


_LIGHTNING_NOISE_FILTER = _LightningRankZeroNoiseFilter()
for _quiet_logger_name in (
    "lightning.pytorch.utilities.rank_zero",
    "lightning.pytorch.accelerators.cuda",
):
    _quiet_logger = logging.getLogger(_quiet_logger_name)
    # Idempotent attach: skip if already filtered on a prior module import.
    if not any(isinstance(f, _LightningRankZeroNoiseFilter) for f in _quiet_logger.filters):
        _quiet_logger.addFilter(_LIGHTNING_NOISE_FILTER)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

# stdlib
import os
import operator  # picklable comparison functions (needed by ddp_spawn strategy)
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

# third-party
import numpy as np
import pandas as pd
import polars as pl
import torch
import lightning as L
from pydantic import BaseModel
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateFinder,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping as EarlyStoppingCallback
from lightning.pytorch.loggers import CSVLogger
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error

# Lightning's data_connector emits a UserWarning ("does not have many workers
# which may be a bottleneck. Consider increasing the value of the num_workers
# argument...") for every train/val/predict DataLoader. That recommendation
# is wrong for mlframe: num_workers > 0 pickles the entire TorchDataset (which
# holds the full Polars/pandas frame) into every worker - catastrophic on
# 100+ GB production frames. Empirical bench at
# _benchmarks/bench_dataloader_workers.py confirms num_workers=0 is the best
# default on Windows/8-core for 3 of 4 measured shapes. Silence the warning
# so the log stays focused on actionable diagnostics.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
)

# local
from pyutilz.pythonlib import get_parent_func_args, store_params_in_object
from mlframe.metrics.core import compute_probabilistic_multiclass_error


# ----------------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------------


def _rmse_metric(y_true, y_score):
    """Wrapper for root_mean_squared_error that accepts y_score parameter name."""
    return root_mean_squared_error(y_true=y_true, y_pred=y_score)


# ----------------------------------------------------------------------------------------------------------------------------
# MetricSpec
# ----------------------------------------------------------------------------------------------------------------------------


class MetricSpec(BaseModel):
    name: str
    fcn: Callable  # the metric function
    requires_argmax: bool = False  # True if metric wants predicted class labels
    requires_probs: bool = False  # True if metric wants probabilities (softmax)
    requires_cpu: bool = True  # True if metric should run on CPU (sklearn), False if GPU-compatible


def custom_collate_fn(batch):
    """Identity collate: hands the raw batch list through unchanged.

    DataLoader's default collate stacks each sample's tensors into a
    batched tensor. Some datasets in this package yield non-tensor
    objects (e.g. variable-length sequences pre-collated by their own
    helpers) where that default raises. Passing this collate selects
    "don't touch the batch" and lets the consumer handle structure.
    Equivalent to ``lambda x: x`` but defined as a top-level callable
    so it can be pickled for multi-worker DataLoaders.
    """
    return batch


def to_tensor_any(data, dtype=torch.float32, device=None, safe=True):
    """
    Converts pandas / polars / numpy / torch input to a torch.Tensor
    with minimal copies and correct dtype.

    If safe=True, ignores categorical/object columns gracefully.
    """

    # --- Pandas
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
    # --- Polars
    elif isinstance(data, pl.DataFrame):
        data = data.to_torch()
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    return data.to(dtype=dtype, device=device)


def to_numpy_safe(tensor: torch.Tensor, cpu: bool = False) -> np.ndarray:
    """Convert a torch.Tensor to a NumPy array safely and efficiently.

    - Moves tensor to CPU if needed.
    - Converts unsupported dtypes (bfloat16, float16) to float32.
    - Keeps dtype otherwise unchanged (no accidental downcast).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    t = tensor.detach()
    if cpu and t.device.type != "cpu":
        t = t.cpu()

    # NumPy-incompatible dtypes
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    elif t.dtype == torch.complex32:
        t = t.to(torch.complex64)

    return t.numpy()


def _ensure_numpy(arr, dtype: np.dtype = np.float32) -> np.ndarray | None:
    """Convert DataFrame/Series/array-like to numpy array; passes None through."""
    if arr is None:
        return None
    if hasattr(arr, "to_numpy"):  # Polars DataFrame/Series
        return arr.to_numpy().astype(dtype)
    if hasattr(arr, "values"):  # Pandas DataFrame/Series
        return arr.values.astype(dtype)
    return np.asarray(arr, dtype=dtype)


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn compatibility
# ----------------------------------------------------------------------------------------------------------------------------


class PytorchLightningEstimator(BaseEstimator):
    """Wrapper that allows Pytorch Lightning model, datamodule and trainer to participate in sklearn pipelines.
    Supports early stopping (via eval_set in fit_params).
    """

    def __init__(
        self,
        model_class: object,
        model_params: dict,
        network_params: dict,
        datamodule_class: object,
        datamodule_params: dict,
        trainer_params: object,
        use_swa: bool = False,
        swa_params: dict = None,
        tune_params: bool = False,
        tune_batch_size: bool = False,
        float32_matmul_precision: str = None,
        early_stopping_rounds: int = 100,
    ):
        # Don't modify swa_params here (e.g., `swa_params or {}`) because sklearn's clone() requires constructor parameters not be
        # modified. Handle None later.
        store_params_in_object(obj=self, params=get_parent_func_args())

    def _fit_common(
        self,
        X,
        y,
        eval_set: tuple = (None, None),
        is_partial_fit: bool = False,
        classes: Optional[np.ndarray] = None,
        fit_params: dict = None,
        sample_weight=None,
    ):
        """Common logic for fit and partial_fit."""
        # Lazy import to avoid circular dependency
        from .flat import generate_mlp

        if fit_params is None:
            fit_params = {}

        # Enable TF32 for float32 matmul if on GPU.
        if self.float32_matmul_precision and torch.cuda.is_available():
            _allowed_matmul = ("highest", "high", "medium")
            if self.float32_matmul_precision not in _allowed_matmul:
                raise ValueError(
                    f"float32_matmul_precision must be one of {_allowed_matmul}, "
                    f"got {self.float32_matmul_precision!r}"
                )
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(self.float32_matmul_precision)
                logger.info("Enabled float32_matmul_precision=%s", self.float32_matmul_precision)

        has_validation = eval_set[0] is not None

        eval_sample_weight = fit_params.get("eval_sample_weight")

        # Multilabel detection must precede datamodule construction: the per-fit
        # ``labels_dtype`` override (int64 -> float32 for BCEWithLogitsLoss) is
        # applied at datamodule time, so it MUST be decided before the dm is
        # built. The earlier code did the check after dm construction, which
        # silently fed int64 labels into a CE-loss model, producing the (3) vs
        # (65536) shape mismatch observed in fuzz combo c0030 (2026-05-20).
        _is_multilabel_target = False
        if isinstance(self, ClassifierMixin):
            _y_check = y.values if isinstance(y, pd.Series) else y
            _y_check = np.asarray(_y_check) if not isinstance(_y_check, np.ndarray) else _y_check
            _is_multilabel_target = bool(_y_check.ndim == 2 and _y_check.shape[1] >= 2)

        _local_dm_params = dict(self.datamodule_params)
        if _is_multilabel_target:
            # BCEWithLogitsLoss requires float labels; CrossEntropyLoss (the
            # classifier default in helpers.py) requires Long. The estimator
            # owns the dispatch — datamodule just delivers the dtype the loss
            # expects.
            _local_dm_params["labels_dtype"] = torch.float32

        dm = self.datamodule_class(
            train_features=X,
            train_labels=y,
            train_sample_weight=sample_weight,
            val_features=eval_set[0],
            val_labels=eval_set[1],
            val_sample_weight=eval_sample_weight,
            **_local_dm_params,
        )
        # Stash for predict-time reuse so we don't re-instantiate (and trigger the
        # "No datamodule found from training. Creating temporary datamodule for
        # prediction." misleading warning at every predict call).
        self.prediction_datamodule = dm

        if isinstance(self, ClassifierMixin):
            # Multilabel was already detected upstream (``_is_multilabel_target``)
            # so the datamodule could swap labels_dtype to float32 in time. The
            # K >= 2 lower bound matters: a single-column 1-D-ish 2-D target
            # (N, 1) is still SINGLE-LABEL classification (the upstream just
            # delivered it as a 1-column frame instead of a 1-D array).
            # Treating it as multilabel sets num_classes=1, so MLP gets
            # output_dim=1, predictions squeeze to (N,), labels also squeeze
            # to (N,), then CrossEntropyLoss interprets predictions.shape ==
            # labels.shape as the class-probabilities input mode and rejects
            # Long labels with ``Expected floating point type for target with
            # class probabilities, got Long``. Observed 2026-05-20 on S: in
            # fuzz_3way combo cb_lgb_mlp_xgb-pl_nullable-n1000 binary
            # classification.
            self._is_multilabel = _is_multilabel_target

            if self._is_multilabel:
                _y_check = y.values if isinstance(y, pd.Series) else y
                _y_check = np.asarray(_y_check) if not isinstance(_y_check, np.ndarray) else _y_check
                self.n_labels_ = int(_y_check.shape[1])
                self.classes_ = None  # sentinel; predict_proba returns per-label sigmoid probs
                num_classes = self.n_labels_
            else:
                if is_partial_fit and classes is not None:
                    self.classes_ = np.asarray(classes)
                elif not hasattr(self, "classes_"):
                    # Must be ndarray (not list) for numpy fancy indexing in evaluation.py::report_probabilistic_model_perf
                    # (line ``preds = model.classes_[preds]`` fails on list + ndarray index). Sklearn convention is classes_ ndarray.
                    _y_arr = (y.unique() if isinstance(y, pd.Series) else np.unique(y))
                    # Wave 61 (2026-05-20): object-dtype y (mixed-type label set
                    # incl. None / np.nan + str) would TypeError on Python sorted();
                    # use np.sort for ndarrays and str-key fallback for object dtype.
                    if hasattr(_y_arr, "dtype") and _y_arr.dtype != object:
                        self.classes_ = np.sort(_y_arr)
                    else:
                        self.classes_ = np.asarray(sorted(_y_arr, key=lambda v: (v is None, str(v))))
                num_classes = len(self.classes_)
        else:
            num_classes = 1
            self._is_multilabel = False

        # Reset network on fit() to match sklearn convention (fit resets, partial_fit continues). Each fit() call must create a
        # fresh network with correct input dimensions; critical when feature counts change between training iterations.
        if not is_partial_fit:
            self.network = None
            self.model = None  # also reset the LightningModule wrapper

        # getattr handles freshly cloned models that don't have network attribute yet
        if getattr(self, 'network', None) is None:
            self.network = generate_mlp(num_features=X.shape[1], num_classes=num_classes, **self.network_params)

        if num_classes > 1:
            metric_name = "ICE"
            metrics = [MetricSpec(name=metric_name, fcn=compute_probabilistic_multiclass_error, requires_probs=True)]
        else:
            metric_name = "MSE"
            metrics = [MetricSpec(name=metric_name, fcn=_rmse_metric)]

        # When no validation data, monitor train_loss instead of train metrics (which may not be logged)
        if has_validation:
            monitor_metric = f"val_{metric_name}"
        else:
            monitor_metric = "train_loss"

        # Nest checkpoints + lightning_logs under a unique per-fit subdir so concurrent / sequential fits don't dump into a shared
        # project-root ``logs/`` folder and resolve different runs only by the (unsafe) ``model-val_MSE=0.7555.ckpt`` filename
        # collision via Lightning's version counter.
        #
        # Path resolution (in order of preference):
        #   1. ``self.checkpoint_dir_override`` - public attribute the suite sets to a target-nested path (eg
        #      ``data/models/{target}/{exp}/regression/{tgt}/{model_file_basename}/``). Honoured verbatim.
        #   2. Auto-derived ``{default_root_dir}/_run_{id(self)}_{ts}`` - unique sub-dir under the root; fully isolates concurrent
        #      runs even when no caller plumbing.
        # ``CSVLogger`` save_dir resolved the same way - mirror nesting so the on-disk layout stays uniform per fit.
        _ckpt_root = getattr(self, "checkpoint_dir_override", None)
        if _ckpt_root is None:
            import time as _time
            # Wave 46 (2026-05-20): trainer_params["default_root_dir"] is caller-controlled
            # per the standard Lightning Trainer contract. Caller is responsible for any
            # trusted-root validation upstream; this join is intentionally permissive and
            # matches Lightning's documented behaviour for default_root_dir.
            _default_root = self.trainer_params.get("default_root_dir") or "logs"
            _ckpt_root = os.path.join(_default_root, f"_run_{id(self)}_{int(_time.time())}")
        os.makedirs(_ckpt_root, exist_ok=True)

        checkpointing = BestEpochModelCheckpoint(
            monitor=monitor_metric,
            dirpath=_ckpt_root,
            # Filename no longer needs the ``model-`` prefix - the enclosing dir already identifies the model uniquely.
            filename=f"{{{monitor_metric}:.4f}}",
            enable_version_counter=True,
            save_last=False,
            save_top_k=1,
            mode="min",
        )

        trainer_params = self.trainer_params.copy()
        if not has_validation:
            logger.info("No validation data - training without validation")
            trainer_params.update({"num_sanity_val_steps": 0, "limit_val_batches": 0})

        # Default logger for LearningRateMonitor compatibility. CSV logs land in the SAME per-fit subdir as the checkpoint so the
        # entire run's artifacts (ckpt + metrics + LR-monitor csvs) are co-located under one path; trivially diffable / archivable.
        if "logger" not in trainer_params:
            trainer_params["logger"] = CSVLogger(save_dir=_ckpt_root, name="")

        callbacks = [
            checkpointing,
            TQDMProgressBar(refresh_rate=10),
        ]

        # Only add LearningRateMonitor if logger is enabled
        if trainer_params.get("logger") is not False:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        if self.use_swa:
            swa_params = self.swa_params or {}
            callbacks.append(StochasticWeightAveraging(**swa_params))

        if has_validation:
            logger.info(f"Using early_stopping_rounds={self.early_stopping_rounds:_}")
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=f"val_{metric_name}",
                    min_delta=0.001,
                    patience=self.early_stopping_rounds,
                    mode="min",
                    # verbose=False: BestEpochModelCheckpoint already emits
                    # "New best model at epoch X with metric=..." via mlframe's
                    # logger (neural/base.py:771). Lightning's verbose=True
                    # would duplicate that as both a logger.info and a print()
                    # for every improvement -- 3 lines per best-epoch event.
                    verbose=False,
                )
            )

        trainer = L.Trainer(**trainer_params, callbacks=callbacks)

        # Per-fit model_params override for multilabel: swap CE loss -> BCE,
        # tag task_type so predict_step uses sigmoid not softmax. We DON'T
        # mutate self.model_params (would break sklearn clone + introspection
        # and bleed multilabel config into subsequent fits on different y).
        _local_model_params = dict(self.model_params)
        if self._is_multilabel:
            import torch.nn.functional as _F
            _local_model_params["loss_fn"] = _F.binary_cross_entropy_with_logits
            _local_model_params["task_type"] = "multilabel"

        with trainer.init_module():
            self.model = self.model_class(network=self.network, metrics=metrics, **_local_model_params)

            features_dtype = self.datamodule_params.get("features_dtype", torch.float32)
            data_slice = X.iloc[0:2, :].values if isinstance(X, pd.DataFrame) else X[0:2, :]

            try:
                self.model.example_input_array = to_tensor_any(data_slice, dtype=features_dtype, safe=True)
            except Exception:
                logger.warning("Failed to prepare example_input_array", exc_info=True)

        if self.tune_params and not (is_partial_fit and hasattr(self, "_tuned")):
            tuner = Tuner(trainer)

            if self.tune_batch_size:
                tuner.scale_batch_size(model=self.model, datamodule=dm, mode="binsearch", init_val=self.datamodule_params.get("batch_size", 32))

            lr_finder = tuner.lr_find(self.model, datamodule=dm, num_training=300)
            new_lr = lr_finder.suggestion()
            logger.info("Using suggested LR=%s", new_lr)
            self.model.hparams.learning_rate = new_lr

            if is_partial_fit:
                self._tuned = True

        trainer.fit(model=self.model, datamodule=dm)

        # Extract best epoch from model (set by checkpoint callback, DDP-safe). Prefer model.best_epoch over callback.best_epoch
        # for distributed training compatibility.
        if hasattr(self.model, "best_epoch") and self.model.best_epoch is not None:
            self.best_epoch = self.model.best_epoch
            logger.info("Best epoch recorded: %s", self.best_epoch)
        else:
            # Fallback to callback for backward compatibility
            for callback in trainer.callbacks:
                if isinstance(callback, BestEpochModelCheckpoint):
                    self.best_epoch = callback.best_epoch
                    if self.best_epoch is not None:
                        logger.info("Best epoch recorded from callback: %s", self.best_epoch)
                    break

        # Clean up to avoid pickle issues and free memory
        self.trainer = None

        return self

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to the data.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional per-sample weights for training
            **fit_params: Additional parameters including:
                - eval_set: Tuple of (X_val, y_val) for validation
                - eval_sample_weight: Optional validation sample weights
        """
        eval_set = fit_params.get("eval_set", (None, None))
        # Support sample_weight both as parameter and in fit_params
        if sample_weight is None:
            sample_weight = fit_params.get("sample_weight")
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=False, fit_params=fit_params, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes: Optional[np.ndarray] = None, sample_weight=None, **fit_params):
        """Incremental training for online learning."""
        eval_set = fit_params.get("eval_set", (None, None))
        if sample_weight is None:
            sample_weight = fit_params.get("sample_weight")
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=True, classes=classes, fit_params=fit_params, sample_weight=sample_weight)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Returns a dictionary of all parameters for scikit-learn compatibility.

        All __init__ parameters must be included for sklearn.base.clone() to work correctly.
        """
        # Wave 26 P1 fix (2026-05-20): pre-fix ``trainer_params`` and
        # ``tune_params`` were returned by reference even with deep=True;
        # the four sibling param-dicts (model_params, network_params,
        # datamodule_params, swa_params) were correctly deepcopied. This
        # asymmetry was an oversight: sklearn's clone() calls
        # get_params(deep=True) and rebinds into a new instance. Any
        # downstream mutation of the clone's trainer_params (e.g. setting
        # a new logger) poisoned the original estimator that was still
        # being trained.
        params = {
            "model_class": self.model_class,
            "model_params": deepcopy(self.model_params) if deep else self.model_params,
            "network_params": deepcopy(self.network_params) if deep else self.network_params,
            "datamodule_class": self.datamodule_class,
            "datamodule_params": deepcopy(self.datamodule_params) if deep else self.datamodule_params,
            "trainer_params": deepcopy(self.trainer_params) if deep else self.trainer_params,
            "use_swa": self.use_swa,
            "swa_params": deepcopy(self.swa_params) if deep and self.swa_params else self.swa_params,
            "tune_params": deepcopy(self.tune_params) if deep and self.tune_params else self.tune_params,
            "tune_batch_size": self.tune_batch_size,
            "float32_matmul_precision": self.float32_matmul_precision,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        return params

    def set_params(self, **params: Any) -> "PytorchLightningEstimator":
        """Sets parameters for scikit-learn compatibility."""
        for key, value in params.items():
            if key in ("model_params", "datamodule_params"):
                setattr(self, key, deepcopy(value))  # deep copy nested dicts
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Parameter {key} not found in {self.__class__.__name__}")
        return self

    def _predict_raw(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Internal method for memory-efficient batched prediction using Lightning's trainer.predict().

        This method processes data in batches to avoid OOM errors, leveraging the existing
        DataModule prediction infrastructure.

        Args:
            X: Input data (numpy array, pandas DataFrame, polars DataFrame, or torch.Tensor)
            device: Optional device string ('cpu' or 'cuda'). If None, uses trainer's device.
            precision: Optional precision mode for inference ('16-mixed', 'bf16-mixed', 'bf16-true', or None).
                       If not provided, falls back to the trainer's precision.
            batch_size: Optional batch size for prediction. If None, uses datamodule's batch_size.
                        Larger batch sizes can speed up prediction but use more memory.

        Returns:
            numpy.ndarray: Model predictions (probabilities for classification, values for regression)
        """

        if not hasattr(self, "model") or self.model is None:
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("Model has not been fitted yet. Call fit() before predict().")

        if not hasattr(self, "prediction_datamodule") or self.prediction_datamodule is None:
            # Create a minimal datamodule for prediction if not available
            logger.warning("No datamodule found from training. Creating temporary datamodule for prediction.")
            datamodule = self.datamodule_class(**self.datamodule_params)
        else:
            # Pre-fix this else-branch was missing and ``datamodule`` was
            # left unbound; line 522 ``datamodule.setup_predict(...)`` then
            # raised ``UnboundLocalError`` whenever a training-time
            # datamodule was retained on the estimator.
            datamodule = self.prediction_datamodule

        # Determine batch size for prediction. Three layers of precedence:
        #   1. ``batch_size`` arg explicitly passed to predict() - always wins.
        #   2. ``datamodule_params["predict_batch_size"]`` - the suite-level knob plumbed through ``train_mlframe_models_suite``
        #      via ``hyperparams_config.mlp_predict_batch_size``.
        #   3. Adaptive resolver based on free memory + input width.
        # The legacy ``datamodule_params["batch_size"]`` (train batch) is the last-resort fallback when the resolver fails.
        #
        # The legacy fallback was a hardcoded 64, which made 4M-row predict paths spend minutes on DataLoader overhead for
        # microseconds of actual MLP compute. The adaptive resolver picks the biggest batch that fits 25% of free memory at the
        # input width, clamped to ``[64, 16384]``.
        if batch_size is not None:
            pred_batch_size = int(batch_size)
            _batch_source = "predict argument"
        else:
            override = self.datamodule_params.get("predict_batch_size")
            if override is not None:
                pred_batch_size = max(1, int(override))
                _batch_source = "datamodule predict_batch_size"
            else:
                try:
                    from mlframe.training.mlp_runtime_defaults import resolve_mlp_predict_batch_size
                    # Probe input width when possible - cheap on numpy / pandas / polars. shape[1] is the standard width on all three.
                    _n_features: Optional[int] = None
                    try:
                        if hasattr(X, "shape") and len(X.shape) >= 2:
                            _n_features = int(X.shape[1])
                        elif hasattr(X, "columns"):
                            _n_features = int(len(X.columns))
                    except Exception:
                        _n_features = None
                    pred_batch_size = resolve_mlp_predict_batch_size(
                        n_features=_n_features,
                        train_batch_size=self.datamodule_params.get("batch_size"),
                    )
                    _batch_source = f"auto n_features={_n_features if _n_features is not None else 'unknown'}"
                except Exception:
                    # Resolver failed - fall back to the train-time batch size (still vastly better than 64 on production setups).
                    _train_batch_hint = self.datamodule_params.get("batch_size", 1024)
                    if isinstance(_train_batch_hint, str):
                        _train_batch_hint = 1024
                    pred_batch_size = int(_train_batch_hint)
                    _batch_source = "fallback train batch_size"
        logger.info("MLP prediction batch_size=%s (%s)", pred_batch_size, _batch_source)

        datamodule.setup_predict(X, batch_size=pred_batch_size)

        if not hasattr(self, "trainer") or self.trainer is None:
            trainer_params = {
                "accelerator": "auto",
                "devices": 1,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            }
            prediction_trainer = L.Trainer(**trainer_params)
        else:
            # Use existing trainer but create a new one to avoid state issues
            trainer_params = {
                "accelerator": (
                    self.trainer.accelerator.__class__.__name__.replace("Accelerator", "").lower() if hasattr(self.trainer, "accelerator") else "auto"
                ),
                "devices": 1,  # single device for prediction
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            }

            if device is not None:
                if device.startswith("cuda"):
                    trainer_params["accelerator"] = "cuda"
                elif device == "cpu":
                    trainer_params["accelerator"] = "cpu"

            if precision is not None:
                trainer_params["precision"] = precision
            elif hasattr(self.trainer, "precision"):
                trainer_params["precision"] = self.trainer.precision

            prediction_trainer = L.Trainer(**trainer_params)

        # Unconditional eval() switch - cheap idempotent op, removes the spurious
        # "Model was in training mode during prediction" warning that fired on
        # every legit predict-after-fit (Lightning's Trainer.fit leaves the model
        # in train mode on exit).
        self.model.eval()
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.eval()

        try:
            predictions = prediction_trainer.predict(
                model=self.model,
                datamodule=datamodule,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        self.trainer = None

        if len(predictions) == 0:
            raise RuntimeError("No predictions were generated. Check your data and model.")

        # Handle different return types from predict_step
        if isinstance(predictions[0], torch.Tensor):
            predictions = torch.cat(predictions, dim=0)
            predictions = to_numpy_safe(predictions, cpu=True)
        elif isinstance(predictions[0], np.ndarray):
            predictions = np.concatenate(predictions, axis=0)
        else:
            raise TypeError(f"Unexpected prediction type: {type(predictions[0])}")

        logger.info("Generated predictions with shape %s", predictions.shape)

        return predictions

    def predict(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict using the model with memory-efficient batched processing.

        Args:
            X: Input data (numpy array, pandas DataFrame, polars DataFrame, or torch.Tensor)
            device: Optional device string ('cpu' or 'cuda'). If None, uses trainer's device.
            precision: Optional precision mode for inference ('16-mixed', 'bf16-mixed', etc.)
            batch_size: Optional batch size for prediction. Larger values speed up prediction but use more memory.

        Returns:
            numpy.ndarray: Model predictions (class labels for classification, values for regression)
        """
        predictions = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)

        # For regression, return raw predictions (squeeze to 1D for single-target regression)
        if isinstance(self, RegressorMixin):
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions.squeeze(axis=1)
            return predictions

        # Base class returns probabilities for classification; PytorchLightningClassifier overrides to return labels.
        return predictions

    def score(self, X, y, sample_weight: Optional[np.ndarray] = None) -> float:
        """Returns the coefficient of determination R^2 for regression or accuracy for classification."""
        y_pred = self.predict(X)
        if isinstance(self, RegressorMixin):
            return r2_score(y, y_pred, sample_weight=sample_weight)
        elif isinstance(self, ClassifierMixin):
            # y_pred is already class labels from PytorchLightningClassifier.predict()
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            raise TypeError(f"Estimator must be a RegressorMixin or ClassifierMixin, got {type(self).__name__}")


class PytorchLightningRegressor(RegressorMixin, PytorchLightningEstimator):  # RegressorMixin must come first
    _estimator_type = "regressor"


class PytorchLightningClassifier(
    ClassifierMixin,
    PytorchLightningEstimator,
):  # ClassifierMixin must come first
    _estimator_type = "classifier"

    def predict(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Input data
            device: Optional device string ('cpu' or 'cuda')
            precision: Optional precision mode for inference
            batch_size: Optional batch size for prediction

        Returns:
            numpy.ndarray: Predicted class labels
        """
        proba = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Input data
            device: Optional device string ('cpu' or 'cuda')
            precision: Optional precision mode for inference
            batch_size: Optional batch size for prediction

        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        return self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)


# ----------------------------------------------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------------------------------------------


class NetworkGraphLoggingCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        pl_module.logger.log_graph(model=pl_module)


class AggregatingValidationCallback(Callback):

    def __init__(self, metric_name: str, metric_fcn: object, on_epoch: bool = True, on_step: bool = False, prog_bar: bool = True):
        # Wave 56 (2026-05-20): forward to Lightning's Callback base so any future
        # state it sets in __init__ is populated (currently a no-op).
        super().__init__()
        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)
        self.init_accumulators()

    def init_accumulators(self):
        self.batched_predictions = []
        self.batched_labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        predictions, labels = outputs
        self.batched_labels.append(labels)
        self.batched_predictions.append(predictions)

    def on_validation_epoch_end(self, trainer, pl_module):
        labels = torch.concat(self.batched_labels).detach().cpu().numpy()
        predictions = torch.concat(self.batched_predictions).detach().cpu().float().numpy()
        metric_value = self.metric_fcn(y_true=labels, y_score=predictions)
        pl_module.log(name="val_" + self.metric_name, value=metric_value, on_epoch=self.on_epoch, on_step=self.on_step, prog_bar=True)
        self.init_accumulators()


class BestEpochModelCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint that tracks the epoch of the best model according to the monitored metric.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min", **kwargs):
        super().__init__(monitor=monitor, mode=mode, **kwargs)
        self.best_epoch: Optional[int] = None
        self.best_score: Optional[float] = None

        # operator module used for pickling support
        if mode == "min":
            self.monitor_op = operator.lt
            self.best_score = float("inf")
        elif mode == "max":
            self.monitor_op = operator.gt
            self.best_score = float("-inf")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logger.info("Initialized BestEpochModelCheckpoint with monitor=%s, mode=%s", monitor, mode)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Update best_epoch after each validation step if current metric improves.
        """
        super().on_validation_end(trainer, pl_module)

        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            logger.warning(f"Monitor metric '{self.monitor}' not found in callback_metrics.")
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = trainer.current_epoch
            # Also set on pl_module for DDP synchronization
            pl_module.best_epoch = self.best_epoch
            logger.info("New best model at epoch %s with %s=%.4f", self.best_epoch, self.monitor, self.best_score)


class PeriodicLearningRateFinder(LearningRateFinder):
    def __init__(self, period: int, *args, **kwargs):
        if not isinstance(period, int):
            raise TypeError(f"period must be an int, got {type(period).__name__}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period!r}")
        super().__init__(*args, **kwargs)
        self.period = period

    def on_train_epoch_start(self, trainer, pl_module):
        if (trainer.current_epoch % self.period) == 0 or trainer.current_epoch == 0:
            print(f"Finding optimal learning rate. Current rate={pl_module.learning_rate}")
            self.lr_find(trainer, pl_module)
            print(f"Set learning rate to {pl_module.learning_rate}")
