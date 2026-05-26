"""
Base infrastructure for PyTorch Lightning models in mlframe.

This module provides:
- sklearn-compatible estimator wrappers (PytorchLightningEstimator, Regressor, Classifier)
- Callbacks (BestEpochModelCheckpoint, AggregatingValidationCallback, etc.)
- Utilities (MetricSpec, to_tensor_any, to_numpy_safe)
"""

from __future__ import annotations


import logging

logger = logging.getLogger(__name__)

# Logging filters / MetricSpec / suppression context manager / _rmse_metric: side-effect log-filter attach runs at sibling import time. Re-exports preserve identity for downstream isinstance / hasattr.
from ._base_logging import (  # noqa: F401, E402
    _LightningRankZeroNoiseFilter,
    _LIGHTNING_NOISE_FILTER,
    suppress_lightning_workers_warning,
    _rmse_metric,
    MetricSpec,
)

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

import warnings as _warnings
from contextlib import contextmanager as _contextmanager


# iter189 (2026-05-23): Lightning's _load_external_callbacks scans every
# installed Python distribution via importlib.metadata.entry_points on EACH
# Trainer.fit() / Trainer.predict() invocation -- ~180ms / call on a Windows
# box with a typical anaconda site-packages (5346 dist-info METADATA reads).
# c0065 iter189 profile attributed 1.484s to this across 6 fit calls (~6% of
# the 23.4s wall). Result is process-stable (sys.path + installed dists don't
# change between fits), so cache it.
#
# Mirrors the _PROBE_PRECISION_CACHE pattern in mlp_runtime_defaults.py
# (iter181) and _CB_GPU_USABLE_CACHE in _cb_pool.py. Defensive try/except so
# a Lightning internal-API rename surfaces as a slow-but-correct fallback,
# not an ImportError that crashes mlframe import.
#
# iter259 (2026-05-23) follow-up: patching only ``_lf_registry`` misses the
# real callers. ``callback_connector.py`` and ``fabric.py`` do
# ``from lightning.fabric.utilities.registry import _load_external_callbacks``
# at import time, which binds the ORIGINAL function into the caller's module
# namespace -- mutating ``_lf_registry._load_external_callbacks`` after that
# leaves caller bindings stale. c0119 iter259 profile still attributed 3.73s
# to ``_load_external_callbacks`` (12 calls x 311ms) despite the iter189
# patch. Rebind in every importer to make the cache actually fire.
try:
    from lightning.fabric.utilities import registry as _lf_registry
    if not getattr(_lf_registry, "_mlframe_callback_cache_installed", False):
        _orig_load_external_callbacks = _lf_registry._load_external_callbacks
        _external_callback_cache: Dict[str, list] = {}

        def _load_external_callbacks_cached(group: str) -> list:
            cached = _external_callback_cache.get(group)
            if cached is None:
                cached = _orig_load_external_callbacks(group)
                _external_callback_cache[group] = cached
            return list(cached)  # defensive copy so callers can't mutate cache

        _lf_registry._load_external_callbacks = _load_external_callbacks_cached
        # Rebind in every Lightning module that imported the original by name.
        # Each ``from ... import _load_external_callbacks`` creates a local
        # binding that mutating the source module does not affect. Walk
        # sys.modules and rebind every match. Best-effort: a Lightning version
        # that adds a new caller will fall back to the slow path until the
        # next mlframe release, never breaking.
        #
        # Both the umbrella ``lightning.*`` and the standalone ``lightning_fabric.*``
        # namespaces exist in modern Lightning installs (the umbrella re-exports
        # the standalone package); we must cover both prefixes or callers via
        # the standalone path bypass the cache.
        import sys as _sys_for_rebind
        _rebind_prefixes = ("lightning.", "lightning_fabric.", "lightning_pytorch.")
        for _mod_name, _mod in list(_sys_for_rebind.modules.items()):
            if _mod is None:
                continue
            if not (
                _mod_name == "lightning"
                or _mod_name == "lightning_fabric"
                or _mod_name.startswith(_rebind_prefixes)
            ):
                continue
            _local_ref = getattr(_mod, "_load_external_callbacks", None)
            if _local_ref is None or _local_ref is _load_external_callbacks_cached:
                continue
            try:
                _mod._load_external_callbacks = _load_external_callbacks_cached
            except Exception:
                # Frozen / immutable module objects: skip silently.
                pass
        _lf_registry._mlframe_callback_cache_installed = True
except Exception:
    pass


from pyutilz.pythonlib import get_parent_func_args, store_params_in_object  # noqa: F401
from mlframe.metrics.core import compute_probabilistic_multiclass_error

# Tensor / dataframe helpers carved to sibling. Re-exports preserve identity.
from ._base_tensor_helpers import (  # noqa: F401, E402
    custom_collate_fn,
    to_tensor_any,
    to_numpy_safe,
    _ensure_numpy,
)


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

        # Accept both eval_set conventions:
        #   - bare 2-tuple ``(X_val, y_val)`` (this estimator's native form)
        #   - list-of-tuples ``[(X_val, y_val), ...]`` (LightGBM / XGBoost form,
        #     which ``_maybe_pass_sample_weight`` in composite_ensemble.py emits
        #     uniformly so the same fit-call works across boosters and MLP).
        # Without this normalisation, the OOF refit path indexes ``eval_set[1]``
        # below and raises IndexError on the 1-element list -> MLP component
        # silently dropped from CT_ENSEMBLE for every target (observed in prod).
        if isinstance(eval_set, list) and eval_set and isinstance(eval_set[0], tuple):
            eval_set = eval_set[0]
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

        # Compute output_activation_scale / center from the y the MLP sees
        # at fit-time (Fix 1, 2026-05-26). When the wrapping TTR z-scores y,
        # the MLP sees scaled y and the tanh window lives in scaled space;
        # TTR.inverse_transform unwinds it correctly. Only applied for
        # regression (num_classes==1) with output_activation set; left
        # untouched for classification and the linear default.
        _net_params = dict(self.network_params)
        _out_act = _net_params.get("output_activation", "linear")
        if (
            _out_act == "tanh_train_range"
            and num_classes == 1
            and _net_params.get("output_activation_scale") is None
            and _net_params.get("output_activation_center") is None
        ):
            try:
                _y_arr = np.asarray(
                    y.values if isinstance(y, pd.Series) else y,
                    dtype=np.float64,
                ).reshape(-1)
                _y_finite = _y_arr[np.isfinite(_y_arr)]
                if _y_finite.size > 1:
                    _ymin = float(_y_finite.min())
                    _ymax = float(_y_finite.max())
                    _ystd = float(_y_finite.std())
                    # scale = (max-min)/2 + 3*std; ~6-sigma half-window
                    # around the train midpoint. center = (min+max)/2.
                    _net_params["output_activation_scale"] = (_ymax - _ymin) / 2.0 + 3.0 * _ystd
                    _net_params["output_activation_center"] = (_ymin + _ymax) / 2.0
                    logger.info(
                        "MLP output_activation='tanh_train_range' "
                        "auto-derived from y_train: scale=%.4g, center=%.4g "
                        "(y_min=%.4g, y_max=%.4g, y_std=%.4g).",
                        _net_params["output_activation_scale"],
                        _net_params["output_activation_center"],
                        _ymin, _ymax, _ystd,
                    )
                else:
                    logger.warning(
                        "MLP output_activation='tanh_train_range' requested "
                        "but y_train has <=1 finite value; falling back to "
                        "'linear' for this fit.",
                    )
                    _net_params["output_activation"] = "linear"
            except Exception as _oa_err:
                logger.warning(
                    "MLP output_activation='tanh_train_range' y_train "
                    "derivation failed (%s); falling back to 'linear'.",
                    _oa_err,
                )
                _net_params["output_activation"] = "linear"

        # getattr handles freshly cloned models that don't have network attribute yet
        if getattr(self, 'network', None) is None:
            self.network = generate_mlp(num_features=X.shape[1], num_classes=num_classes, **_net_params)

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

        callbacks = [checkpointing]
        # Lightning raises ``MisconfigurationException`` when both
        # ``enable_progress_bar=False`` is in trainer_params AND a
        # ``TQDMProgressBar`` is registered in callbacks. Only attach the
        # progress-bar callback when the caller hasn't explicitly disabled it.
        if trainer_params.get("enable_progress_bar", True):
            callbacks.append(TQDMProgressBar(refresh_rate=10))

        # Only add LearningRateMonitor if logger is enabled
        if trainer_params.get("logger") is not False:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        if self.use_swa:
            swa_params = self.swa_params or {}
            callbacks.append(StochasticWeightAveraging(**swa_params))

        if has_validation:
            logger.info("Using early_stopping_rounds=%d", self.early_stopping_rounds)
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
            # 2026-05-23 audit-followup #6: divergence detector. Warns
            # when val_loss climbs >=100x its baseline within training
            # so operators catch Identity-MLP-style collapses before
            # paying the full training budget. No automatic stop --
            # ES already covers the no-improvement case.
            callbacks.append(
                ValLossDivergenceCallback(
                    monitor=f"val_{metric_name}",
                    divergence_factor=100.0,
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


# Callback classes carved to ``_base_callbacks.py``. Re-exports preserve class identity so downstream isinstance / Trainer callback-list checks keep working unchanged.
from ._base_callbacks import (  # noqa: F401, E402
    NetworkGraphLoggingCallback,
    AggregatingValidationCallback,
    ValLossDivergenceCallback,
    BestEpochModelCheckpoint,
    PeriodicLearningRateFinder,
)
