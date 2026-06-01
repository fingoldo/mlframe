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

# Keys the suite stashes in ``datamodule_params`` for the estimator's own
# predict-time use that are NOT constructor parameters of the datamodule
# classes (TorchDataModule / RecurrentDataModule). They must be stripped
# before any ``datamodule_class(**params)`` splat. ``predict_batch_size`` is
# read directly off ``self.datamodule_params`` in ``_predict_raw``; passing it
# to the datamodule constructor raises ``unexpected keyword argument``.
_PREDICT_ONLY_DM_PARAM_KEYS = frozenset({"predict_batch_size"})

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
    safe_accelerator,
)

# sklearn get_params/set_params carved to a sibling; bound as methods on the
# estimator class below. Re-import keeps them accessible at module scope.
from ._base_sklearn_params import (  # noqa: F401, E402
    get_params as _sklearn_get_params,
    set_params as _sklearn_set_params,
)


def _make_binary_focal_loss(gamma: float, alpha: float):
    """F-29 (2026-05-31): build a callable focal-loss for binary classification.

    Uses torchvision.ops.sigmoid_focal_loss when available (one fused
    kernel, well-tested). Falls back to a pure-PyTorch implementation
    otherwise (torchvision is optional). Both paths accept the standard
    ``(predictions, targets, reduction='mean'|'sum'|'none')`` signature
    so they compose with the estimator's _compute_weighted_loss /
    _loss_unreduced shape handling.

    Lin et al. 2017 (RetinaNet): FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t).
    Default alpha=0.25 weights the minority class (class 1) higher;
    gamma=2.0 is the original paper's recommended starting point.
    """
    try:
        from torchvision.ops import sigmoid_focal_loss as _tv_focal
        _has_torchvision = True
    except ImportError:
        _has_torchvision = False

    def _focal(input, target, reduction: str = "mean"):
        # The estimator passes labels as float for the BCE-replacement
        # path (labels_dtype was set to float32 in F-05 / F-29). Cast
        # defensively in case a custom carrier slips a Long through.
        if target.dtype != input.dtype:
            target = target.to(input.dtype)
        # Shape alignment: squeeze either side's (N, 1) -> (N,) so the
        # focal kernel sees matching ranks (BCE-shaped path).
        if input.dim() == 2 and input.shape[-1] == 1 and target.dim() == 1:
            input = input.squeeze(-1)
        elif target.dim() == 2 and target.shape[-1] == 1 and input.dim() == 1:
            target = target.squeeze(-1)
        if _has_torchvision:
            return _tv_focal(input, target, alpha=alpha, gamma=gamma, reduction=reduction)
        # Pure-PyTorch fallback: mirror torchvision's implementation.
        p = torch.sigmoid(input)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction="none",
        )
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            loss = alpha_t * loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss  # reduction == "none"

    return _focal


def _validate_no_nan_inf(arg_name: str, data, allow_object_dtype: bool = False) -> None:
    """F-23 (2026-05-30) helper: reject NaN / inf in features or labels at
    fit() entry with a clear, actionable error. Pre-fix NaN propagated
    silently through the network producing all-NaN predictions.

    ``allow_object_dtype=True`` short-circuits the check for object-dtype
    targets (string / Python labels), which the LabelEncoder block will
    reject downstream with its own message if invalid.
    """
    if data is None:
        return
    # Normalise to np.ndarray for the check. Avoid copy when possible.
    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy()
    elif isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, pl.DataFrame):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    # Skip the finite-check on non-numeric dtypes; np.isnan raises on
    # object/str arrays. LabelEncoder will reject string labels for
    # regressors downstream; this guard is for numeric data only.
    if arr.dtype.kind not in ("f", "i", "u", "b"):
        if allow_object_dtype:
            return
        raise ValueError(
            f"{arg_name} has dtype {arr.dtype!r}; PytorchLightningEstimator "
            "requires numeric dtype (float / int / bool). Convert via "
            "pd.get_dummies, sklearn OrdinalEncoder, or similar."
        )
    if arr.dtype.kind == "f":
        # Only float arrays can carry NaN / inf; int / bool arrays can't.
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise ValueError(
                f"{arg_name} contains {n_nan} NaN and {n_inf} inf values. "
                "PytorchLightningEstimator does NOT impute internally because "
                "NaN propagates through the network -> all-NaN predictions. "
                "Pre-process with sklearn.impute.SimpleImputer / "
                "IterativeImputer, drop the offending rows via "
                f"{arg_name}.dropna(), or wrap the estimator in a sklearn "
                "Pipeline whose first step handles missing values."
            )


class PytorchLightningEstimator(BaseEstimator):
    """Wrapper that allows Pytorch Lightning model, datamodule and trainer to participate in sklearn pipelines.
    Supports early stopping (via eval_set in fit_params).
    """

    def __getstate__(self) -> dict:
        """F-73b (2026-06-01): drop runtime-only, non-picklable caches on
        serialise. The F-67 prediction-trainer cache holds live
        ``lightning.pytorch.Trainer`` objects which reference a
        ``lightning_utilities.core.rank_zero.WarningCache`` -- a class the
        mlframe save_load ``_SafeUnpickler`` allowlist (correctly) blocks.
        These caches exist purely to skip per-predict Trainer
        re-construction; the next predict() on the restored estimator
        lazily rebuilds the cache. Also null the live ``trainer`` for the
        same reason (it's already nulled after every fit/predict, but a
        mid-lifecycle pickle could still catch a live one).
        """
        state = self.__dict__.copy()
        state.pop("_prediction_trainer_cache", None)
        state["trainer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Rebuilt lazily on the next predict(); start clean.
        self._prediction_trainer_cache = {}

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
        use_ema: bool = False,
        ema_params: dict = None,
        label_smoothing: float = 0.0,
        focal_loss_gamma: Optional[float] = None,
        focal_loss_alpha: float = 0.25,
        tune_params: bool = False,
        tune_batch_size: bool = False,
        float32_matmul_precision: str = None,
        early_stopping_rounds: int = 100,
        random_state: Optional[int] = None,
        class_weight=None,
    ):
        # ``random_state``: sklearn-canonical seed parameter (F-06, 2026-05-30).
        # When set to an integer, ``_fit_common`` seeds torch / numpy / Python
        # random + the Lightning DataLoader worker seed at fit() entry, so two
        # ``fit()`` calls on the same data with the same ``random_state``
        # produce bit-identical predictions. ``None`` (the default) preserves
        # the pre-fix non-deterministic behaviour: callers who manage their
        # own seed (e.g. via a higher-level pipeline) are not overridden.
        #
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

        # F-06 (2026-05-30): sklearn-canonical reproducibility seed. When
        # ``random_state`` is an int, seed torch + numpy + Python random +
        # the Lightning DataLoader worker seed BEFORE any random op fires
        # (network init at line 357, dataloader shuffle, dropout mask
        # sampling). Same data + same random_state -> bit-identical
        # predictions. ``None`` leaves the prior non-deterministic
        # behaviour intact; callers managing their own seed are not
        # overridden. partial_fit honours the same seed on every batch
        # (idempotent — re-seeding before each call is fine).
        if self.random_state is not None:
            # ``verbose`` was added to ``L.seed_everything`` in lightning >=2.x;
            # older installs (the TVT-regression test box) raise TypeError on
            # the kwarg. Try the quiet form first, fall back to the legacy
            # signature so the same code base stays portable across Lightning
            # versions. (When we fall through, Lightning prints the seed line
            # at INFO; the ``_LightningRankZeroNoiseFilter`` further down still
            # suppresses noisy rank-zero chatter, so the on-disk log stays
            # essentially identical.)
            try:
                L.seed_everything(int(self.random_state), workers=True, verbose=False)
            except TypeError:
                L.seed_everything(int(self.random_state), workers=True)

        # F-23 (2026-05-30): reject NaN / inf in features or labels at fit()
        # entry. Pre-fix any NaN propagated through the first Linear ->
        # all-NaN activations -> all-NaN gradients -> all-NaN weights after
        # one step -> all-NaN predictions; the suite saw a flat val curve
        # with no log signal. Now: explicit ValueError with a remediation
        # hint. Skip the check on string / object dtypes (LabelEncoder will
        # reject those further down with its own clear error).
        _validate_no_nan_inf("X", X)
        _validate_no_nan_inf("y", y, allow_object_dtype=True)
        if eval_set is not None and not (isinstance(eval_set, tuple) and eval_set[0] is None):
            # eval_set may be a 2-tuple ``(X_val, y_val)`` or a list-of-tuples
            # (LightGBM convention) -- normalise to peek at the val frame.
            _ev = eval_set[0] if isinstance(eval_set, list) and eval_set else eval_set
            if isinstance(_ev, tuple) and _ev[0] is not None:
                _validate_no_nan_inf("X_val", _ev[0])
                _validate_no_nan_inf("y_val", _ev[1], allow_object_dtype=True)

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

        # ``predict_batch_size`` is a predict-time-only knob the suite plumbs
        # into ``datamodule_params`` (see _helpers_training_configs.py:733); it
        # is consumed at predict() directly off ``self.datamodule_params`` and
        # is NOT a constructor parameter of TorchDataModule / RecurrentDataModule.
        # Strip it (and any future predict-only keys) before splatting into the
        # datamodule constructor, otherwise the fit-time build raises
        # ``TorchDataModule.__init__() got an unexpected keyword argument
        # 'predict_batch_size'`` the moment a caller sets mlp_predict_batch_size.
        _local_dm_params = {
            k: v for k, v in self.datamodule_params.items()
            if k not in _PREDICT_ONLY_DM_PARAM_KEYS
        }
        if _is_multilabel_target:
            # BCEWithLogitsLoss requires float labels; CrossEntropyLoss (the
            # classifier default in helpers.py) requires Long. The estimator
            # owns the dispatch — datamodule just delivers the dtype the loss
            # expects.
            _local_dm_params["labels_dtype"] = torch.float32

        # Single-label classifier label encoding. sklearn convention is that
        # ``y`` can be any hashable (strings, non-dense ints, booleans);
        # CrossEntropyLoss + ``labels_dtype=int64`` require ``{0..K-1}``
        # integer indices. Without this encoding, ``fit`` crashed with
        # ``IndexError: Target N is out of bounds`` for any y whose value set
        # is not exactly ``{0..K-1}`` (e.g. ``{10, 20}`` or ``{"low","high"}``;
        # F-19 in the 2026-05-30 mlp audit). Build the bidirectional encoder
        # once and stash on ``self`` so ``predict`` can ``inverse_transform``
        # at inference time (F-01).
        _classifier_single_label = (
            isinstance(self, ClassifierMixin) and not _is_multilabel_target
        )
        if _classifier_single_label:
            from sklearn.preprocessing import LabelEncoder as _LabelEncoder
            if is_partial_fit and classes is not None:
                # ``classes`` is the caller's full universe of labels even if
                # this partial_fit batch only sees a subset. Fit encoder to it
                # so the index space stays stable across partial_fit calls.
                self._label_encoder = _LabelEncoder().fit(np.asarray(classes))
                self.classes_ = self._label_encoder.classes_
            elif not hasattr(self, "_label_encoder") or self._label_encoder is None:
                _y_for_le = y.values if isinstance(y, pd.Series) else np.asarray(y)
                if _y_for_le.ndim == 2 and _y_for_le.shape[1] == 1:
                    _y_for_le = _y_for_le.ravel()
                self._label_encoder = _LabelEncoder().fit(_y_for_le)
                self.classes_ = self._label_encoder.classes_
            # else: partial_fit continuation with encoder already built; reuse.

            # Encode training y to integer indices for the loss function.
            _y_arr_train = y.values if isinstance(y, pd.Series) else np.asarray(y)
            if _y_arr_train.ndim == 2 and _y_arr_train.shape[1] == 1:
                _y_arr_train = _y_arr_train.ravel()
            y = self._label_encoder.transform(_y_arr_train)

            # Encode validation labels with the SAME encoder so val_loss /
            # val_MSE share the index space the model trains on.
            if eval_set[1] is not None:
                _y_arr_val = (
                    eval_set[1].values if isinstance(eval_set[1], pd.Series)
                    else np.asarray(eval_set[1])
                )
                if _y_arr_val.ndim == 2 and _y_arr_val.shape[1] == 1:
                    _y_arr_val = _y_arr_val.ravel()
                eval_set = (eval_set[0], self._label_encoder.transform(_y_arr_val))

            # F-05 (2026-05-30): binary classification uses 1-output
            # sigmoid + BCEWithLogitsLoss instead of 2-output softmax +
            # CrossEntropyLoss. The two-output softmax head is
            # overparameterised (softmax is shift-invariant in z0-z1)
            # and inconsistent with the multilabel BCE path. Switching
            # halves the output-layer params and aligns binary with the
            # K=1 case of multilabel. predict_proba keeps returning the
            # sklearn-canonical (N, 2) shape by stacking [1-p, p] in the
            # classifier wrapper. Detection happens here (before dm
            # construction) so labels_dtype can be set to float32 in
            # time for BCEWithLogitsLoss.
            self._binary_sigmoid_head = bool(len(self.classes_) == 2)
            if self._binary_sigmoid_head:
                _local_dm_params["labels_dtype"] = torch.float32
        else:
            # Multilabel or non-classifier paths: never binary.
            self._binary_sigmoid_head = False

        if _classifier_single_label:
            # F-13 (2026-05-30): sklearn-canonical ``class_weight`` support.
            # ``class_weight="balanced"`` -> per-sample weights = n / (K * count(class))
            # ``class_weight={cls: w, ...}`` -> per-sample weights = w[cls]
            # ``class_weight=None`` -> no per-class weighting
            # The resulting per-sample weights are multiplied INTO any
            # caller-supplied ``sample_weight`` (sklearn convention) so
            # both knobs compose: a caller can weight rare events AND
            # rebalance classes simultaneously.
            if self.class_weight is not None:
                from sklearn.utils.class_weight import (
                    compute_sample_weight as _compute_sample_weight,
                )
                # compute_sample_weight expects ORIGINAL (un-encoded)
                # class labels; pass the train y BEFORE the encoder
                # transformed it. We reconstruct the original via
                # inverse_transform from the already-encoded ``y``.
                _y_for_cw = self._label_encoder.inverse_transform(y)
                _cw_weights = _compute_sample_weight(
                    class_weight=self.class_weight, y=_y_for_cw,
                ).astype(np.float32)
                if sample_weight is None:
                    sample_weight = _cw_weights
                else:
                    # Multiplicative composition with caller's weights.
                    _sw_arr = np.asarray(sample_weight, dtype=np.float32).ravel()
                    if _sw_arr.shape != _cw_weights.shape:
                        raise ValueError(
                            f"class_weight-derived weights shape "
                            f"{_cw_weights.shape} != sample_weight shape "
                            f"{_sw_arr.shape}; cannot multiply."
                        )
                    sample_weight = _sw_arr * _cw_weights
                logger.info(
                    "Applied class_weight=%r -> per-sample weights "
                    "with mean=%.4g, min=%.4g, max=%.4g",
                    self.class_weight,
                    float(np.mean(sample_weight)),
                    float(np.min(sample_weight)),
                    float(np.max(sample_weight)),
                )

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
            # F-24 (2026-05-31): native multi-target regression. When y has
            # shape (N, K>=2) for a regressor (not multilabel), train K
            # output heads sharing the trunk. MSE between (N, K) preds and
            # (N, K) labels works without any loss-shape gymnastics.
            # Single-target (N,) or (N, 1) y keeps num_classes=1.
            _y_check_reg = y.values if isinstance(y, pd.Series) else y
            _y_check_reg = np.asarray(_y_check_reg) if not isinstance(_y_check_reg, np.ndarray) else _y_check_reg
            if _y_check_reg.ndim == 2 and _y_check_reg.shape[1] >= 2:
                num_classes = int(_y_check_reg.shape[1])
                self._is_multi_target_regression = True
            else:
                num_classes = 1
                self._is_multi_target_regression = False
            self._is_multilabel = False

        # F-05 (2026-05-30): binary uses 1-output sigmoid + BCE instead of
        # 2-output softmax + CE -- see the matching block above the dm
        # construction. ``_binary_sigmoid_head`` flag was set there; here
        # we just resolve the network output dim for the network reset
        # below.
        _network_output_dim = 1 if self._binary_sigmoid_head else num_classes

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
        # 2026-06-01: condition uses OR (any None) so a partially-set
        # ``scale OR center`` triggers the auto-fill instead of falling
        # through to ``generate_mlp`` which raises on either one being
        # None. Pre-fix the AND condition would skip the derivation for
        # the (scale=2.0, center=None) shape, then ``generate_mlp`` at
        # flat.py:537 would error on the missing center. The auto-fill
        # body below only overwrites the missing field via ``setdefault``
        # so an explicit user-set scale or center is preserved.
        _scale_set = _net_params.get("output_activation_scale") is not None
        _center_set = _net_params.get("output_activation_center") is not None
        if (
            _out_act == "tanh_train_range"
            and num_classes == 1
            and not getattr(self, "_is_multi_target_regression", False)
            and not (_scale_set and _center_set)
        ):
            try:
                _y_arr = np.asarray(
                    y.values if isinstance(y, pd.Series) else y,
                    dtype=np.float64,
                ).reshape(-1)
                # Single-pass numba kernel: min + max + mean + std over
                # finite entries in ONE traversal of the buffer (Welford's
                # online variance is numerically stable on high-range y;
                # the naive ``y_finite.min() / max() / std()`` triple did
                # three independent passes after materialising an
                # ``isfinite`` mask). Saves ~3x memory bandwidth on a
                # multi-million-row regression target and stays bit-exact
                # vs numpy ddof=0 to ~1e-15.
                from ._neural_numba_kernels import finite_min_max_std as _fmms
                _n_finite, _ymin, _ymax, _ymean, _ystd = _fmms(_y_arr)
                if _n_finite > 1:
                    # scale = (max-min)/2 + 3*std; ~6-sigma half-window
                    # around the train midpoint. center = (min+max)/2.
                    # Fill ONLY the None slots so an explicit user-set
                    # value (scale OR center) is preserved. Asymmetric-
                    # partial input (scale=2.0, center=None) is the case
                    # the pre-fix AND-condition skipped, then
                    # ``generate_mlp`` raised on the missing field.
                    if _net_params.get("output_activation_scale") is None:
                        _net_params["output_activation_scale"] = (_ymax - _ymin) / 2.0 + 3.0 * _ystd
                    if _net_params.get("output_activation_center") is None:
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
            self.network = generate_mlp(num_features=X.shape[1], num_classes=_network_output_dim, **_net_params)

        if num_classes > 1:
            metric_name = "ICE"
            metrics = [MetricSpec(name=metric_name, fcn=compute_probabilistic_multiclass_error, requires_probs=True)]
        else:
            # F-02 (2026-05-30 mlp audit): the metric function is sklearn's
            # ``root_mean_squared_error`` (RMSE), so the label MUST be "RMSE"
            # too. Pre-fix the label was "MSE" -- monitor keys ("val_MSE"),
            # checkpoint filenames (``model-val_MSE=0.7555.ckpt``), and
            # CSV-logger columns all carried the wrong scale label. The
            # metric_direction_dispatcher / metric_name_higher_is_better
            # registry already knew both keys as min-direction, so the
            # rename does not break direction-dependent code paths.
            metric_name = "RMSE"
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
            # F-25 (2026-05-31 cProfile finding): checkpoint writes were
            # 9.59s out of 15.6s total fit wall (61%) on a 10k x 50 / 10-epoch
            # baseline. Lightning's default ModelCheckpoint includes the
            # optimizer state + LR scheduler state + RNG state in every
            # snapshot -- but on_train_end only reads checkpoint["state_dict"]
            # (see _flat_torch_module.py:530-533), so the optimizer / scheduler
            # / RNG bytes are written then discarded at load time. Switching
            # to save_weights_only=True drops them at write time: ~6x smaller
            # snapshot, ~6x faster per-write. Net fit-wall speedup is
            # proportional to checkpoint-write share -- larger networks +
            # longer fits see the most benefit.
            save_weights_only=True,
        )

        trainer_params = self.trainer_params.copy()
        if not has_validation:
            logger.info("No validation data - training without validation")
            trainer_params.update({"num_sanity_val_steps": 0, "limit_val_batches": 0})

        # CUDA-broken-host guard: when the caller leaves the accelerator at
        # ``auto`` (or asks for ``cuda``/``gpu`` outright), probe a 1-element
        # allocation BEFORE Lightning builds the strategy. On hosts with CUDA
        # libs but a broken driver / no device / a context the calling proc
        # can't open, ``Trainer`` would otherwise die deep inside
        # ``model_to_device`` with ``CUDA error: an illegal memory access``;
        # the probe lets us fall back to CPU cleanly so the fit completes.
        # When the operator explicitly forces ``accelerator='cuda'`` and CUDA
        # is unusable, surface that as a log warning + still downgrade
        # (silently failing the fit on a 100-call suite is worse than
        # ignoring a single forced flag).
        _requested = trainer_params.get("accelerator", "auto")
        _resolved = safe_accelerator(_requested)
        if _resolved != _requested and _requested in ("cuda", "gpu"):
            logger.warning(
                "Requested accelerator=%r but CUDA probe failed; "
                "downgrading to CPU so fit can complete.",
                _requested,
            )
        trainer_params["accelerator"] = _resolved

        # F-27 (2026-05-31): auto-enable bf16-mixed precision on Ampere+
        # GPUs. bf16 has the same dynamic range as fp32 (no GradScaler,
        # no NaN risk -- unlike '16-mixed' / fp16). Measured 1.2-1.8x
        # forward+backward speedup on Ampere+ for GEMM-bound workloads,
        # ~30-40% activation-memory reduction.
        #
        # Gating:
        #   * Only when caller didn't set ``precision`` in trainer_params
        #     (explicit > default).
        #   * Only when resolved accelerator is cuda/gpu (CPU bf16 is
        #     slow / unsupported).
        #   * Only when the device's compute capability is >= 8 (Ampere
        #     A100, RTX 30/40 series, H100, etc.). Pre-Ampere bf16 falls
        #     back to fp32 with no speedup but adds autocast overhead.
        # The predict path already accepts precision (base.py:840-957)
        # so inference parity is automatic; fp32 checkpoint load is
        # unaffected because bf16-mixed stores fp32 master weights.
        if "precision" not in trainer_params and _resolved in ("cuda", "gpu"):
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    _cc_major, _ = torch.cuda.get_device_capability(0)
                    if _cc_major >= 8:
                        trainer_params["precision"] = "bf16-mixed"
                        logger.info(
                            "F-27: auto-enabled precision='bf16-mixed' on "
                            "Ampere+ GPU (compute capability %d.x). Set "
                            "trainer_params['precision'] explicitly to "
                            "override (e.g. '32-true' or '16-mixed').",
                            _cc_major,
                        )
            except Exception as _cc_err:
                logger.debug(
                    "F-27 bf16 auto-enable probe failed (%s); leaving "
                    "precision at Lightning default.", _cc_err,
                )

        # Default logger for LearningRateMonitor compatibility. CSV logs land in the SAME per-fit subdir as the checkpoint so the
        # entire run's artifacts (ckpt + metrics + LR-monitor csvs) are co-located under one path; trivially diffable / archivable.
        if "logger" not in trainer_params:
            trainer_params["logger"] = CSVLogger(save_dir=_ckpt_root, name="")

        # F-36 (2026-05-31): opt-in torch.profiler integration via
        # MLFRAME_TORCH_PROFILE=1. Per the 2026-05-31 PyTorch optimization
        # audit (Agent B profiler research), shallow tabular MLPs are
        # typically kernel-launch-bound rather than compute-bound — the
        # 20-40% wall typically spent in inter-kernel gaps is invisible
        # to cProfile (a pure CPU profiler) but immediately visible in
        # torch.profiler's CUDA trace. Lightning's PyTorchProfiler wraps
        # torch.profiler with per-hook record_function ranges already
        # present in the LightningModule call graph, so the trace shows
        # training_step / backward / optimizer_step bounds for free.
        # Defaults: 5-step rolling window (wait=1, warmup=1, active=3) +
        # Chrome trace export to MLFRAME_TORCH_PROFILE_DIR (or ./torch_traces).
        if "profiler" not in trainer_params:
            if os.environ.get("MLFRAME_TORCH_PROFILE", "0") == "1":
                try:
                    from lightning.pytorch.profilers import PyTorchProfiler
                    _activities = [torch.profiler.ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        _activities.append(torch.profiler.ProfilerActivity.CUDA)
                    _prof_dir = os.environ.get(
                        "MLFRAME_TORCH_PROFILE_DIR",
                        os.path.join(_ckpt_root, "torch_traces"),
                    )
                    os.makedirs(_prof_dir, exist_ok=True)
                    # group_by_input_shapes helps recurrent models where
                    # variable seq-lens would otherwise collapse into a
                    # single bucket; harmless for fixed-shape MLP.
                    trainer_params["profiler"] = PyTorchProfiler(
                        dirpath=_prof_dir,
                        filename=f"mlp_{os.getpid()}",
                        export_to_chrome=True,
                        record_module_names=True,
                        activities=_activities,
                        schedule=torch.profiler.schedule(
                            wait=1, warmup=1, active=3, repeat=1,
                        ),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=False,
                        with_flops=True,
                        group_by_input_shapes=True,
                    )
                    logger.info(
                        "F-36: MLFRAME_TORCH_PROFILE=1 active; chrome traces "
                        "land in %s. Open via chrome://tracing or Perfetto.",
                        _prof_dir,
                    )
                except Exception as _prof_err:
                    logger.warning(
                        "MLFRAME_TORCH_PROFILE=1 but profiler setup failed "
                        "(%s); fit continues without profiling.",
                        _prof_err,
                    )

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

        if self.use_swa and self.use_ema:
            raise ValueError(
                "use_swa and use_ema are mutually exclusive — both rewrite "
                "the live model weights at train end (last-write-wins). "
                "Pick one: SWA (broad LR cycle averaging) or EMA "
                "(per-step exponential moving average)."
            )
        if self.use_swa:
            swa_params = self.swa_params or {}
            callbacks.append(StochasticWeightAveraging(**swa_params))
        if self.use_ema:
            # F-28 (2026-05-31): exponential moving average of weights via
            # Lightning's WeightAveraging callback + torch's EMA averaging
            # function. Lightning auto-swaps the averaged weights into the
            # live model on on_train_end, so downstream predict() uses the
            # EMA copy transparently — zero changes to save/load needed.
            # Cross-cited in two 2026-05-31 research agents
            # (Lightning-plugins + activations/optimizers): +0.04-0.66% on
            # tabular MLPs, cheaper than SWA (no LR warm-restart phase).
            # Falls back to a SWA-as-EMA shim when WeightAveraging is not
            # in the installed Lightning (added in Lightning ~2.5).
            try:
                from lightning.pytorch.callbacks import WeightAveraging  # noqa: F401
                _ema_has_native = True
            except ImportError:
                _ema_has_native = False
            from torch.optim.swa_utils import get_ema_avg_fn
            _ema_params = dict(self.ema_params or {})
            # ``decay`` is exposed at the mlframe level for ergonomics;
            # plumb it into get_ema_avg_fn. Default 0.999 mirrors the
            # torch.optim.swa_utils default.
            _decay = float(_ema_params.pop("decay", 0.999))
            _ema_params.setdefault("avg_fn", get_ema_avg_fn(decay=_decay))
            if _ema_has_native:
                from lightning.pytorch.callbacks import WeightAveraging
                callbacks.append(WeightAveraging(**_ema_params))
            else:
                # SWA-as-EMA fallback: SWA accepts ``avg_fn`` (passes to
                # torch's AveragedModel under the hood). Default
                # ``swa_lrs`` to the user's learning_rate so SWA does NOT
                # trigger a LR-restart phase — that would defeat the EMA
                # semantic by tuning a separate "averaged" model with a
                # different LR. ``swa_epoch_start=0.5`` starts averaging
                # halfway through training (standard SWA default).
                _ema_params.setdefault(
                    "swa_lrs",
                    float(self.model_params.get("learning_rate", 1e-3)),
                )
                _ema_params.setdefault("swa_epoch_start", 0.5)
                callbacks.append(StochasticWeightAveraging(**_ema_params))
                logger.info(
                    "use_ema=True: lightning.pytorch.callbacks.WeightAveraging "
                    "is unavailable (Lightning < 2.5?); falling back to "
                    "StochasticWeightAveraging with EMA avg_fn + constant "
                    "swa_lrs=learning_rate so no LR-restart phase. Upgrade "
                    "Lightning to >=2.5 for the dedicated EMA path."
                )

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
        elif self._binary_sigmoid_head:
            # F-05: binary sigmoid head -> BCEWithLogitsLoss + task_type
            # marker so predict_step / compute_metrics emit sigmoid probs
            # and the classifier wrapper stacks (N, 2).
            # F-29 (2026-05-31): optional focal loss for binary. When
            # ``focal_loss_gamma`` is set, replace BCE with the sigmoid
            # focal loss formulation (Lin et al. 2017): heavier penalty
            # on hard examples, mitigates class imbalance even WITHOUT
            # explicit class_weight. Default off — focal loss degrades
            # the model's probability calibration (Cattan 2024) so it's
            # opt-in for users who care more about F1 / recall on
            # severely imbalanced binary targets than about calibrated
            # probabilities. focal_loss_alpha is the class-1 weight
            # (default 0.25 per the original paper).
            if self.focal_loss_gamma is not None:
                _local_model_params["loss_fn"] = _make_binary_focal_loss(
                    gamma=float(self.focal_loss_gamma),
                    alpha=float(self.focal_loss_alpha),
                )
            else:
                _local_model_params["loss_fn"] = torch.nn.BCEWithLogitsLoss()
            _local_model_params["task_type"] = "binary"
        elif isinstance(self, RegressorMixin):
            # F-24 (2026-05-31): tag regressors so predict_step returns
            # raw values for ALL shapes including (N, K>=2) multi-target.
            # Without this tag, predict_step's existing
            # ``logits.shape[1] > 1`` branch would mistakenly apply
            # softmax to (N, K) regression outputs.
            _local_model_params["task_type"] = "regression"
        elif (
            isinstance(self, ClassifierMixin)
            and not self._is_multilabel
            and not self._binary_sigmoid_head
            and self.label_smoothing > 0.0
        ):
            # F-30 (2026-05-31): label smoothing for MULTICLASS only.
            # Replaces the caller's CrossEntropyLoss with one carrying
            # label_smoothing=epsilon. Per RealMLP-TD NeurIPS 2024:
            # +1.8% multiclass accuracy on the ablation. Skipped for
            # binary (Cattan 2024 shows calibration regression on
            # imbalanced binary; focal_loss_gamma is the analogue knob).
            _local_model_params["loss_fn"] = torch.nn.CrossEntropyLoss(
                label_smoothing=float(self.label_smoothing),
            )

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

        # Free the train/val tensors held by the cached datamodule
        # WITHOUT dropping the datamodule shell itself. The full
        # train+val feature / label / sample_weight tensors were the
        # actual save() bloat (1788 MB on disk for a 4M x 323 float32
        # frame, 2026-05-27 TVT regression log) -- the shell (~few KB
        # of config + class refs) is fine to pickle. Keeping the shell
        # lets predict() reuse the configured pre-pipeline /
        # batch_size / dataloader_params without rebuilding the
        # datamodule from scratch, AND silences the spurious "No
        # datamodule found from training" WARNING that fired on every
        # predict-after-fit when we used to NULL the whole reference.
        # Opt out via env MLFRAME_KEEP_PREDICTION_DATAMODULE=1 for
        # operators relying on the prior whole-stash behaviour.
        import os as _os_drop_dm
        if not _os_drop_dm.environ.get("MLFRAME_KEEP_PREDICTION_DATAMODULE"):
            _dm = getattr(self, "prediction_datamodule", None)
            if _dm is not None:
                for _attr in (
                    "train_features", "train_labels", "train_sample_weight",
                    "val_features", "val_labels", "val_sample_weight",
                ):
                    if hasattr(_dm, _attr):
                        setattr(_dm, _attr, None)
                # ``_train_dataset`` / ``_val_dataset`` -- if the
                # datamodule materialised PyTorch ``Dataset`` wrappers
                # (which hold the same tensors via the Dataset's own
                # attributes), null those too. Predict-path setup
                # rebuilds them from the predict-side X / y.
                for _attr in ("_train_dataset", "_val_dataset",
                              "train_dataset", "val_dataset"):
                    if hasattr(_dm, _attr):
                        setattr(_dm, _attr, None)
            self._datamodule_tensors_dropped = True

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

    # sklearn protocol methods carved to ``_base_sklearn_params`` (monolith
    # split). Bound here so ``clone()`` / ``get_params(deep=True)`` behave
    # identically -- the functions take ``self`` as first arg.
    get_params = _sklearn_get_params
    set_params = _sklearn_set_params

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
            # Reaches here only when the estimator was reconstructed
            # without ever going through .fit() (e.g. user-constructed
            # bare estimator, or a load() path that bypasses the sklearn
            # lifecycle). The post-fit memory-safety pass now only NULLs
            # the heavy train/val tensors INSIDE the datamodule (see
            # fit() epilogue), keeping the lightweight shell around so
            # predict-after-fit no longer reaches this branch at all.
            logger.warning("No datamodule found from training. Creating temporary datamodule for prediction.")
            # Same predict-only-key strip as the fit-time construction: the
            # temporary datamodule constructor rejects ``predict_batch_size``
            # (it is read off self.datamodule_params at line ~1161 below, not
            # passed to the datamodule).
            datamodule = self.datamodule_class(**{
                k: v for k, v in self.datamodule_params.items()
                if k not in _PREDICT_ONLY_DM_PARAM_KEYS
            })
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
            _cached_acc = getattr(self, "_last_predict_accelerator", None)
            _user_acc = (self.trainer_params or {}).get("accelerator")
            trainer_params = {
                "accelerator": _cached_acc or _user_acc or "auto",
                "devices": 1,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            }
        else:
            trainer_params = {
                "accelerator": (
                    self.trainer.accelerator.__class__.__name__.replace("Accelerator", "").lower() if hasattr(self.trainer, "accelerator") else "auto"
                ),
                "devices": 1,
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

        # F-67 (2026-05-31): cache the prediction trainer keyed by
        # (accelerator, precision). Pre-fix every _predict_raw call
        # built a fresh L.Trainer + destroyed it via self.trainer=None,
        # paying ~236 ms gc.collect per cycle (cProfile 2026-05-31:
        # 2 calls = 472 ms / 6.94 s profiled wall = 6.8% of total
        # predict path). For typical ensemble suites with 5-20 predict
        # calls per fit, that's 1.2-4.7 s of pure GC overhead per fit.
        # Caching reuses the same Trainer across predicts; Lightning's
        # ``trainer.predict()`` is idempotent and resets internal state
        # between calls. Cache invalidates only on (accelerator,
        # precision) change -- the rare case of mid-suite device flip
        # rebuilds, the typical homogeneous suite hits the cache.
        _cache_key = (
            trainer_params.get("accelerator"),
            trainer_params.get("precision"),
        )
        _trainer_cache = getattr(self, "_prediction_trainer_cache", {})
        prediction_trainer = _trainer_cache.get(_cache_key)
        if prediction_trainer is None:
            prediction_trainer = L.Trainer(**trainer_params)
            _trainer_cache[_cache_key] = prediction_trainer
            self._prediction_trainer_cache = _trainer_cache

        # F-G fix: cache the accelerator the current prediction_trainer
        # was built with so the next _predict_raw call (after
        # ``self.trainer = None`` below) can re-resolve to the same
        # device instead of falling through to accelerator="auto".
        try:
            self._last_predict_accelerator = trainer_params.get(
                "accelerator", "auto",
            )
        except Exception:
            pass

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
        except RuntimeError as e:
            # iter293 (2026-05-26): defensive CPU fallback on CUDA runtime
            # errors. Concurrent CUDA usage by another process on the same
            # GPU (or a CUDA context invalidated by an earlier in-process
            # failure) makes the predict trainer hit ``illegal memory
            # access`` / ``out of memory`` / ``device-side assert`` even
            # when the model and data were fine at fit time. Pre-fix the
            # exception propagated and the whole suite died; that masks
            # genuine training results behind a transient CUDA-context
            # problem. Retry exactly once on CPU so the suite still
            # delivers a usable prediction set + surfaces the underlying
            # CUDA issue as a WARNING.
            #
            # Filter is narrow: only RuntimeError messages containing
            # ``CUDA`` or one of the known CUDA-runtime fingerprints get
            # retried. Other RuntimeError variants (shape mismatch,
            # dataloader misconfig) re-raise immediately.
            _msg = str(e)
            _cuda_fingerprints = (
                "CUDA",
                "cuda runtime error",
                "illegal memory access",
                "device-side assert",
                "out of memory",
                "CUBLAS_STATUS_",
                "CUDNN_STATUS_",
            )
            _is_cuda = (
                trainer_params.get("accelerator") in ("cuda", "gpu", "auto")
                and any(fp in _msg for fp in _cuda_fingerprints)
            )
            if not _is_cuda:
                logger.error(f"Prediction failed: {e}")
                raise
            logger.warning(
                "Prediction on accelerator=%r failed with CUDA-side error "
                "(%s); retrying on CPU. Common cause: another process "
                "holds the GPU or the in-process CUDA context was "
                "invalidated by an earlier failure. The CPU fallback "
                "produces equivalent numeric results but loses GPU "
                "acceleration for this single predict.",
                trainer_params.get("accelerator"), _msg,
            )
            try:
                # iter333 (2026-05-27) follow-up to iter293: the original
                # CPU fallback re-raised because the CUDA context was still
                # dirty when the CPU trainer touched a still-on-GPU tensor
                # via the model / datamodule reference graph. Explicitly:
                #   1. Move the model to CPU (sub-modules + buffers).
                #   2. Empty the CUDA cache (releases tensor memory).
                #   3. Synchronise to flush any pending GPU ops so the next
                #      torch op doesn't replay the failed kernel.
                #   4. Build a fresh CPU-only Trainer.
                try:
                    self.model.to("cpu")
                    if hasattr(self.model, "_orig_mod"):
                        self.model._orig_mod.to("cpu")
                except Exception:
                    pass  # best-effort: model may already be on CPU
                # Reset CUDA state best-effort. ``empty_cache`` is a no-op
                # when CUDA isn't initialised; ``synchronize`` only fires
                # when CUDA is available. ``ipc_collect`` releases inter-
                # process tensor references on Windows where mmap can hold
                # GPU memory alive across the failed predict.
                try:
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                except Exception:
                    pass
                _cpu_params = {
                    "accelerator": "cpu",
                    "devices": 1,
                    "logger": False,
                    "enable_checkpointing": False,
                    "enable_progress_bar": False,
                }
                cpu_trainer = L.Trainer(**_cpu_params)
                predictions = cpu_trainer.predict(
                    model=self.model,
                    datamodule=datamodule,
                )
            except Exception as e_cpu:
                # iter341 (2026-05-27): if CPU fallback also fails, the
                # CUDA context is permanently invalidated (verified on
                # real concurrent-GPU contention 2026-05-27 c0014). To
                # let the suite progress without GPU acceleration for
                # the remainder of this process, hard-disable CUDA at
                # the torch module level so subsequent estimators do
                # not even try CUDA. The next predict / fit call then
                # falls through to CPU naturally instead of hitting
                # the same dirty-context error.
                try:
                    _cuda_msg = str(e_cpu)
                    _is_still_cuda = any(
                        fp in _cuda_msg for fp in _cuda_fingerprints
                    )
                except Exception:
                    _is_still_cuda = False
                if _is_still_cuda:
                    logger.error(
                        "CPU fallback after CUDA prediction failure ALSO failed "
                        "with a CUDA-side error: %s. The CUDA context is "
                        "permanently invalidated for this process. Disabling "
                        "CUDA at the torch module level so subsequent "
                        "estimators skip GPU and run on CPU; GPU acceleration "
                        "will resume on the next process restart. Original "
                        "CUDA error: %s",
                        e_cpu, e,
                    )
                    try:
                        # Hide CUDA from torch for the remainder of this
                        # process. The env var only helps if torch hasn't
                        # imported yet; the monkey-patch on
                        # torch.cuda.is_available is the load-bearing piece.
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        torch.cuda.is_available = lambda: False
                    except Exception:
                        pass
                    # iter420 (2026-05-27): explicitly move self.model
                    # parameters to CPU BEFORE the retry. Hiding CUDA at
                    # the module level does NOT relocate tensors that are
                    # already on the (invalidated) GPU; Lightning's
                    # accelerator='cpu' then crashes trying to operate
                    # on GPU-resident weights with a broken context.
                    # Surfaced on c0005 LTR run 2026-05-27: even after
                    # iter341's CUDA-hide, the second CPU retry raised
                    # the same CUDA illegal-memory-access error because
                    # ``self.model.parameters()`` were still cuda:0
                    # tensors. ``.to('cpu')`` reads from GPU memory
                    # which is exactly what's broken -- so do it inside
                    # try/except and continue regardless; if the move
                    # itself fails the Trainer call will still raise
                    # cleanly with the original CUDA error.
                    try:
                        self.model.to("cpu")
                    except Exception as _e_move:
                        logger.error(
                            "Failed to move model parameters off the "
                            "invalidated GPU context (%s); the CPU retry "
                            "below will likely re-raise the CUDA error.",
                            _e_move,
                        )
                    # Retry one more time on CPU now that CUDA is hidden
                    # AND model weights are CPU-resident.
                    try:
                        _cpu_params2 = {
                            "accelerator": "cpu",
                            "devices": 1,
                            "logger": False,
                            "enable_checkpointing": False,
                            "enable_progress_bar": False,
                        }
                        cpu_trainer2 = L.Trainer(**_cpu_params2)
                        predictions = cpu_trainer2.predict(
                            model=self.model,
                            datamodule=datamodule,
                        )
                    except Exception as e_cpu2:
                        logger.error(
                            "Even with CUDA hidden the predict failed: %s. "
                            "Re-raising original CUDA error.",
                            e_cpu2,
                        )
                        raise
                else:
                    logger.error(
                        "CPU fallback after CUDA prediction failure ALSO failed "
                        "with a non-CUDA error: %s. Original CUDA error: %s",
                        e_cpu, e,
                    )
                    raise
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
        raw = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)
        # F-05 binary path: raw has shape (N, 1) and contains P(y=1).
        # Use ``> 0.5`` (strict) so the predict() / predict_proba contract holds:
        # at raw==0.5 a ``>= 0.5`` threshold returned class 1 while
        # ``argmax(predict_proba)`` -> ``argmax([0.5, 0.5])`` returns 0 (numpy
        # tie-break to first index). The strict comparison aligns the two
        # public methods on collapsed / under-trained models where many rows
        # sit at exactly 0.5.
        if getattr(self, "_binary_sigmoid_head", False):
            idx = (raw.reshape(-1) > 0.5).astype(np.int64)
        else:
            idx = np.argmax(raw, axis=1)
        # sklearn convention: ``predict`` returns LABELS (entries of
        # ``classes_``), not argmax INDICES. Pre-fix this returned the bare
        # indices, which the downstream reporting layer band-aided with
        # ``model.classes_[preds]`` (see _reporting_probabilistic.py:266);
        # any direct ``accuracy_score(y, model.predict(X))`` silently
        # miscalled for any y whose value set was not ``{0..K-1}``. F-01 in
        # the 2026-05-30 mlp audit. The ``_label_encoder`` branch is the
        # canonical path; ``classes_`` direct indexing covers estimators
        # loaded from an older pickle that has classes_ but no encoder; the
        # final ``return idx`` covers multilabel / dropped-state cases.
        if getattr(self, "_label_encoder", None) is not None:
            return self._label_encoder.inverse_transform(idx)
        if getattr(self, "classes_", None) is not None:
            return self.classes_[idx]
        return idx

    def predict_proba(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Input data
            device: Optional device string ('cpu' or 'cuda')
            precision: Optional precision mode for inference
            batch_size: Optional batch size for prediction

        Returns:
            numpy.ndarray: Predicted class probabilities, shape (N, K).
        """
        raw = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)
        # F-05 binary path: raw has shape (N, 1) with P(y=1); stack
        # [1-p, p] to honour the sklearn (N, 2) ``predict_proba`` contract.
        # The column order matches ``classes_`` (sorted): col 0 = P(class[0]),
        # col 1 = P(class[1]).
        if getattr(self, "_binary_sigmoid_head", False):
            p1 = raw.reshape(-1)
            return np.column_stack([1.0 - p1, p1])
        return raw


# Callback classes carved to ``_base_callbacks.py``. Re-exports preserve class identity so downstream isinstance / Trainer callback-list checks keep working unchanged.
from ._base_callbacks import (  # noqa: F401, E402
    NetworkGraphLoggingCallback,
    AggregatingValidationCallback,
    ValLossDivergenceCallback,
    BestEpochModelCheckpoint,
    PeriodicLearningRateFinder,
)
