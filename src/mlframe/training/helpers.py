"""
Training helper functions and callback classes.

This module contains helper utilities:
- parse_catboost_devices: GPU device parsing for CatBoost
- get_training_configs: Training configuration factory
- get_trainset_features_stats: Compute training set statistics (pandas)
- get_trainset_features_stats_polars: Compute training set statistics (polars)
- UniversalCallback: Base callback class for training monitoring
- LightGBMCallback, XGBoostCallback, CatBoostCallback: Model-specific callbacks
"""

from __future__ import annotations


import logging
import psutil
from dataclasses import dataclass
from timeit import default_timer as timer
from types import SimpleNamespace
from typing import Optional, Dict, List, Callable, Sequence, Any, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

# NOTE: torch + mlframe.lightninglib are imported lazily inside `get_training_configs`
# (only needed for MLP configs). Top-level import cost ~2-3s — avoided for CB/LGB/XGB-only runs.
import lightgbm as lgb

import xgboost as xgb
from xgboost.callback import TrainingCallback

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from pyutilz.system import get_gpuinfo_gpu_info, tqdmu, get_own_memory_usage
from pyutilz.pythonlib import get_parent_func_args, store_params_in_object
from ._gpu_probe import CUDA_IS_AVAILABLE, LGB_GPU_AVAILABLE, XGB_GPU_AVAILABLE
from mlframe.metrics.core import (
    compute_probabilistic_multiclass_error,
    robust_mlperf_metric,
    ICE,
)

from .utils import get_numeric_columns, get_categorical_columns
# `_probe_xgb_gpu_support` / `_probe_lgb_gpu_support` re-export was dropped:
# they are private to `_gpu_probe.py` (only used to compute the module-level
# XGB_GPU_AVAILABLE / LGB_GPU_AVAILABLE booleans imported above), and no other
# module ever imported them via this re-export. Removing closes the "noqa F401
# on private name" anti-pattern flagged in the Wave-3 audit.
from ._classif_helpers import (  # noqa: E402,F401
    _canonical_predict_proba_shape,
    _predict_from_probs,
    _classif_objective_kwargs,
    _maybe_wrap_multilabel,
    _compute_chain_orders,
    _ChainEnsemble,
    _build_classifier_chain_ensemble,
)  # _build_classifier_chain_ensemble kept in helpers.py (line 86)
from ._callbacks import (  # noqa: E402,F401
    UniversalCallback,
    LightGBMCallback,
    XGBoostCallback,
    CatBoostCallback,
)

logger = logging.getLogger(__name__)


# Constant - CUDA availability
try:
    from numba.cuda import is_available as is_cuda_available

    CUDA_IS_AVAILABLE = is_cuda_available()
except (ImportError, AttributeError, ModuleNotFoundError):
    CUDA_IS_AVAILABLE = False


# 2026-05-10: per-library GPU support gating. ``CUDA_IS_AVAILABLE``
# (numba probe) tells us only whether the system has a usable CUDA
# device -- it does NOT tell us whether the installed XGBoost / LightGBM
# binaries were COMPILED with GPU support. On Windows, the default
# ``pip install xgboost`` ships the CPU-only wheel, so passing
# ``device='cuda'`` triggers ``WARNING: Device is changed from GPU to
# CPU as we couldn't find any available GPU on the system`` per-fit
# (verified on prod log 2026-05-09 with a custom 3rdParty XGB build:
# ``xgb.build_info() == {... 'USE_CUDA': False ...}`` despite CUDA
# being available for CatBoost). Probe each binary's actual GPU
# support ONCE at module-import so the helpers never set device='cuda'

def parse_catboost_devices(devices: str, all_gpus: list = None) -> List[Dict]:
    """
    Parses a GPU devices string and returns a list of GPU info dicts
    corresponding to the specified device indices.

    Parameters
    ----------
    devices : str
        A string specifying device indices. Formats supported:
          - "0"             (single GPU)
          - "0:1:3"         (multiple GPUs)
          - "0-3"           (range of GPUs, inclusive)

    Returns
    -------
    list[dict]
        Filtered list of GPU info dictionaries.
    """

    if not all_gpus:
        all_gpus = get_gpuinfo_gpu_info()

    if not devices:
        return all_gpus

    # Parse the devices string
    device_indices = []
    try:
        if "-" in devices:  # range format
            parts = devices.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range format '{devices}'. Expected 'start-end' (e.g., '0-3')")
            start, end = parts
            start_int, end_int = int(start), int(end)
            if start_int > end_int:
                raise ValueError(f"Invalid range '{devices}': start ({start_int}) > end ({end_int})")
            device_indices = list(range(start_int, end_int + 1))
        elif ":" in devices:  # multiple specific GPUs
            device_indices = [int(x) for x in devices.split(":")]
        else:  # single GPU
            device_indices = [int(devices)]
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid device specification '{devices}'. Must contain integers only.") from e
        raise

    # Validate indices
    max_index = len(all_gpus) - 1
    invalid = [i for i in device_indices if i < 0 or i > max_index]
    if invalid:
        raise ValueError(f"Invalid GPU indices {invalid}. Available range: 0-{max_index}")

    # Filter GPU list
    filtered_gpus = [gpu for gpu in all_gpus if gpu["index"] in device_indices]
    return filtered_gpus


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Training Configuration Factory
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_training_configs(
    iterations: int = 5000,
    early_stopping_rounds: Optional[int] = 0,
    validation_fraction: float = 0.1,
    use_explicit_early_stopping: bool = True,
    has_time: bool = True,
    has_gpu: bool = None,
    subgroups: dict = None,
    learning_rate: float = 0.1,
    def_regr_metric: str = "MAE",
    def_classif_metric: str = "AUC",
    # 2026-04-24: target_type-aware classifier objective injection.
    # When target_type is BINARY_CLASSIFICATION (default), the existing
    # binary objectives ("binary:logistic" / "binary" etc.) are kept.
    # For MULTICLASS / MULTILABEL, _classif_objective_kwargs replaces
    # them with the right native dispatch ("multi:softprob"+num_class,
    # "MultiLogloss", etc.).
    target_type: Optional[Any] = None,  # TargetTypes; None = legacy binary
    n_classes: int = 2,
    catboost_custom_classif_metrics: Optional[Sequence] = None,
    catboost_custom_regr_metrics: Optional[Sequence] = None,
    random_seed: Optional[int] = None,
    verbose: int = 0,
    # ----------------------------------------------------------------------------------------------------------------------------
    # probabilistic errors
    # ----------------------------------------------------------------------------------------------------------------------------
    method: str = "multicrit",
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    pr_auc_weight: float = 0.1,
    brier_loss_weight: float = 0.8,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 10,
    # ----------------------------------------------------------------------------------------------------------------------------
    # robustness parameters for early stopping metric
    # ----------------------------------------------------------------------------------------------------------------------------
    robustness_num_ts_splits: int = 0,  # 0 = disabled, >0 = number of consecutive time splits
    robustness_std_coeff: float = 0.1,  # multiplier for std penalty
    robustness_greater_is_better: bool = False,  # False for ICE (lower is better)
    # ----------------------------------------------------------------------------------------------------------------------------
    # model-specific params
    # ----------------------------------------------------------------------------------------------------------------------------
    cb_kwargs: dict = None,
    hgb_kwargs: dict = None,
    lgb_kwargs: dict = None,
    xgb_kwargs: dict = None,
    mlp_kwargs: dict = None,
    ngb_kwargs: dict = None,
    # 2026-05-12: first-class predict-time MLP batch override. When None
    # (default) the wrapper auto-adapts to memory + input width. Plumbed in
    # from ``ModelHyperparamsConfig.mlp_predict_batch_size`` so callers don't
    # need to dig into ``mlp_kwargs["datamodule_params"]``.
    mlp_predict_batch_size: Optional[int] = None,
    # ----------------------------------------------------------------------------------------------------------------------------
    # featureselectors
    # ----------------------------------------------------------------------------------------------------------------------------
    rfecv_kwargs: dict = None,
    # ----------------------------------------------------------------------------------------------------------------------------
    # 2026-05-08 perf: skip MLP-config build (incl. ~14s pytorch / lightning
    # import on first call) when MLP isn't in the requested model list.
    # Default ``None`` preserves legacy behaviour (build all configs).
    # ----------------------------------------------------------------------------------------------------------------------------
    enabled_models: Optional[Sequence[str]] = None,
) -> tuple:
    """Returns comparable training configs for different types of models,
    based on general params supplied like learning rate, task type, time budget.
    Useful for more or less fair comparison between different models on the same data/task, and their upcoming ensembling.
    This procedure is good for getting the feeling of what ML models are capable of for a particular task.
    """

    if has_gpu is None:
        has_gpu = CUDA_IS_AVAILABLE

    # Initialize mutable defaults
    if catboost_custom_classif_metrics is None:
        catboost_custom_classif_metrics = ["AUC", "BrierScore", "PRAUC"]
    if catboost_custom_regr_metrics is None:
        catboost_custom_regr_metrics = ["RMSE", "MAPE"]

    # Initialize kwargs dicts with defaults, making copies to avoid mutating caller's dicts
    if cb_kwargs is None:
        cb_kwargs = dict(verbose=0)
    else:
        cb_kwargs = cb_kwargs.copy()  # Don't mutate caller's dict
    if lgb_kwargs is None:
        lgb_kwargs = dict(verbose=-1)
    else:
        lgb_kwargs = lgb_kwargs.copy()  # Don't mutate caller's dict
    if xgb_kwargs is None:
        xgb_kwargs = dict(verbosity=0)
    else:
        xgb_kwargs = xgb_kwargs.copy()  # Don't mutate caller's dict
    if hgb_kwargs is None:
        hgb_kwargs = dict(verbose=0)
    else:
        hgb_kwargs = hgb_kwargs.copy()
    if mlp_kwargs is None:
        mlp_kwargs = dict()
    else:
        mlp_kwargs = mlp_kwargs.copy()
    if ngb_kwargs is None:
        ngb_kwargs = dict(verbose=True)
    else:
        ngb_kwargs = ngb_kwargs.copy()

    # None = disabled (don't pass to model fit at all); 0 = auto (iterations // 3); int = as-is.
    early_stopping_disabled = early_stopping_rounds is None
    if not early_stopping_disabled and not early_stopping_rounds:
        early_stopping_rounds = max(2, iterations // 3)

    def neg_ovr_roc_auc_score(*args, **kwargs):
        return -roc_auc_score(*args, **kwargs, multi_class="ovr")

    # Build defaults, then let caller's kwargs override any of them
    # via .update(). Using **cb_kwargs for merge crashes when the
    # caller passes a key that's already in the defaults dict
    # (TypeError: got multiple values).
    # ``has_gpu`` reports nvidia-smi presence; the installed catboost wheel
    # may still be CPU-only (default PyPI Windows wheels). Confirm via
    # ``_cb_gpu_usable`` (one-shot tiny-fit probe, cached) so machines with
    # a working GPU but a CPU-only CB binary fall back to ``task_type="CPU"``
    # instead of crashing every fit with "Environment for task type [GPU]
    # not found". Skip the probe entirely when CB is not in scope - the
    # tiny CB-GPU fit costs ~150ms per process and is wasted on
    # linear/ridge/lgb/xgb-only suites. ``models_in_scope`` is a hint;
    # when None we keep the conservative behaviour and probe.
    _cb_in_scope = (
        enabled_models is None
        or any(str(m).lower() in ("cb", "catboost") for m in enabled_models)
    )
    if has_gpu and _cb_in_scope:
        from ._cb_pool import _cb_gpu_usable as _cb_gpu_probe
        _cb_task = "GPU" if _cb_gpu_probe() else "CPU"
    else:
        _cb_task = "CPU"
    CB_GENERAL_PARAMS = dict(
        iterations=iterations,
        has_time=has_time,
        learning_rate=learning_rate,
        eval_fraction=(0.0 if use_explicit_early_stopping else validation_fraction),
        task_type=_cb_task,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        # metric_period=5: evaluate the custom eval metric every 5th boost
        # iteration instead of every iteration. On 1M-row multiclass with
        # the ICE calibration metric this cut suite wall from ~95s to ~60s
        # (the per-iter metric was 110ms and CB ran 350+ boost rounds).
        # Trade-off: early-stopping detects "best iteration" with a 5-iter
        # granularity instead of 1-iter; on a 100+-iter run this is a
        # negligible accuracy hit. CB caller can override via cb_kwargs.
        metric_period=5,
    )
    CB_GENERAL_PARAMS.update(cb_kwargs)

    CB_CLASSIF = CB_GENERAL_PARAMS.copy()
    CB_CLASSIF.update({"eval_metric": def_classif_metric})
    # NOTE: custom_metric breaks sklearn.clone() - CatBoost modifies this param after init.
    # TODO(2026-04-28): Raise issue at https://github.com/catboost/catboost/issues
    # "custom_metric": tuple(catboost_custom_classif_metrics or [])

    CB_REGR = CB_GENERAL_PARAMS.copy()
    CB_REGR.update({"eval_metric": def_regr_metric})
    # NOTE: custom_metric breaks sklearn.clone() - CatBoost modifies this param after init.
    # TODO(2026-04-28): Raise issue at https://github.com/catboost/catboost/issues
    # "custom_metric": tuple(catboost_custom_regr_metrics or [])

    HGB_GENERAL_PARAMS = dict(
        max_iter=iterations,
        learning_rate=learning_rate,
        early_stopping=True,
        validation_fraction=(None if use_explicit_early_stopping else validation_fraction),
        n_iter_no_change=early_stopping_rounds,
        categorical_features="from_dtype",
        random_state=random_seed,
    )
    HGB_GENERAL_PARAMS.update(hgb_kwargs)

    # 2026-05-10: device gating now reflects XGB build-info, not just
    # CUDA presence. ``has_gpu and XGB_GPU_AVAILABLE`` skips ``cuda`` on
    # CPU-only XGB binaries (avoids per-fit ``Device is changed from GPU
    # to CPU`` warning storm). Caller's xgb_kwargs.update overrides win
    # for advanced users.
    _xgb_device = "cuda" if (has_gpu and XGB_GPU_AVAILABLE) else "cpu"
    XGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,  # affects model size heavily when high cardinality cat features r present!
        tree_method="hist",
        device=_xgb_device,
        n_jobs=psutil.cpu_count(logical=False),
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
    )
    XGB_GENERAL_PARAMS.update(xgb_kwargs)

    XGB_GENERAL_CLASSIF = XGB_GENERAL_PARAMS.copy()
    XGB_GENERAL_CLASSIF.update({"objective": "binary:logistic", "eval_metric": neg_ovr_roc_auc_score})

    # 2026-04-24: target_type-aware objective injection. For non-binary
    # classification target types, replace the binary defaults with the
    # native multi-output objective. Binary path is a no-op (helper
    # returns the same kwargs as the explicit defaults above).
    from .configs import TargetTypes

    _resolved_tt = target_type if target_type is not None else TargetTypes.BINARY_CLASSIFICATION
    if _resolved_tt.is_classification and not _resolved_tt.is_binary:
        # Non-binary classification: inject native objective per library.
        cb_obj = _classif_objective_kwargs("catboost", _resolved_tt, n_classes)
        xgb_obj = _classif_objective_kwargs("xgboost", _resolved_tt, n_classes)
        lgb_obj = _classif_objective_kwargs("lightgbm", _resolved_tt, n_classes)
        if cb_obj:
            CB_CLASSIF.update(cb_obj)
            # 2026-04-24 Session 6: when loss_function=MultiLogloss (multilabel),
            # CB REJECTS eval_metric='AUC' with "metric AUC and loss MultiLogloss
            # are incompatible". Override to HammingLoss for MultiLogloss,
            # Accuracy for MultiClass. Caller can still override via cb_kwargs.
            if cb_obj.get("loss_function") == "MultiLogloss":
                CB_CLASSIF["eval_metric"] = "HammingLoss"
            elif cb_obj.get("loss_function") == "MultiClass":
                CB_CLASSIF["eval_metric"] = "Accuracy"
        if xgb_obj:
            # For multiclass, multi:softprob conflicts with binary metric.
            # Strip the binary eval_metric — caller can re-set if needed.
            XGB_GENERAL_CLASSIF.update(xgb_obj)
            # XGB multiclass eval_metric: mlogloss aligns with multi:softprob.
            # (binary binary_logloss / AUC don't apply.)
            if xgb_obj.get("objective") == "multi:softprob":
                XGB_GENERAL_CLASSIF["eval_metric"] = "mlogloss"
        if lgb_obj:
            # LGB_GENERAL_PARAMS gets the multiclass objective too — it has
            # no separate _CLASSIF variant currently.
            pass  # applied to LGB after LGB_GENERAL_PARAMS is built (below)
        # NOTE: _mlframe_target_type metadata tag was historically attached
        # here but REMOVED 2026-04-24 Session 6 — CatBoostClassifier init
        # raises TypeError on unknown kwargs, blocking the entire multilabel
        # path. Downstream observability (which lib + target_type) is covered
        # by the per-model model_schemas metadata record populated in
        # core.py around the fit call. Adding a side-channel tag here was
        # a premature optimisation that forked a 4-year-stable init API
        # contract for a diagnostic that's available elsewhere.

    def integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, verbose: bool = False) -> float:
        """Compute integral calibration error for probabilistic predictions.

        Wraps compute_probabilistic_multiclass_error with the outer function's
        configuration parameters (method, weights, etc.).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_score : np.ndarray
            Predicted probabilities.
        verbose : bool, default=False
            If True, print calibration error info.

        Returns
        -------
        float
            The computed calibration error (lower is better).
        """
        err = compute_probabilistic_multiclass_error(
            y_true=y_true,
            y_score=y_score,
            method=method,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        if verbose:
            print(len(y_true), "integral_calibration_error=", err)
        return err

    def make_robust_ts_metric(
        metric_fn,
        num_splits: int,
        std_coeff: float,
        greater_is_better: bool,
        min_samples_per_split: int = 100,
        ensure_enough_classes: bool = False,
        verbose: int = 0,
    ):
        """Wrap a metric to evaluate across consecutive time splits.

        Returns mean(metric_values) ± std(metric_values) * std_coeff
        where ± is + if greater_is_better=False (penalize variance for minimization)
              and - if greater_is_better=True (penalize variance for maximization)
        """

        def robust_metric(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):
            n = len(y_true)

            # Fallback 1: Not enough data for any splits
            if n < min_samples_per_split:
                if verbose:
                    logger.info("make_robust_ts_metric: n=%s < min_samples_per_split=%s, using full data", n, min_samples_per_split)
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Compute actual number of splits we can do
            actual_splits = min(num_splits, n // min_samples_per_split)

            # Fallback 2: Can only do 1 split
            if actual_splits <= 1:
                if verbose:
                    logger.info("make_robust_ts_metric: actual_splits=%s <= 1, using full data", actual_splits)
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Split into consecutive intervals
            split_size = n // actual_splits
            values = []

            for i in range(actual_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < actual_splits - 1 else n

                y_true_split = y_true[start_idx:end_idx]
                y_score_split = y_score[start_idx:end_idx]

                # Skip split if not enough samples
                if len(y_true_split) < min_samples_per_split:
                    if verbose:
                        logger.info("make_robust_ts_metric: split %s skipped, len=%d < %d", i, len(y_true_split), min_samples_per_split)
                    continue

                # Skip split if single class (classification only)
                if ensure_enough_classes and len(np.unique(y_true_split)) < 2:
                    if verbose:
                        logger.info("make_robust_ts_metric: split %s skipped, single class in y_true", i)
                    continue

                val = metric_fn(y_true_split, y_score_split, *args, **kwargs)
                if not np.isnan(val):
                    values.append(val)

            # Fallback 3: No valid splits computed
            if len(values) == 0:
                if verbose:
                    logger.info("make_robust_ts_metric: no valid splits, using full data")
                return metric_fn(y_true, y_score, *args, **kwargs)

            # Fallback 4: Only one valid split
            if len(values) == 1:
                if verbose:
                    logger.info("make_robust_ts_metric: only 1 valid split, returning %.6f", values[0])
                return values[0]

            mean_val = np.mean(values)
            std_val = np.std(values)

            if verbose:
                logger.info("make_robust_ts_metric: %d splits, mean=%.6f, std=%.6f", len(values), mean_val, std_val)

            # Penalize high variance
            if greater_is_better:
                # For maximization: subtract std penalty (lower result = worse)
                return mean_val - std_val * std_coeff
            else:
                # For minimization: add std penalty (higher result = worse)
                return mean_val + std_val * std_coeff

        return robust_metric

    if subgroups:

        def final_integral_calibration_error(y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):  # partial won't work with xgboost
            return robust_mlperf_metric(
                y_true,
                y_score,
                *args,
                metric=integral_calibration_error,
                higher_is_better=False,
                subgroups=subgroups,
                **kwargs,
            )

    else:
        final_integral_calibration_error = integral_calibration_error

    # Apply robustness wrapper if enabled
    if robustness_num_ts_splits > 0:
        final_integral_calibration_error = make_robust_ts_metric(
            final_integral_calibration_error,
            num_splits=robustness_num_ts_splits,
            std_coeff=robustness_std_coeff,
            greater_is_better=robustness_greater_is_better,
            ensure_enough_classes=True,  # ICE is for classification
            verbose=verbose,
        )

    def fs_and_hpt_integral_calibration_error(*args, verbose: bool = True, **kwargs):
        err = compute_probabilistic_multiclass_error(
            *args,
            **kwargs,
            mae_weight=mae_weight,
            std_weight=std_weight,
            brier_loss_weight=brier_loss_weight,
            roc_auc_weight=roc_auc_weight,
            pr_auc_weight=pr_auc_weight,
            min_roc_auc=min_roc_auc,
            roc_auc_penalty=roc_auc_penalty,
            use_weighted_calibration=use_weighted_calibration,
            weight_by_class_npositives=weight_by_class_npositives,
            nbins=nbins,
            verbose=verbose,
        )
        return err

    XGB_CALIB_CLASSIF = XGB_GENERAL_CLASSIF.copy()
    XGB_CALIB_CLASSIF.update({"eval_metric": final_integral_calibration_error})

    def lgbm_integral_calibration_error(y_true, y_score):
        metric_name = "integral_calibration_error"
        value = final_integral_calibration_error(y_true, y_score)
        higher_is_better = False
        return metric_name, value, higher_is_better

    CB_CALIB_CLASSIF = CB_CLASSIF.copy()
    # 2026-04-24 Session 6: ICE custom-metric only works for single-target
    # CB objectives (binary/multiclass). For MultiLogloss (multilabel), CB
    # asserts the custom metric inherits from MultiTargetCustomMetric. Until
    # we ship a multi-target ICE variant, fall back to HammingLoss for
    # multilabel — same as CB_CLASSIF (so calibrated path == base path).
    if _resolved_tt.is_classification and not _resolved_tt.is_binary and CB_CLASSIF.get("loss_function") == "MultiLogloss":
        # eval_metric already set to HammingLoss above; keep it.
        pass
    else:
        CB_CALIB_CLASSIF.update({"eval_metric": ICE(metric=final_integral_calibration_error, higher_is_better=False, max_arr_size=0)})

    # 2026-05-10: same gating story as XGB. ``has_gpu and LGB_GPU_AVAILABLE``
    # respects the LightGBM build's actual GPU support (default LightGBM
    # wheels are CPU-only; opt-in via env ``MLFRAME_TRUST_LGB_CUDA=1``
    # if you've built / installed a GPU-enabled LGB binary).
    _lgb_device = "cuda" if (has_gpu and LGB_GPU_AVAILABLE) else "cpu"
    LGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        early_stopping_rounds=early_stopping_rounds,
        device_type=_lgb_device,
        random_state=random_seed,
        # histogram_pool_size=16384,
    )
    LGB_GENERAL_PARAMS.update(lgb_kwargs)
    # Target-type-aware objective for LGB (no separate _CLASSIF variant).
    if _resolved_tt.is_classification and not _resolved_tt.is_binary:
        _lgb_obj = _classif_objective_kwargs("lightgbm", _resolved_tt, n_classes)
        if _lgb_obj:
            LGB_GENERAL_PARAMS.update(_lgb_obj)
            LGB_GENERAL_PARAMS["_mlframe_target_type"] = str(_resolved_tt.value)

    NGB_GENERAL_PARAMS = dict(
        n_estimators=iterations,
        learning_rate=learning_rate,
    )
    NGB_GENERAL_PARAMS.update(ngb_kwargs)

    # 2026-05-08 perf: when caller declares which models are in scope
    # AND mlp / recurrent are NOT among them, skip the entire MLP
    # config block (saves ~14s of pytorch + lightning import overhead
    # on the first call to get_training_configs in a process). Any
    # caller that asks for MLP_GENERAL_PARAMS later will get None.
    _mlp_in_scope = (
        enabled_models is None
        or any(m in ("mlp", "recurrent") for m in enabled_models)
    )

    if not _mlp_in_scope:
        # Skip the heavy MLP path entirely. Downstream consumers must
        # not assume MLP_GENERAL_PARAMS is non-None when mlp isn't
        # requested -- the dispatch in trainer.py's _get_mlp_imports
        # gates on the presence of 'mlp' in mlframe_models already.
        MLP_GENERAL_PARAMS = None
    else:
        # Note: ``num_sanity_val_steps`` is intentionally left at
        # Lightning's default (2). A 2026-05-09 A/B benchmark on the
        # tightest MLP combo (c0120: cb+mlp pandas n=300 multiclass)
        # showed setting it to 0 saves only 32 ms / 1.7% per fit
        # (within combined stdev), because the "Sanity Checking" pass
        # is just the first forward through cuDNN's auto-tuner --
        # disabling it pushes the same kernel selection cost onto
        # epoch 0 instead. Keeping the default preserves Lightning's
        # fail-fast on a broken val pipeline at no measurable cost.
        # 2026-05-12: AMP precision is now auto-resolved on Ampere+ CUDA hosts
        # (RTX 30/40, A100, H100) -> "bf16-mixed" for ~2x throughput on tabular
        # MLPs. Falls back to "32-true" on older GPUs / CPU. The user override
        # under mlp_kwargs["trainer_params"]["precision"] always wins.
        from .mlp_runtime_defaults import resolve_mlp_precision_default
        _user_precision = (
            (mlp_kwargs or {}).get("trainer_params", {}).get("precision")
        )
        _resolved_precision = resolve_mlp_precision_default(
            user_override=_user_precision,
        )

        mlp_trainer_params: dict = dict(
            devices=1,  # Always use single device by default to avoid multi-GPU complexity
            # ------------------------------------------------------------------
            # Runtime:
            # ------------------------------------------------------------------
            min_epochs=1,
            max_epochs=iterations,
            # 2026-05-12: hard 30-min cap on a 4M-row TVT regression got the
            # MLP stopped at ~6 epochs (RMSE 585 vs Linear 252). Default
            # raised to 2h so a slow DataLoader path can still converge; the
            # caller-side ``early_stopping_rounds`` (default 100 epochs of
            # no val improvement) still terminates well before then on
            # healthy training. Override via
            # ``mlp_kwargs["trainer_params"]["max_time"]``.
            max_time={"days": 0, "hours": 2, "minutes": 0},
            # ------------------------------------------------------------------
            # Intervals:
            # ------------------------------------------------------------------
            check_val_every_n_epoch=1,
            # ------------------------------------------------------------------
            # Flags:
            # ------------------------------------------------------------------
            enable_model_summary=False,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=2,
            # ------------------------------------------------------------------
            # Precision & accelerators:
            # ------------------------------------------------------------------
            precision=_resolved_precision,
            num_nodes=1,
            # ------------------------------------------------------------------
            # Logging:
            # ------------------------------------------------------------------
            default_root_dir="logs",
        )

        if mlp_kwargs:
            mlp_trainer_params.update(mlp_kwargs.get("trainer_params", {}))

        # Lazy imports -- only paid when MLP configs are actually being built.
        import torch
        import torch.nn.functional as F
        from mlframe.training.neural.flat import MLPTorchModel
        from mlframe.training.neural.data import TorchDataModule

        # Default loss function and dtype (classification)
        loss_fn = F.cross_entropy
        labels_dtype = torch.int64

        # 2026-05-13 (TVT-failure root cause): defaults switched from
        # AdamW + LR=1e-3 to Adam + LR=3e-3.
        #
        # Why Adam (not AdamW): AdamW's built-in weight_decay=0.01
        # penalises large weights -- which is EXACTLY what an MLP needs
        # to learn a near-linear target with one dominant feature (y =
        # 0.95 * TVT_prev + small_residual requires a weight of ~0.95
        # on the dominant input). Weight decay shrinks that weight every
        # step, fighting the loss. Adam (no decay) is the safer default
        # for tabular regression with strong linear / additive signal.
        # Users whose dataset is overfit-prone can opt back in via
        # ``mlp_kwargs["model_params"]["optimizer"]=torch.optim.AdamW``.
        #
        # Why LR=3e-3 (not 1e-3): with the new zero-dropout architecture,
        # gradient flow is unobstructed; the larger LR converges in ~1/3
        # the epochs without overshoot. On the 2-hour TVT run with
        # LR=1e-3 + dropout=0.15, MLP plateaued at val_MSE=0.7555 after 9
        # epochs (out of ~20 the time budget allowed); with LR=3e-3 +
        # dropout=0, the same architecture converges to val_MSE ~ 0.15
        # in similar epoch count -- matching linear regression on a
        # near-linear DGP.
        mlp_model_params = dict(
            loss_fn=loss_fn,
            learning_rate=3e-3,
            l1_alpha=0.0,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
        )
        if mlp_kwargs:
            mlp_model_params.update(mlp_kwargs.get("model_params", {}))

        # 2026-05-12: cross-platform-safe DataLoader defaults. num_workers
        # stays at 0 on EVERY OS because ``TorchDataset`` keeps the full
        # input frame in self.features. On Windows that means each worker
        # pickles 100 GB; on Linux fork's CoW gets broken by Polars indexing
        # + Python refcount writes -> swap death. pin_memory still defaults
        # ON for CUDA hosts (no IPC landmine). User opts in to workers via
        # ``mlp_kwargs["dataloader_params"]["num_workers"]`` once their
        # specific dataset is verified to fit each worker's memory budget.
        from .mlp_runtime_defaults import resolve_mlp_dataloader_defaults
        _user_dataloader_overrides = (
            (mlp_kwargs or {}).get("dataloader_params", {}) or {}
        )
        _resolved_dataloader_extras = resolve_mlp_dataloader_defaults(
            user_overrides=_user_dataloader_overrides,
        )

        mlp_dataloader_params = dict(
            sampler=None,
            batch_sampler=None,
            num_workers=_resolved_dataloader_extras["num_workers"],
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=_resolved_dataloader_extras["prefetch_factor"],
            persistent_workers=_resolved_dataloader_extras["persistent_workers"],
            pin_memory=_resolved_dataloader_extras["pin_memory"],
            batch_size="auto",
            shuffle=False,
        )
        if mlp_kwargs:
            mlp_dataloader_params.update(mlp_kwargs.get("dataloader_params", {}))

        mlp_datamodule_params = dict(
            read_fcn=None, data_placement_device=None,
            features_dtype=torch.float32, labels_dtype=labels_dtype,
            dataloader_params=mlp_dataloader_params,
        )
        # 2026-05-12: plumb the suite-level predict-batch-size knob through
        # the datamodule -- the wrapper's _predict_raw consults this when no
        # explicit batch_size is passed to .predict(). None (default) lets
        # the adaptive resolver pick based on memory + dataframe width.
        if mlp_predict_batch_size is not None:
            mlp_datamodule_params["predict_batch_size"] = int(mlp_predict_batch_size)
        if mlp_kwargs:
            mlp_datamodule_params.update(mlp_kwargs.get("datamodule_params", {}))

        MLP_GENERAL_PARAMS = dict(
            model_class=MLPTorchModel,
            model_params=mlp_model_params,
            datamodule_class=TorchDataModule,
            datamodule_params=mlp_datamodule_params,
            trainer_params=mlp_trainer_params,
            use_swa=mlp_kwargs.get("use_swa", False) if mlp_kwargs else False,
            swa_params=(
                mlp_kwargs.get("swa_params", dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4))
                if mlp_kwargs
                else dict(swa_epoch_start=5, annealing_epochs=5, swa_lrs=1e-4)
            ),
            tune_params=mlp_kwargs.get("tune_params", False) if mlp_kwargs else False,
            float32_matmul_precision=mlp_kwargs.get("float32_matmul_precision", None) if mlp_kwargs else None,
            early_stopping_rounds=early_stopping_rounds,
        )

    if rfecv_kwargs is None:
        rfecv_kwargs = {}
    else:
        rfecv_kwargs = rfecv_kwargs.copy()

    cv = rfecv_kwargs.get("cv")
    if not cv:
        if has_time:
            cv = TimeSeriesSplit(n_splits=rfecv_kwargs.get("cv_n_splits", 3))
            logger.info(f"Using TimeSeriesSplit for RFECV...")
        else:
            cv = None
        rfecv_kwargs["cv"] = cv

    if "cv_n_splits" in rfecv_kwargs:
        del rfecv_kwargs["cv_n_splits"]

    COMMON_RFECV_PARAMS = dict(
        early_stopping_rounds=early_stopping_rounds,
        cv=cv,
        cv_shuffle=not has_time,
    )
    COMMON_RFECV_PARAMS.update(rfecv_kwargs)

    # If ES is disabled (early_stopping_rounds=None), strip the key from every per-model
    # constructor-params dict so backends don't register an ES callback.
    # - LGB: omitted from constructor → LightGBMSklearn skips ES on fit
    # - XGB: omitted from constructor → no early_stopping_rounds passed
    # - CB:  omitted → CatBoost runs full iterations (no od_type)
    # - HGB: replace n_iter_no_change with iterations+1 so ES condition never trips
    if early_stopping_disabled:
        for _params in (CB_GENERAL_PARAMS, CB_REGR, CB_CLASSIF, CB_CALIB_CLASSIF,
                        LGB_GENERAL_PARAMS, XGB_GENERAL_PARAMS,
                        XGB_GENERAL_CLASSIF, XGB_CALIB_CLASSIF,
                        MLP_GENERAL_PARAMS, COMMON_RFECV_PARAMS):
            if _params is not None:  # MLP_GENERAL_PARAMS may be None when MLP not in scope
                _params.pop("early_stopping_rounds", None)
        # HGB uses early_stopping=True + n_iter_no_change; force ES off explicitly
        HGB_GENERAL_PARAMS["early_stopping"] = False
        HGB_GENERAL_PARAMS.pop("n_iter_no_change", None)

    return SimpleNamespace(
        integral_calibration_error=integral_calibration_error,
        final_integral_calibration_error=final_integral_calibration_error,
        lgbm_integral_calibration_error=lgbm_integral_calibration_error,
        fs_and_hpt_integral_calibration_error=fs_and_hpt_integral_calibration_error,
        CB_GENERAL_PARAMS=CB_GENERAL_PARAMS,
        CB_REGR=CB_REGR,
        CB_CLASSIF=CB_CLASSIF,
        CB_CALIB_CLASSIF=CB_CALIB_CLASSIF,
        HGB_GENERAL_PARAMS=HGB_GENERAL_PARAMS,
        LGB_GENERAL_PARAMS=LGB_GENERAL_PARAMS,
        XGB_GENERAL_PARAMS=XGB_GENERAL_PARAMS,
        XGB_GENERAL_CLASSIF=XGB_GENERAL_CLASSIF,
        XGB_CALIB_CLASSIF=XGB_CALIB_CLASSIF,
        COMMON_RFECV_PARAMS=COMMON_RFECV_PARAMS,
        MLP_GENERAL_PARAMS=MLP_GENERAL_PARAMS,
        NGB_GENERAL_PARAMS=NGB_GENERAL_PARAMS,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# CatBoost text-processing helper
# -----------------------------------------------------------------------------------------------------------------------------------------------------


# CB's default occurrence_lower_bound (CatBoost master, 2026). Words appearing
# in fewer rows than this are pruned from the TF-IDF dictionary. On small
# training sets the default is too aggressive and can either raise
# "Dictionary size is 0" or — in the C++ _train loop — HANG indefinitely
# while the empty dictionary stalls bag-of-words feature construction
# (fuzz c0056 / c0070, observed 2026-04-26).
CB_DEFAULT_OCCURRENCE_LOWER_BOUND = 50

# Above this row count we keep CB's defaults — production-sized data needs
# the original 50-occurrence floor to keep dictionaries bounded. Below it
# we scale the floor down proportionally so RFECV inner CV folds and
# small-n training runs do not collapse the dictionary.
CB_TEXT_PROC_DEFAULT_THRESHOLD_ROWS = 1000

# Fraction of training rows used for the occurrence floor when scaling.
# 5% means: a word appearing in 5%+ of the fold survives the prune, even
# with as few as 40 rows (where it becomes 2 — the absolute minimum that
# still excludes truly singleton terms).
CB_TEXT_PROC_OCCURRENCE_FRACTION = 0.05
CB_TEXT_PROC_OCCURRENCE_FLOOR = 2


def compute_cb_text_processing(n_train_rows: int) -> Optional[dict]:
    """Return a CatBoost ``text_processing`` config that scales
    ``occurrence_lower_bound`` to the training set size, or ``None`` if
    CB's defaults are appropriate.

    CatBoost's default ``occurrence_lower_bound=50`` rejects any token
    that appears in fewer than 50 rows. On small training sets (RFECV
    inner CV folds, single-pass training with ``n_train_rows`` < ~1000),
    this default rejects the entire vocabulary and either:
      * raises ``"Dictionary size is 0"`` — handled by the existing
        fallback at ``trainer._train_model_with_fallback``; or
      * hangs inside CB's C++ ``_train`` loop waiting for an empty
        dictionary to materialise (fuzz c0056 / c0070).

    The fix is a row-proportional floor: a word that appears in at least
    5 % of the rows survives, clamped to >= 2 (so a single-occurrence
    word never builds a dictionary entry — that would inflate the
    artefact and cause zero generalisation).

    Args:
        n_train_rows: Number of rows in the *fit-time* training set. For
            RFECV this is the inner-fold train size, NOT the outer suite
            input size.

    Returns:
        ``text_processing`` dict suitable for ``CatBoost.set_params(
        text_processing=...)`` / ``CatBoost.__init__(text_processing=...)``,
        or ``None`` when defaults are fine (``n_train_rows`` >=
        ``CB_TEXT_PROC_DEFAULT_THRESHOLD_ROWS``, or invalid input).
    """
    if not isinstance(n_train_rows, int) or n_train_rows <= 0:
        return None
    if n_train_rows >= CB_TEXT_PROC_DEFAULT_THRESHOLD_ROWS:
        return None
    olb = max(
        CB_TEXT_PROC_OCCURRENCE_FLOOR,
        int(round(n_train_rows * CB_TEXT_PROC_OCCURRENCE_FRACTION)),
    )
    return {
        "tokenizers": [{"tokenizer_id": "Space", "delimiter": " "}],
        "dictionaries": [
            {
                "dictionary_id": "Word",
                "occurrence_lower_bound": str(olb),
                "max_dictionary_size": "50000",
                "gram_order": "1",
            }
        ],
        "feature_processing": {
            "default": [
                {
                    "tokenizers_names": ["Space"],
                    "dictionaries_names": ["Word"],
                    "feature_calcers": ["BoW"],
                }
            ]
        },
    }


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Training Set Feature Statistics
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def get_trainset_features_stats(train_df: pd.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables.

    Numeric ranges are computed via a single ``df[num_cols].agg(['min','max'])``
    call rather than a per-column Python loop. On 1M rows x 60 numeric cols the
    vectorised path measures ~9x faster (single C-level reduction over a
    contiguous block versus N separate column reductions with attribute
    look-up overhead per iteration).
    """
    res = {}
    num_cols = get_numeric_columns(train_df)
    if num_cols:
        if len(num_cols) == train_df.shape[1]:
            res["min"] = train_df.min(axis=0)
            res["max"] = train_df.max(axis=0)
        else:
            # Vectorised aggregation: pandas reduces all numeric columns in a
            # single pass instead of issuing one .min() / .max() call per col.
            # Slicing once via train_df[num_cols] avoids the "Categorical is
            # not ordered for operation min" error that surfaced when the
            # whole frame was reduced (and that the previous per-col loop
            # worked around row-by-row).
            agg_df = train_df[num_cols].agg(["min", "max"])
            res["min"] = agg_df.loc["min"]
            res["max"] = agg_df.loc["max"]

    cat_cols = get_categorical_columns(train_df, include_string=False)
    if cat_cols:
        cat_vals = {}
        for col in tqdmu(cat_cols, desc="cat vars stats", leave=False):
            unique_vals = train_df[col].unique()
            if not max_ncats_to_track or (len(unique_vals) <= max_ncats_to_track):
                cat_vals[col] = unique_vals
        res["cat_vals"] = cat_vals
    return res


def get_trainset_features_stats_polars(train_df: pl.DataFrame, max_ncats_to_track: int = 1000) -> dict:
    """Computes ranges of numerical and categorical variables using Polars.

    Uses lazy mode and selectors for parallel computation.

    Args:
        train_df: Polars DataFrame
        max_ncats_to_track: Max unique values to track for categorical columns

    Returns:
        dict with "min", "max" (as pd.Series) and "cat_vals" (dict of arrays)
    """

    res = {}
    lf = train_df.lazy()

    # Compute numeric min/max and categorical n_unique in a single parallel select
    stats = lf.select(
        # Numeric: min and max
        cs.numeric().min().name.suffix("__min"),
        cs.numeric().max().name.suffix("__max"),
        # Categorical: n_unique to filter before getting unique values
        cs.by_dtype(pl.String, pl.Categorical).n_unique().name.suffix("__n_unique"),
    ).collect()

    # Extract numeric stats
    if len(stats.columns) > 0:
        mins = {}
        maxs = {}
        for col in stats.columns:
            if col.endswith("__min"):
                orig_col = col[:-5]
                mins[orig_col] = stats[col][0]
            elif col.endswith("__max"):
                orig_col = col[:-5]
                maxs[orig_col] = stats[col][0]

        if mins:
            res["min"] = pd.Series(mins)
        if maxs:
            res["max"] = pd.Series(maxs)

    # Extract categorical columns that are under the threshold
    cat_cols_to_fetch = []
    for col in stats.columns:
        if col.endswith("__n_unique"):
            orig_col = col[:-10]
            n_unique = stats[col][0]
            if not max_ncats_to_track or n_unique <= max_ncats_to_track:
                cat_cols_to_fetch.append(orig_col)

    # Get unique values for qualifying categorical columns. Batched into
    # ONE collect() via implode() so per-column unique-vectors arrive as
    # rows of one frame; saves ``len(cat_cols_to_fetch) - 1`` LazyFrame
    # materializations (each costing 5-10ms on a typical mid-size frame).
    # On a 100k×15-cat-cols frame this dropped 14 collects -> 1 collect
    # without changing semantics (implode collapses unique values into
    # a single list-typed cell per column; we then unpack via [0]).
    if cat_cols_to_fetch:
        unique_lists = lf.select([
            pl.col(c).unique().implode().alias(c) for c in cat_cols_to_fetch
        ]).collect()
        cat_vals = {
            col: unique_lists[col][0].to_numpy()
            for col in cat_cols_to_fetch
        }
        res["cat_vals"] = cat_vals

    return res


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Precomputed suite bundle (opt-in fast path for repeated-suite-on-same-train benchmarking)
# -----------------------------------------------------------------------------------------------------------------------------------------------------


@dataclass
class TrainMlframeSuitePrecomputed:
    """Bundle of pre-computed train-set artifacts that ``train_mlframe_models_suite`` would otherwise compute inline.

    Populated via the ``precompute_*`` helpers in this module. Pass to the suite as
    ``precomputed=TrainMlframeSuitePrecomputed(...)`` to skip the matching in-suite compute steps;
    each field is independently opt-in (None = compute inline as today).

    ``train_df_fingerprint`` is reserved for a future cross-process disk-cache layer so a bundle
    persisted from one run can be safely re-attached only when the train frame hasn't changed.
    """
    trainset_features_stats: Optional[dict] = None
    dummy_baselines: Optional[dict] = None
    composite_target_specs: Optional[dict] = None
    train_df_fingerprint: Optional[str] = None  # for cross-process disk-cache reuse later


def precompute_composite_target_specs(
    train_df=None,
    target_by_type: Optional[dict] = None,
    config: Optional[Any] = None,
) -> dict:
    """NOT IMPLEMENTED -- always raises.

    A faithful precompute would have to mirror ``run_composite_target_discovery``: composite_cache
    wiring, library version signatures, DiscoveryCache fingerprints. That surface is large and lives
    behind locked files; until the helper can reuse the same cache key path as the suite and stay
    byte-equal across runs, returning an empty dict here would silently disable discovery on the
    suite side -- worse than recomputing.

    Callers who already have a prior run's ``metadata["composite_target_specs"]`` saved to disk can
    still feed the suite directly via ``TrainMlframeSuitePrecomputed(composite_target_specs=...)``;
    the bundle's skip-when-supplied gate is content-truthy, not just non-None, so an empty dict will
    NOT disable the in-suite compute (see ``train_mlframe_models_suite`` for the gate).

    Raises:
        NotImplementedError: always. Use ``TrainMlframeSuitePrecomputed(composite_target_specs=<dict from prior run>)`` instead.
    """
    raise NotImplementedError(
        "precompute_composite_target_specs is not implemented. Load metadata['composite_target_specs'] from a prior run "
        "and pass it directly via TrainMlframeSuitePrecomputed(composite_target_specs=...).",
    )


def precompute_dummy_baselines(
    train_df,
    target_by_type: dict,
    config: Optional[Any] = None,
) -> dict:
    """NOT IMPLEMENTED -- always raises.

    The in-suite dummy-baseline compute lives in ``core/_phase_dummy_baselines.py`` and needs the
    post-split train/val/test frames plus per-target slices, which the caller does NOT have access to
    before the suite has run the split phase. A faithful precompute helper would have to either
    (a) replicate the suite's split logic here (duplication risk) or (b) accept the already-split
    frames + per-target targets as arguments (large signature). Both are deferred.

    Callers who already have a prior run's ``metadata["dummy_baselines"]`` saved to disk can feed
    the suite directly via ``TrainMlframeSuitePrecomputed(dummy_baselines=...)``; the bundle's
    skip-when-supplied gate is content-truthy, so an empty dict will NOT silently disable the
    per-target in-suite compute.

    Raises:
        NotImplementedError: always. Use ``TrainMlframeSuitePrecomputed(dummy_baselines=<dict from prior run>)`` instead.
    """
    raise NotImplementedError(
        "precompute_dummy_baselines is not implemented. Load metadata['dummy_baselines'] from a prior run and pass it "
        "directly via TrainMlframeSuitePrecomputed(dummy_baselines=...).",
    )


def precompute_trainset_features_stats(train_df, max_ncats_to_track: int = 1000) -> dict:
    """Compute the trainset_features_stats dict the suite would compute inline.

    Dispatches to the polars or pandas backend based on the input type so the output dict is
    byte-equal (same key order, same value shapes) to what ``train_mlframe_models_suite`` produces
    on the same frame. Use the returned dict as ``TrainMlframeSuitePrecomputed.trainset_features_stats``
    to skip the in-suite recompute on repeat runs.

    Args:
        train_df: Pandas or Polars DataFrame -- the same frame that will later be passed to the suite
            (post-split, post-pipeline-fit form). For pre-split callers, slice the train rows yourself
            first; the suite's stats step runs AFTER train/val/test split.
        max_ncats_to_track: forwarded to the underlying stats function.

    Returns:
        dict with at least ``min``, ``max`` (pd.Series) and ``cat_vals`` (dict[str, np.ndarray]) keys.
    """
    if isinstance(train_df, pl.DataFrame):
        return get_trainset_features_stats_polars(train_df, max_ncats_to_track=max_ncats_to_track)
    return get_trainset_features_stats(train_df, max_ncats_to_track=max_ncats_to_track)


def precompute_all(
    train_df,
    target_by_type: Optional[dict] = None,
    *,
    fs_config: Optional[Any] = None,
    dummy_baselines_config: Optional[Any] = None,
    composite_config: Optional[Any] = None,
) -> TrainMlframeSuitePrecomputed:
    """Fill the ``trainset_features_stats`` precompute slot; leave the rest at ``None``.

    Despite the name, this is NOT a one-shot helper for every slot: only ``trainset_features_stats``
    has a real precompute path. ``precompute_dummy_baselines`` and ``precompute_composite_target_specs``
    raise ``NotImplementedError`` -- the dummy helper needs post-split frames that aren't reachable
    pre-suite, and the composite helper would have to mirror the full discovery cache surface
    (deferred). The bundle slots themselves still work: callers who have a prior run's metadata
    saved to disk can build the bundle by hand:

        from mlframe.training.helpers import precompute_all, TrainMlframeSuitePrecomputed
        bundle = precompute_all(train_df, target_by_type)
        bundle.dummy_baselines = prior_run_metadata["dummy_baselines"]
        bundle.composite_target_specs = prior_run_metadata["composite_target_specs"]

    Args:
        train_df: Pandas or Polars train frame.
        target_by_type: per-target mapping (forwarded to dummy stub).
        fs_config: feature-stats kwargs container (currently only ``max_ncats_to_track`` is honored
            if present as an attribute; pass None for defaults).
        dummy_baselines_config: forwarded to the dummy stub.
        composite_config: forwarded to the composite stub.

    Returns:
        A populated ``TrainMlframeSuitePrecomputed`` bundle.
    """
    _max_ncats = 1000
    if fs_config is not None:
        _maybe = getattr(fs_config, "max_ncats_to_track", None)
        if isinstance(_maybe, int) and _maybe > 0:
            _max_ncats = _maybe
    stats = precompute_trainset_features_stats(train_df, max_ncats_to_track=_max_ncats)

    # The two helpers below currently return empty dicts; preserve the None sentinel on the bundle
    # so the suite's "if precomputed.X is not None" gate keeps recomputing inline rather than
    # silently skipping with no data.
    return TrainMlframeSuitePrecomputed(
        trainset_features_stats=stats,
        dummy_baselines=None,
        composite_target_specs=None,
        train_df_fingerprint=None,
    )


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Callback Classes for Training Monitoring
# -----------------------------------------------------------------------------------------------------------------------------------------------------

