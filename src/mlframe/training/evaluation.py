"""
Model evaluation and reporting functions for mlframe training.

This module contains functions for evaluating trained models and generating
performance reports for both regression and classification tasks.

Functions:
    evaluate_model: High-level model evaluation interface
    report_model_perf: Unified performance report for classifiers and regressors
    report_regression_model_perf: Detailed regression performance report
    report_probabilistic_model_perf: Detailed classification performance report
    get_model_feature_importances: Extract feature importances from a model
    plot_model_feature_importances: Plot feature importances
    post_calibrate_model: Post-calibrate a model using a meta-model
"""

from __future__ import annotations


import logging
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

DEFAULT_RANDOM_SEED = 42
DEFAULT_BINARY_THRESHOLD = 0.5
DEFAULT_PLOT_SAMPLE_SIZE = 500
# Default tightened from 4 to 2 d.p. With adaptive widening on sub-1 values (see ``_format.format_metric``), 2 keeps headers short for typical metrics like RMSE=11497.47 while still rendering small values like 0.0034 correctly.
DEFAULT_REPORT_NDIGITS = 2
DEFAULT_CALIB_REPORT_NDIGITS = 2
DEFAULT_NBINS = 10
DEFAULT_FIGSIZE = (15, 5)
# DEFAULT_FI_FIGSIZE moved to sibling _feature_importances.py alongside
# the FI block that's the sole consumer; re-exported below.

# Module-level cache for the plot-sample index. Hot report paths re-draw the same
# scatter repeatedly with identical (len(preds), seed) -- caching the choice avoids
# rebuilding the RNG and resampling each call.
#
# Bounded + thread-safe: pre-fix this was an unbounded dict with no lock. A long-
# running notebook reporting on many target sizes (per-target, per-fold) could grow
# the cache to 10k+ entries; at the typical sample_size=10_000 (80 KB per int64
# index), a 10k-entry cache leaks 800 MB RSS. Now: OrderedDict + LRU cap so
# bounded growth, threading.Lock so concurrent fold reporters don't race the write.
import threading as _threading
from collections import OrderedDict as _OrderedDict
_PLOT_IDX_CACHE: "OrderedDict[tuple, np.ndarray]" = _OrderedDict()
_PLOT_IDX_CACHE_LOCK = _threading.Lock()
_PLOT_IDX_CACHE_MAX = 256

# Suite-level override for the residual_audit block. Set by ``train_mlframe_models_suite`` from ``behavior_config.report_residual_audit``. None means "use default True" so direct callers of ``report_model_perf`` outside the suite keep the historical behaviour.
_RESIDUAL_AUDIT_OVERRIDE: "Optional[bool]" = None


def _set_residual_audit_enabled(enabled: Optional[bool]) -> None:
    """Suite-level setter used by ``train_mlframe_models_suite`` to propagate ``behavior_config.report_residual_audit`` down to ``report_model_perf`` without threading the param through every call site."""
    global _RESIDUAL_AUDIT_OVERRIDE
    _RESIDUAL_AUDIT_OVERRIDE = enabled


def _get_residual_audit_enabled() -> bool:
    """Resolve the residual-audit toggle. Honour suite override when set; default True for stand-alone calls."""
    return True if _RESIDUAL_AUDIT_OVERRIDE is None else bool(_RESIDUAL_AUDIT_OVERRIDE)


def _get_cached_plot_idx(n: int, sample_size: int, seed: int) -> "np.ndarray":
    key = (n, sample_size, seed)
    with _PLOT_IDX_CACHE_LOCK:
        cached = _PLOT_IDX_CACHE.get(key)
        if cached is not None:
            _PLOT_IDX_CACHE.move_to_end(key)  # LRU touch
            return cached
    # Build the index OUTSIDE the lock so two threads with different keys don't
    # serialise on the np.random.choice call. The lock is only re-acquired briefly
    # to store + evict.
    import numpy as _np
    _rng = _np.random.default_rng(seed)
    idx = _rng.choice(_np.arange(n), size=min(sample_size, n), replace=False)
    with _PLOT_IDX_CACHE_LOCK:
        _PLOT_IDX_CACHE[key] = idx
        while len(_PLOT_IDX_CACHE) > _PLOT_IDX_CACHE_MAX:
            _PLOT_IDX_CACHE.popitem(last=False)
    return idx

import polars as pl
import matplotlib.pyplot as plt

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# ``root_mean_squared_error`` is re-exported here because external test modules
# import it from ``mlframe.training.evaluation``; under sklearn < 1.4 fall back
# to a thin wrapper around ``mean_squared_error``. The other sklearn / fast-*
# metric imports the module used to carry were dead and were removed.
try:
    from sklearn.metrics import root_mean_squared_error  # noqa: F401
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
        output_errors = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values"))
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return output_errors
            elif multioutput == "uniform_average":
                multioutput = None
        return np.average(output_errors, weights=multioutput)


from mlframe.feature_selection.importance import plot_feature_importance
from ._reporting import (  # noqa: E402,F401
    _canonical_multilabel_y,
    report_model_perf,
    report_regression_model_perf,
    report_probabilistic_model_perf,
)
# Feature-importance helpers (~550 lines: 5 internal extractors + 2 public
# entry points) carved into sibling ``_feature_importances.py`` to drop
# this file below the 1k-LOC monolith threshold. Re-exported here so
# existing ``from mlframe.training.evaluation import
# plot_model_feature_importances`` callers keep working unchanged.
from ._feature_importances import (  # noqa: E402,F401
    DEFAULT_FI_FIGSIZE,
    _captum_integrated_gradients_importance,
    _cuda_batched_permutation_importance,
    _first_layer_weight_importance,
    _permutation_feature_importances,
    _torch_module_from_model,
    _unwrap_estimator_chain,
    get_model_feature_importances,
    plot_model_feature_importances,
)

logger = logging.getLogger(__name__)



def post_calibrate_model(
    original_model: Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Sequence[str], Any, Dict],
    target_series: pd.Series,
    target_label_encoder: Optional[LabelEncoder],
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    configs: Any,
    calib_set_size: int = 2000,
    nbins: int = DEFAULT_NBINS,
    show_val: bool = False,
    meta_model: Optional[Any] = None,
    calib_idx: Optional[np.ndarray] = None,
    calib_probs: Optional[np.ndarray] = None,
    calib_target: Optional[np.ndarray] = None,
    **fit_params: Any,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Sequence[str], Any, Dict]:
    """
    Post-calibrate a trained model using a meta-model.

    Trains a meta-model (CatBoost by default) on the original model's probability outputs to improve calibration.
    The calibrator must NEVER see test rows (doing so re-prices the test slice as a tuning surface and inflates the
    reported test metric). Calibration sources, in precedence:

    1. ``(calib_probs, calib_target)`` supplied directly -- typical for OOF-train probs produced by the trainer's
       ``cross_val_predict`` step. No row leakage by construction.
    2. ``calib_idx`` -- a reserved slice from train (opt-in via ``TrainingSplitConfig.calib_size``); must be disjoint
       from ``test_idx``. We assert the intersection is empty before any ``meta_model.fit(...)`` call.

    Parameters
    ----------
    original_model : tuple
        8-element tuple containing:
        (model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline, metrics)
    target_series : pd.Series
        Target values indexed to match val_idx and test_idx.
    target_label_encoder : LabelEncoder or None
        Encoder for target labels.
    val_idx : np.ndarray
        Indices for validation set.
    test_idx : np.ndarray
        Indices for test set.
    configs : object
        Configuration object with `integral_calibration_error` attribute.
    calib_set_size : int, default=2000
        Legacy alias kept for backward compat with callers that still pass it; only honoured for the multi-output
        per-class isotonic path when ``calib_probs`` is not supplied. Binary path no longer slices test rows.
    nbins : int, default=10
        Number of bins for calibration analysis.
    show_val : bool, default=False
        Whether to display validation set results.
    meta_model : Any, optional
        Custom meta-model. If None, uses CatBoostClassifier with GPU.
    calib_idx : np.ndarray, optional
        Indices of the reserved calibration slice (typically reserved from the train split via
        ``TrainingSplitConfig.calib_size``). Must be disjoint from ``test_idx``; intersection -> ValueError.
    calib_probs : np.ndarray, optional
        Probability vector (OOF-train preferred) to fit the meta-calibrator on. Shape (M, 2) for binary.
    calib_target : np.ndarray, optional
        Target vector aligned with ``calib_probs``. Required when ``calib_probs`` is supplied.
    **fit_params
        Additional parameters passed to meta_model.fit().

    Returns
    -------
    tuple
        Same 8-element structure as input, but with calibrated probabilities:
        (model, test_preds, meta_test_probs, val_preds, meta_val_probs, columns, pre_pipeline, metrics)
    """
    from catboost import CatBoostClassifier
    from mlframe.metrics.core import ICE

    if meta_model is None:
        meta_model = CatBoostClassifier(
            iterations=3000,
            verbose=False,
            has_time=False,
            learning_rate=0.2,
            eval_fraction=0.1,
            task_type="GPU",
            early_stopping_rounds=400,
            eval_metric=ICE(metric=configs.integral_calibration_error, higher_is_better=False),
            custom_metric="AUC",
        )

    # Validate original_model structure
    if not isinstance(original_model, (tuple, list)) or len(original_model) != 8:
        raise ValueError(
            f"original_model must be an 8-element tuple/list containing "
            f"(model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline, metrics), "
            f"got {type(original_model).__name__} with {len(original_model) if hasattr(original_model, '__len__') else 'unknown'} elements"
        )
    model, test_preds, test_probs, val_preds, val_probs, columns, pre_pipeline, metrics = original_model

    # Test-row leak guard: any path into ``meta_model.fit(...)`` that includes a row from ``test_idx`` re-prices the
    # test slice as a calibration tuning surface, inflates the reported test metric, and silently invalidates the
    # holdout estimate. Compute the intersection between the calibration index (if user provided one) and the test
    # index once up front; raise before any fit call ever runs.
    if calib_idx is not None and test_idx is not None:
        _calib_arr = np.asarray(calib_idx).ravel()
        _test_arr = np.asarray(test_idx).ravel()
        _overlap = np.intersect1d(_calib_arr, _test_arr, assume_unique=False)
        if _overlap.size > 0:
            raise ValueError(
                f"calibration must not touch test_idx rows; got {int(_overlap.size)} test rows in calibrator input"
            )

    # When the user supplied a direct (calib_probs, calib_target) pair, validate it doesn't degenerate to test rows
    # via a length collision -- a common foot-gun is passing ``test_probs[:k]`` as ``calib_probs`` and assuming the
    # function will route it correctly. We can't compare row IDs without indices, so the contract is: callers either
    # pass an explicit ``calib_idx`` (and we verify disjointness above) OR pass ``calib_probs`` that they have
    # constructed independently of the test slice (OOF-train probs are the canonical source).
    if calib_probs is not None and calib_target is None:
        raise ValueError("post_calibrate_model: calib_probs supplied without calib_target; both required.")

    # Multi-output path (MULTICLASS / MULTILABEL).
    # When probs are (N, K) with K != 2, route through per-class isotonic
    # calibration (K independent IsotonicRegression fits) instead of the
    # univariate meta-model path. The multi-output branch returns the same
    # 8-element tuple shape as the binary branch but with calibrated
    # probability matrices.
    is_multi_output = (
        hasattr(test_probs, "shape")
        and len(test_probs.shape) == 2
        and test_probs.shape[1] != 2
    )
    if is_multi_output:
        from mlframe.training.trainer import (
            _PerClassIsotonicCalibrator, _PostHocMultiCalibratedModel,
        )
        from mlframe.training.configs import TargetTypes
        # Infer target_type from labels: if y_true is (N, K) indicator
        # matrix -> multilabel; otherwise multiclass.
        y_test_full = target_series.iloc[test_idx].values
        if y_test_full.ndim == 2 or (
            hasattr(y_test_full, "dtype") and y_test_full.dtype == object
        ):
            _target_type = TargetTypes.MULTILABEL_CLASSIFICATION
        else:
            _target_type = TargetTypes.MULTICLASS_CLASSIFICATION
        # Fit per-class isotonic on the calibration source. Prefer caller-provided (calib_probs, calib_target);
        # fall back to OOF-train probs stamped on the model; only as last resort -- and only with an explicit ``calib_idx``
        # confirmed disjoint from test_idx above -- do we draw from train_idx via target_series. Pure test-slice
        # calibration (the historical default ``test_probs[:calib_set_size]``) is no longer supported here: it leaks.
        if calib_probs is not None:
            _calib_p = np.asarray(calib_probs)
            _calib_y = np.asarray(calib_target)
        else:
            _oof_probs_mo = getattr(model, "oof_probs", None)
            if _oof_probs_mo is not None:
                _calib_p = np.asarray(_oof_probs_mo)
                # oof_probs are in train-row order (cross_val_predict); pair each
                # with its OWN row's label via the train-aligned oof_target. The
                # old ``target_series.iloc[:len(oof)]`` positional slice is only
                # correct when train is the leading contiguous block, so under a
                # shuffled / group-aware split it fit the calibrator on
                # mismatched (prob, label) pairs.
                _oof_y_mo = getattr(model, "oof_target", None)
                if _oof_y_mo is None:
                    raise ValueError(
                        "post_calibrate_model (multi-output): model.oof_probs is present but "
                        "model.oof_target is missing, so OOF probs cannot be aligned to their "
                        "labels. Retrain so oof_target is stamped, or pass calib_probs+calib_target."
                    )
                _calib_y = np.asarray(_oof_y_mo)[: _calib_p.shape[0]]
            else:
                raise ValueError(
                    "post_calibrate_model (multi-output): no calibration source available. Pass calib_probs+calib_target "
                    "(OOF-train probs preferred) or train the model with oof_n_splits>=2 so model.oof_probs is stamped."
                )
        calibrator = _PerClassIsotonicCalibrator.fit(
            _calib_p, _calib_y, _target_type,
        )
        # Produce calibrated val/test probs.
        meta_val_probs = calibrator.predict_proba(val_probs)
        meta_test_probs = calibrator.predict_proba(test_probs)
        # Wrap model for transparent predict_proba delegation at
        # downstream serving time.
        wrapped_model = _PostHocMultiCalibratedModel(
            model, calibrator, _target_type,
            classes_=getattr(model, "classes_", None),
        )
        # Stamp calibrated probs onto the underlying model so downstream consumers (ensembling blend via
        # use_ap12_calibrated_probs, honest diagnostics, dashboards) can consume the AP12-calibrated surface
        # without re-running the calibrator. RRF is rank-based and intentionally bypasses this on the read side.
        try:
            setattr(model, "calibrated_val_probs", meta_val_probs)
            setattr(model, "calibrated_test_probs", meta_test_probs)
            setattr(wrapped_model, "calibrated_val_probs", meta_val_probs)
            setattr(wrapped_model, "calibrated_test_probs", meta_test_probs)
        except (AttributeError, TypeError):
            # Slot-only / read-only model objects (rare) just skip the stamp; the ensembling
            # read-side falls back to raw .<split>_probs without crashing.
            pass
        # Copy val_preds/test_preds forward -- they're caller-provided and
        # decoupled from the probability calibration.
        return (
            wrapped_model, test_preds, meta_test_probs,
            val_preds, meta_val_probs, columns, pre_pipeline, metrics,
        )

    # Binary path. Source the calibration (probs, y) pair from caller-provided OOF inputs OR model.oof_probs;
    # NEVER from test_probs / test_idx. Historical default ``test_probs[:calib_set_size]`` is removed: it leaks the
    # test slice into the calibrator fit and inflates the reported test metric. See the test-row leak guard above.
    if calib_probs is not None:
        _binary_calib_probs = np.asarray(calib_probs)
        # Accept either (M, 2) or (M,) for the positive-class column.
        if _binary_calib_probs.ndim == 2 and _binary_calib_probs.shape[1] >= 2:
            _binary_fit_X = _binary_calib_probs[:, 1].reshape(-1, 1)
        else:
            _binary_fit_X = _binary_calib_probs.reshape(-1, 1)
        _binary_fit_y = np.asarray(calib_target).ravel()
    else:
        _oof_probs_attr = getattr(model, "oof_probs", None)
        if _oof_probs_attr is None:
            raise ValueError(
                "post_calibrate_model: no calibration source available. Pass calib_probs+calib_target (OOF-train probs "
                "preferred) or train the model with oof_n_splits>=2 so model.oof_probs is stamped on the model object."
            )
        _oof_arr = np.asarray(_oof_probs_attr)
        if _oof_arr.ndim == 2 and _oof_arr.shape[1] >= 2:
            _binary_fit_X = _oof_arr[:, 1].reshape(-1, 1)
        else:
            _binary_fit_X = _oof_arr.reshape(-1, 1)
        # OOF y must be the train-aligned target stamped alongside oof_probs
        # (cross_val_predict order), NOT a positional ``target_series[:len(oof)]``
        # slice -- the latter is correct only when train is the leading
        # contiguous block, so under a shuffled / group-aware split it paired
        # OOF probs with unrelated rows' labels and learned a scrambled mapping.
        _oof_y_bin = getattr(model, "oof_target", None)
        if _oof_y_bin is None:
            raise ValueError(
                "post_calibrate_model: model.oof_probs is present but model.oof_target is "
                "missing, so OOF probs cannot be aligned to their labels. Retrain so "
                "oof_target is stamped, or pass calib_probs+calib_target."
            )
        _binary_fit_y = np.asarray(_oof_y_bin)[: _binary_fit_X.shape[0]].ravel()

    # Final leak assertion at the fit boundary: when an explicit calib_idx was passed, the disjointness check above
    # already cleared this. When (calib_probs, calib_target) were passed instead, the caller asserted independence at
    # source. Here we double-check the shape contract before letting any data hit ``meta_model.fit``.
    if _binary_fit_X.shape[0] != _binary_fit_y.shape[0]:
        raise ValueError(
            f"calibration X / y row counts diverge: X.shape[0]={_binary_fit_X.shape[0]} vs y.shape[0]={_binary_fit_y.shape[0]}"
        )

    meta_model.fit(_binary_fit_X, _binary_fit_y, **fit_params)

    try:
        from mlframe.training.provenance import record_provenance as _record_provenance
        _n_calib_fit = int(_binary_fit_X.shape[0]) if hasattr(_binary_fit_X, "shape") else None
        _record_provenance(
            metrics if isinstance(metrics, dict) else None,
            "post_calibrate",
            source="oof" if calib_probs is not None else "calib",
            n_rows=_n_calib_fit,
        )
    except Exception:
        pass

    # CalibrationConfig.policy_auto_pick: run pick_best_calibrator on the OOF source so the metrics dict carries the auto-pick verdict (chosen method + ECE CI) alongside the legacy meta-model output. Enabled by default; opt-out via configs.calibration.policy_auto_pick = False.
    _calib_cfg = getattr(configs, "calibration", None) if configs is not None else None
    _policy_on = True if _calib_cfg is None else bool(getattr(_calib_cfg, "policy_auto_pick", True))
    if _policy_on and isinstance(metrics, dict):
        try:
            from mlframe.calibration.policy import pick_best_calibrator
            _emit_plot = bool(getattr(_calib_cfg, "emit_plot", False)) if _calib_cfg is not None else False
            _plot_path = getattr(_calib_cfg, "plot_path", None) if _calib_cfg is not None else None
            _n_boot = int(getattr(_calib_cfg, "n_bootstrap", 1000)) if _calib_cfg is not None else 1000
            _alpha = float(getattr(_calib_cfg, "alpha", 0.05)) if _calib_cfg is not None else 0.05
            _cand = getattr(_calib_cfg, "candidates", None) if _calib_cfg is not None else None
            _policy = pick_best_calibrator(
                probs=None,
                y=None,
                oof_probs=_binary_fit_X.ravel(),
                oof_y=_binary_fit_y,
                alpha=_alpha,
                candidates=_cand,
                n_bootstrap=_n_boot,
                emit_plot=_emit_plot,
                plot_path=_plot_path,
            )
            metrics["calibration_policy"] = {k: v for k, v in _policy.items() if k != "calibrated_probs"}
        except Exception as _policy_err:
            logger.warning("post_calibrate_model: calibration policy auto-pick failed: %s", _policy_err)

    # Always materialise calibrated val probs so the return tuple is consistent regardless of show_val. Without this
    # the pre-existing ``return ..., meta_val_probs, ...`` raises NameError whenever the caller leaves show_val=False
    # (the historical default) -- a latent bug surfaced once we tightened the test-leak guard above.
    meta_val_probs = meta_model.predict_proba(val_probs[:, 1].reshape(-1, 1)) if val_probs is not None else None

    if show_val:
        _ = report_model_perf(
            targets=target_series.iloc[val_idx],
            columns=columns,
            df=None,
            model_name="VAL",
            model=None,
            target_label_encoder=target_label_encoder,
            preds=val_preds,
            probs=val_probs,
            report_title="",
            nbins=nbins,
            print_report=False,
            show_fi=False,
            custom_ice_metric=configs.integral_calibration_error,
        )
        _ = report_model_perf(
            targets=target_series.iloc[val_idx],
            columns=columns,
            df=None,
            model_name="VAL fixed",
            model=None,
            target_label_encoder=target_label_encoder,
            preds=val_preds,
            probs=meta_val_probs,
            report_title="",
            nbins=nbins,
            print_report=False,
            show_fi=False,
            custom_ice_metric=configs.integral_calibration_error,
        )

    meta_test_probs = meta_model.predict_proba(test_probs[:, 1].reshape(-1, 1))

    _ = report_model_perf(
        targets=target_series.iloc[test_idx],
        columns=columns,
        df=None,
        model_name="TEST original",
        model=None,
        target_label_encoder=target_label_encoder,
        preds=test_preds,
        probs=test_probs,
        report_title="",
        nbins=nbins,
        print_report=False,
        show_fi=False,
        custom_ice_metric=configs.integral_calibration_error,
    )

    # Stamp calibrated probs onto the underlying model so downstream consumers (ensembling blend via
    # use_ap12_calibrated_probs, honest diagnostics, dashboards) can read the AP12-calibrated surface. The
    # ensembling read-side falls back to raw probs when the attribute is absent; RRF is rank-based and is
    # intentionally bypassed on the consume side (scale-invariant).
    try:
        setattr(model, "calibrated_val_probs", meta_val_probs)
        setattr(model, "calibrated_test_probs", meta_test_probs)
    except (AttributeError, TypeError):
        pass

    # The full test slice gets a single "TEST fixed" report -- the historical "fixed [calib_set_size:]" / "fixed lucky
    # [:calib_set_size]" two-bucket reporting reflected the old behaviour where the calibrator was fit on the first
    # ``calib_set_size`` test rows and evaluated on the remainder; now the calibrator never touches test, so every
    # test row is an honest holdout and the buckets collapse.
    _ = report_model_perf(
        targets=target_series.iloc[test_idx],
        columns=columns,
        df=None,
        model_name="TEST fixed",
        model=None,
        target_label_encoder=target_label_encoder,
        preds=(meta_test_probs[:, 1] > DEFAULT_BINARY_THRESHOLD).astype(int),
        probs=meta_test_probs,
        report_title="",
        nbins=nbins,
        print_report=True,
        show_fi=False,
        custom_ice_metric=configs.integral_calibration_error,
    )

    return model, test_preds, meta_test_probs, val_preds, meta_val_probs, columns, pre_pipeline, metrics


def evaluate_model(
    model: Union[ClassifierMixin, RegressorMixin],
    model_name: str,
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    show_fi: bool = True,
    verbose: int = 1,
    **kwargs,
) -> tuple:
    """
    Evaluate a trained model and generate reports.

    Args:
        model: Trained model
        model_name: Name for reporting
        targets: True target values
        columns: Feature column names
        preds: Predictions (optional, will be generated if not provided)
        probs: Probabilities for classification (optional)
        df: DataFrame with features (optional)
        show_fi: Whether to show feature importances
        verbose: Verbosity level
        **kwargs: Additional arguments passed to report functions

    Returns:
        Tuple of (preds, probs) or (preds, None) for regression
    """
    return report_model_perf(
        targets=targets,
        columns=columns,
        model_name=model_name,
        model=model,
        preds=preds,
        probs=probs,
        df=df,
        show_fi=show_fi,
        **kwargs,
    )


__all__ = [
    "evaluate_model",
    "report_model_perf",
    "report_regression_model_perf",
    "report_probabilistic_model_perf",
    "get_model_feature_importances",
    "plot_model_feature_importances",
    "post_calibrate_model",
    "compute_ml_perf_by_time",
    "visualize_ml_metric_by_time",
]


# =============================================================================
# Salvaged from training_old.py (compute_ml_perf, visualize_ml_metric_by_time)
# =============================================================================


def _compute_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Thin dispatcher used by compute_ml_perf_by_time.

    Accepts ``y_pred`` as either a 1-D vector of positive-class probabilities
    OR a 2-D ``(N, 2)`` probability matrix from a binary classifier; in the
    latter case the positive-class column is sliced out before metric call.
    Without this guard, callers passing 2-D probs hit sklearn's
    ``ValueError: bad input shape`` deep inside the metric.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_squared_error

    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        y_pred = y_pred[:, 1]

    if metric == "roc_auc":
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_pred))
    if metric == "average_precision":
        return float(average_precision_score(y_true, y_pred))
    if metric == "brier":
        return float(brier_score_loss(y_true, y_pred))
    if metric == "mse":
        return float(mean_squared_error(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {metric}")


def _normalize_pandas_offset_alias(freq: str) -> str:
    """Map legacy single-letter pandas offset aliases ("M", "Q", "Y", "A") to
    their pandas-2.2+ end-of-period equivalents ("ME", "QE", "YE", "YE") so
    callers using the historical aliases don't emit FutureWarning.

    Pandas 2.2 deprecated bare "M"/"Q"/"Y"/"A" in favour of explicit
    "ME"/"QE"/"YE" (month-end / quarter-end / year-end). This shim is a
    forwards-compatible no-op for already-correct strings and for non-affected
    aliases ("D", "H", "h", "W", weekday-anchored frequencies).
    """
    _ALIAS_MAP = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    if not isinstance(freq, str):
        return freq
    return _ALIAS_MAP.get(freq, freq)


def compute_ml_perf_by_time(
    y_true,
    y_pred,
    timestamps,
    freq: str = "D",
    metric: str = "roc_auc",
    min_samples: int = 100,
) -> pd.DataFrame:
    """Bin predictions by time frequency and compute a metric per bin.

    Salvaged shape of training_old.compute_ml_perf, adapted to a clean
    y_true/y_pred/timestamps interface. Returns a DataFrame indexed by time
    bucket with columns [metric, n_samples].
    """
    # Use the audit timestamp coercer so int64 epoch-seconds (~1.7e9) read as 2023-11
    # not 1970-01-01T00:00:01.7 (the default pd.to_datetime(int64) = ns interpretation).
    # Without this the pd.Grouper(freq=_freq) collapses every row into a single bucket
    # spanning the entire data, silently turning a *time-resolved* metric report into a
    # *single-bucket* report -- caller sees one row with the global metric value and
    # cannot tell from the output that the time axis collapsed.
    from .target_temporal_audit import coerce_timestamps_for_audit as _coerce_ts
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred, dtype=float),
            "ts": _coerce_ts(np.asarray(timestamps)),
        }
    )
    df = df.set_index("ts").sort_index()
    rows = []
    _freq = _normalize_pandas_offset_alias(freq)
    for bin_start, chunk in df.groupby(pd.Grouper(freq=_freq)):
        n = len(chunk)
        if n == 0:
            continue
        if n < min_samples:
            val = float("nan")
        else:
            try:
                val = _compute_metric(metric, chunk["y_true"].values, chunk["y_pred"].values)
            except (ValueError, TypeError, ZeroDivisionError, FloatingPointError) as exc:
                # Per-bin metric can fail on degenerate inputs (single-class
                # y_true, all-NaN y_pred); record NaN and continue with the
                # remaining bins. Programming bugs / Memory still propagate.
                logger.warning("metric %s failed on bin %s: %s", metric, bin_start, exc)
                val = float("nan")
        rows.append({"bin": bin_start, metric: val, "n_samples": n})
    out = pd.DataFrame(rows).set_index("bin") if rows else pd.DataFrame(columns=[metric, "n_samples"])
    return out


def visualize_ml_metric_by_time(
    perf_df: pd.DataFrame,
    ax=None,
    good_metric_threshold: Optional[float] = None,
    higher_is_better: bool = True,
    good_color: str = "green",
    bad_color: str = "red",
):
    """Line-plot a perf DataFrame produced by compute_ml_perf_by_time.

    Threshold-aware color banding: bars below/above good_metric_threshold get
    coloured by goodness. Returns a matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    metric_cols = [c for c in perf_df.columns if c != "n_samples"]
    if not metric_cols:
        return fig
    metric = metric_cols[0]
    values = perf_df[metric].values
    xs = np.arange(len(perf_df))
    ax.plot(xs, values, marker="o", color="steelblue", label=metric)
    if good_metric_threshold is not None:
        for x, v in zip(xs, values):
            if np.isnan(v):
                continue
            if higher_is_better:
                color = good_color if v >= good_metric_threshold else bad_color
            else:
                color = good_color if v <= good_metric_threshold else bad_color
            ax.axvspan(x - 0.4, x + 0.4, color=color, alpha=0.08)
    try:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in perf_df.index], rotation=45)
    except (ValueError, TypeError, AttributeError):
        # matplotlib raises ValueError on bad tick counts and AttributeError
        # when a non-Axes was passed; either way, skip xticks and let the
        # caller see the default labels.
        pass
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by time bin")
    ax.legend(loc="best")
    return fig
