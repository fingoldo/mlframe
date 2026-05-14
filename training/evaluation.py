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

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

DEFAULT_RANDOM_SEED = 42
DEFAULT_BINARY_THRESHOLD = 0.5
DEFAULT_PLOT_SAMPLE_SIZE = 500
# 2026-05-11 (user request): default tightened from 4 to 2 d.p. With adaptive widening on sub-1 values (see ``_format.format_metric``), 2 keeps headers short for typical metrics like RMSE=11497.47 while still rendering small values like 0.0034 correctly.
DEFAULT_REPORT_NDIGITS = 2
DEFAULT_CALIB_REPORT_NDIGITS = 2
DEFAULT_NBINS = 10
DEFAULT_FIGSIZE = (15, 5)
# 2026-05-13 (user request): FI plot figsize is half the perf-chart figsize.
# Pre-fix it was 15x5, which still dominated the suite report.
DEFAULT_FI_FIGSIZE = (7.5, 2.5)

# Module-level cache for the plot-sample index. Hot report paths re-draw the same
# scatter repeatedly with identical (len(preds), seed) -- caching the choice avoids
# rebuilding the RNG and resampling each call.
_PLOT_IDX_CACHE: "dict[tuple, np.ndarray]" = {}

# 2026-05-11 (user request): suite-level override for the residual_audit block. Set by ``train_mlframe_models_suite`` from ``behavior_config.report_residual_audit``. None means "use default True" so direct callers of ``report_model_perf`` outside the suite keep the historical behaviour.
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
    cached = _PLOT_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    import numpy as _np
    _rng = _np.random.default_rng(seed)
    idx = _rng.choice(_np.arange(n), size=min(sample_size, n), replace=False)
    _PLOT_IDX_CACHE[key] = idx
    return idx

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    max_error,
    r2_score,
    classification_report,
)

# 2026-05-09: numba-accelerated drop-ins for the regression metrics.
# Used on 1-D float arrays (the common case in report_regression_model_perf);
# 6-23x faster than sklearn at N=1M (sklearn's input validation overhead
# dominates the tiny reductions). Multioutput / sample_weight paths still
# use sklearn (the fast helpers don't cover those).
from mlframe.metrics import (
    fast_mean_absolute_error,
    fast_max_error,
    fast_r2_score,
    fast_root_mean_squared_error,
)

try:
    from sklearn.metrics import root_mean_squared_error
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


from pyutilz.pythonlib import get_human_readable_set_size
from IPython.display import display

from mlframe.metrics import fast_calibration_report, fast_roc_auc, compute_fairness_metrics
from mlframe.feature_importance import plot_feature_importance
from mlframe.training.phases import phase
from ._reporting import (  # noqa: E402,F401
    _canonical_multilabel_y,
    report_model_perf,
    report_regression_model_perf,
    report_probabilistic_model_perf,
)

logger = logging.getLogger(__name__)


def get_model_feature_importances(
    model: Any,
    columns: Sequence[str],
    return_df: bool = False,
) -> Optional[Union[np.ndarray, pd.DataFrame]]:
    """
    Extract feature importances from a trained model.

    Supports models with `feature_importances_` attribute (tree-based models)
    or `coef_` attribute (linear models). For Pipeline objects, extracts
    importances from the final estimator.

    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ or coef_ attribute.
    columns : Sequence[str]
        Feature column names.
    return_df : bool, default=False
        If True, return a DataFrame with feature names and importances.

    Returns
    -------
    np.ndarray, pd.DataFrame, or None
        Feature importances array, DataFrame (if return_df=True), or None
        if the model doesn't have importances.
    """
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        if model.coef_.ndim == 1:
            feature_importances = model.coef_
        else:
            feature_importances = model.coef_[-1, :]
    else:
        feature_importances = None

    if feature_importances is not None:
        if return_df:
            feature_importances = pd.DataFrame({"feature": columns, "importance": feature_importances})

    return feature_importances


def plot_model_feature_importances(
    model: Any,
    columns: Sequence[str],
    model_name: Optional[str] = None,
    num_factors: int = 10,
    figsize: Tuple[int, int] = DEFAULT_FI_FIGSIZE,
    positive_fi_only: bool = False,
    show_plots: bool = True,
    plot_file: str = "",
    max_zero_fi_to_plot: int = 4,
) -> Optional[np.ndarray]:
    """
    Plot feature importances for a trained model.

    Extracts and visualizes feature importances as a bar chart.

    Parameters
    ----------
    model : Any
        Trained model with extractable feature importances.
    columns : Sequence[str]
        Feature column names.
    model_name : str, optional
        Title for the plot.
    num_factors : int, default=10
        Maximum number of features to display. Reduced from 40 to 10
        (2026-05-12) so plots stay scannable on common feature counts;
        override via ``reporting_config.fi_top_n`` in
        ``train_mlframe_models_suite``.
    figsize : tuple, default=(15, 10)
        Figure size for the plot.
    positive_fi_only : bool, default=False
        If True, only show features with positive importance.
    plot_file : str, default=""
        Path for saving the plot.

    Returns
    -------
    np.ndarray or None
        Feature importances array, or None if extraction failed.
    """
    feature_importances = get_model_feature_importances(model=model, columns=columns)

    if feature_importances is not None:
        try:
            plot_feature_importance(
                feature_importances=feature_importances,
                columns=columns,
                kind=model_name,
                figsize=figsize,
                plot_file=plot_file,
                positive_fi_only=positive_fi_only,
                n=num_factors,
                show_plots=show_plots,
                max_zero_fi_to_plot=max_zero_fi_to_plot,
            )
        except (ValueError, AttributeError, IndexError, TypeError) as e:
            logger.warning(f"Could not plot feature importances: {e}. Maybe data shape is changed within a pipeline?")

        return feature_importances


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
    **fit_params: Any,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Sequence[str], Any, Dict]:
    """
    Post-calibrate a trained model using a meta-model.

    Trains a meta-model (CatBoost by default) on the original model's
    probability outputs to improve calibration. Uses a portion of the
    test set for calibration training.

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
        Number of samples from test set to use for meta-model training.
    nbins : int, default=10
        Number of bins for calibration analysis.
    show_val : bool, default=False
        Whether to display validation set results.
    meta_model : Any, optional
        Custom meta-model. If None, uses CatBoostClassifier with GPU.
    **fit_params
        Additional parameters passed to meta_model.fit().

    Returns
    -------
    tuple
        Same 8-element structure as input, but with calibrated probabilities:
        (model, test_preds, meta_test_probs, val_preds, meta_val_probs, columns, pre_pipeline, metrics)
    """
    from catboost import CatBoostClassifier
    from mlframe.metrics import ICE

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

    # 2026-04-24 Session 4: multi-output path (MULTICLASS / MULTILABEL).
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
        # Fit per-class isotonic on the calibration slice of test_probs.
        calib_probs = test_probs[:calib_set_size]
        calib_y = y_test_full[:calib_set_size]
        calibrator = _PerClassIsotonicCalibrator.fit(
            calib_probs, calib_y, _target_type,
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
        # Copy val_preds/test_preds forward -- they're caller-provided and
        # decoupled from the probability calibration.
        return (
            wrapped_model, test_preds, meta_test_probs,
            val_preds, meta_val_probs, columns, pre_pipeline, metrics,
        )

    meta_model.fit(test_probs[:calib_set_size, 1].reshape(-1, 1), target_series.iloc[test_idx].values[:calib_set_size], **fit_params)

    if show_val:
        meta_val_probs = meta_model.predict_proba(val_probs[:, 1].reshape(-1, 1))
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

    _ = report_model_perf(
        targets=target_series.iloc[test_idx].values[calib_set_size:],
        columns=columns,
        df=None,
        model_name="TEST fixed ",
        model=None,
        target_label_encoder=target_label_encoder,
        preds=(meta_test_probs[calib_set_size:, 1] > DEFAULT_BINARY_THRESHOLD).astype(int),
        probs=meta_test_probs[calib_set_size:, :],
        report_title="",
        nbins=nbins,
        print_report=True,
        show_fi=False,
        custom_ice_metric=configs.integral_calibration_error,
    )

    _ = report_model_perf(
        targets=target_series.iloc[test_idx].values[:calib_set_size],
        columns=columns,
        df=None,
        model_name="TEST fixed lucky",
        model=None,
        target_label_encoder=target_label_encoder,
        preds=(meta_test_probs[:calib_set_size, 1] > DEFAULT_BINARY_THRESHOLD).astype(int),
        probs=meta_test_probs[:calib_set_size, :],
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
    """Thin dispatcher used by compute_ml_perf_by_time."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_squared_error

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
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred, dtype=float),
            "ts": pd.to_datetime(pd.Series(timestamps).values),
        }
    )
    df = df.set_index("ts").sort_index()
    rows = []
    for bin_start, chunk in df.groupby(pd.Grouper(freq=freq)):
        n = len(chunk)
        if n == 0:
            continue
        if n < min_samples:
            val = float("nan")
        else:
            try:
                val = _compute_metric(metric, chunk["y_true"].values, chunk["y_pred"].values)
            except Exception as exc:
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
    except Exception:
        pass
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by time bin")
    ax.legend(loc="best")
    return fig
