"""Model performance reporting functions extracted from ``evaluation.py``.

``report_model_perf`` — top-level dispatch (regression vs classification).
``report_regression_model_perf`` — MAE, RMSE, MaxError, R2, scatter plots.
``report_probabilistic_model_perf`` — AUC, logloss, calibration, ICE, PR curves.

All three accept pre-computed ``preds`` / ``probs`` (from cached models)
or live-compute via ``model.predict`` when none are provided.
"""

from __future__ import annotations

import logging
import os
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment] -- plot branches gated; headless envs skip

try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None  # type: ignore[assignment] -- only used in optional sklearn-multilabel report path

from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
# Metrics: use mlframe's fast njit versions, not sklearn
from sklearn.preprocessing import LabelEncoder

from mlframe.metrics.core import compute_fairness_metrics, fast_calibration_report, fast_mean_absolute_error, fast_max_error, fast_r2_score, fast_regression_metrics_block, fast_roc_auc, fast_root_mean_squared_error
from pyutilz.pythonlib import get_human_readable_set_size

# .evaluation imports back from ._reporting; deferring breaks the cycle.
# (See line 477 / 580 for the two call-sites.)
from .phases import phase

if TYPE_CHECKING:
    from .configs import MultilabelDispatchConfig  # forward annotation only; importing at runtime is unnecessary

# Inline to avoid circular import (_reporting <- evaluation <- _reporting)
DEFAULT_PLOT_SAMPLE_SIZE = 500
DEFAULT_REPORT_NDIGITS = 2
DEFAULT_CALIB_REPORT_NDIGITS = 2
DEFAULT_NBINS = 10
DEFAULT_FIGSIZE = (15, 5)
DEFAULT_RANDOM_SEED = 42

logger = logging.getLogger(__name__)

try:
    from IPython.display import display as _ipython_display
except ImportError:  # pragma: no cover
    _ipython_display = None


def _maybe_display(obj):
    """Display ``obj`` in Jupyter; no-op in scripts/CI."""
    if _ipython_display is not None:
        _ipython_display(obj)


def _canonical_multilabel_y(targets) -> np.ndarray:
    """Coerce a multilabel-shaped target to a clean ``(N, K) ndarray``.

    Accepts the three shapes the suite encounters in production:
    1. ``np.ndarray`` 2-D ``(N, K)`` — passed through.
    2. ``pd.DataFrame`` — ``.values``.
    3. ``np.ndarray`` 1-D ``object`` dtype where each cell is a
       length-K array-like (the polars ``pl.List(pl.Int8)`` /
       ``pl.List(pl.Float32)`` -> pandas object roundtrip). Stacked
       to 2-D via ``np.stack``.
    Otherwise returns ``np.asarray(targets)`` unchanged (1-D / scalar
    / unhandled — caller decides).

    For float-typed cells (``pl.List(pl.Float32)`` source), the
    output is cast to ``int64`` via ``>= 0.5`` threshold:
    downstream metrics expect ``{0, 1}`` indicators.

    Extracted from the inline canonicalization at evaluation.py so
    ``mlframe.training.dummy_baselines`` and other consumers share
    one path.
    """
    if isinstance(targets, pd.DataFrame):
        targets_arr = targets.values
    elif isinstance(targets, np.ndarray):
        targets_arr = targets
    else:
        targets_arr = np.asarray(targets)

    if targets_arr.dtype == object and targets_arr.ndim == 1 and targets_arr.shape[0] > 0:
        first = targets_arr[0]
        if hasattr(first, "shape") or (
            hasattr(first, "__len__") and not isinstance(first, (str, bytes))
        ):
            try:
                targets_arr = np.stack([np.asarray(c) for c in targets_arr], axis=0)
            except Exception:
                # Unstackable (jagged / mixed shape) — leave as-is so the
                # caller's exception path surfaces the underlying issue.
                return targets_arr

    # Coerce float-indicator multilabel to int via threshold.
    if targets_arr.ndim == 2 and targets_arr.dtype.kind == "f":
        targets_arr = (targets_arr >= 0.5).astype(np.int64)

    return targets_arr


def report_model_perf(
    targets: np.ndarray | pd.Series,
    columns: Sequence[str],
    model_name: str,
    model: ClassifierMixin | RegressorMixin | None,
    subgroups: dict[str, np.ndarray] | None = None,
    subset_index: np.ndarray | None = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = DEFAULT_CALIB_REPORT_NDIGITS,
    verbose: bool = False,
    classes: Sequence | None = None,
    preds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    target_label_encoder: LabelEncoder | None = None,
    nbins: int = DEFAULT_NBINS,
    print_report: bool = True,
    show_perf_chart: bool = True,
    show_fi: bool = True,
    fi_kwargs: dict[str, Any] | None = None,
    plot_file: str = "",
    custom_ice_metric: Callable | None = None,
    custom_rice_metric: Callable | None = None,
    metrics: dict[str, Any] | None = None,
    group_ids: np.ndarray | None = None,
    n_features: int | None = None,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: tuple[str, ...] | None = None,
    multilabel_dispatch_config: MultilabelDispatchConfig | None = None,
    plot_outputs: str | None = None,
    plot_dpi: int | None = None,
    target_type: str | None = None,
    multiclass_panels: str | None = None,
    multilabel_panels: str | None = None,
    ltr_panels: str | None = None,
    quantile_panels: str | None = None,
    quantile_alphas: Sequence[float] | None = None,
    y_train_envelope_stats: Any = None,
    reporting_config: Any = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Generate a unified performance report for both classifiers and regressors.

    Automatically detects model type and routes to the appropriate reporting
    function (classification or regression).

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target values.
    columns : Sequence[str]
        Feature column names used for training.
    model_name : str
        Name of the model for display in reports.
    model : ClassifierMixin, RegressorMixin, or None
        Trained model. Can be None if preds/probs are provided.
    subgroups : dict, optional
        Dictionary mapping subgroup names to boolean arrays for fairness analysis.
    subset_index : np.ndarray, optional
        Indices to subset the data for fairness analysis.
    report_ndigits : int, default=4
        Number of decimal digits for metric reporting.
    figsize : tuple, default=(15, 5)
        Figure size for plots.
    report_title : str, default=""
        Title prefix for reports.
    use_weights : bool, default=True
        Whether to use weighted calibration metrics.
    calib_report_ndigits : int, default=2
        Decimal digits for calibration metrics.
    verbose : bool, default=False
        Enable verbose output.
    classes : Sequence, optional
        Class labels for classification.
    preds : np.ndarray, optional
        Pre-computed predictions.
    probs : np.ndarray, optional
        Pre-computed probabilities for classification.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    target_label_encoder : LabelEncoder, optional
        Encoder for target labels.
    nbins : int, default=10
        Number of bins for calibration analysis.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display performance charts.
    show_fi : bool, default=True
        Whether to show feature importances.
    fi_kwargs : dict, optional
        Additional kwargs for feature importance plotting.
    plot_file : str, default=""
        Base path for saving plots.
    custom_ice_metric : Callable, optional
        Custom integral calibration error metric.
    custom_rice_metric : Callable, optional
        Custom robust integral calibration error metric.
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).
    group_ids : np.ndarray, optional
        Group identifiers for grouped calibration analysis.

    Returns
    -------
    tuple
        (preds, probs) for classification, (preds, None) for regression.
    """
    if fi_kwargs is None:
        fi_kwargs = {}

    # Common parameters shared by both classification and regression
    common_params = dict(
        targets=targets,
        columns=columns,
        model_name=model_name,
        model=model,
        subgroups=subgroups,
        subset_index=subset_index,
        report_ndigits=report_ndigits,
        figsize=figsize,
        report_title=report_title,
        verbose=verbose,
        preds=preds,
        df=df,
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        plot_file=plot_file,
        metrics=metrics,
        n_features=n_features,
    )

    # sklearn>=1.6 raises AttributeError when is_classifier(None) triggers
    # get_tags(None) (formerly just returned False). The just_evaluate=True
    # path passes model=None with pre-computed preds/probs -- infer task type
    # from whether probs were supplied (presence of probs => classification).
    if model is None:
        is_probabilistic = probs is not None
    else:
        is_probabilistic = is_classifier(model) or type(model).__name__ == "NGBClassifier"
    if is_probabilistic:
        with phase(
            "report_probabilistic_model_perf",
            n_rows=(len(targets) if hasattr(targets, '__len__') else None),
        ):
            preds, probs = report_probabilistic_model_perf(
                **common_params,
                use_weights=use_weights,
                calib_report_ndigits=calib_report_ndigits,
                classes=classes,
                probs=probs,
                target_label_encoder=target_label_encoder,
                nbins=nbins,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                group_ids=group_ids,
                show_prob_histogram=show_prob_histogram,
                prob_histogram_yscale=prob_histogram_yscale,
                show_inline_population_labels=show_inline_population_labels,
                title_metrics_tokens=title_metrics_tokens,
                multilabel_dispatch_config=multilabel_dispatch_config,
                plot_dpi=plot_dpi,
            )
    else:
        with phase(
            "report_regression_model_perf",
            n_rows=(len(targets) if hasattr(targets, '__len__') else None),
        ):
            # Thread plot_outputs explicitly so the regression
            # report can pick the plotly DSL path. Probabilistic path
            # uses plot_outputs differently (multi_target_panels at L277);
            # regression's only consumer is the residual-diagnostics chart.
            #
            # Decompose ``y_train_envelope_stats`` (computed once per
            # (model, target) in the trainer) into the three legacy
            # kwargs so the prediction-envelope clip uses the TRAIN
            # bound (k=3 sigma) instead of the per-split eval-fallback
            # bound (k=10 sigma). Composite-target wrap pass attaches
            # its own y-scale stats; degenerate train targets pass
            # None and the reporter falls back to eval-derived bounds.
            _ytmin = _ytmax = _ytstd = None
            if y_train_envelope_stats is not None:
                _ytmin = getattr(y_train_envelope_stats, "y_min", None)
                _ytmax = getattr(y_train_envelope_stats, "y_max", None)
                _ytstd = getattr(y_train_envelope_stats, "y_std", None)
            preds, probs = report_regression_model_perf(
                **common_params, plot_outputs=plot_outputs, plot_dpi=plot_dpi,
                y_train_min=_ytmin, y_train_max=_ytmax, y_train_std=_ytstd,
                reporting_config=reporting_config,
            )

    # Render multiclass / multilabel / LTR panel
    # grids when the caller has supplied per-target_type templates +
    # output DSL. No-op for binary classification / regression / when
    # templates are unset. Failures are logged + swallowed (panels are
    # additive; existing perf chart + FI still emit).
    if plot_file and plot_outputs and (
        multiclass_panels or multilabel_panels or ltr_panels or quantile_panels
    ):
        from mlframe.reporting.auto_dispatch import render_multi_target_panels
        with phase("render_multi_target_panels"):
            render_multi_target_panels(
                targets=np.asarray(targets) if not isinstance(targets, np.ndarray) else targets,
                probs=probs, preds=preds,
                classes=classes, group_ids=group_ids,
                quantile_alphas=quantile_alphas,
                plot_outputs=plot_outputs,
                plot_dpi=plot_dpi,
                multiclass_panels=multiclass_panels,
                multilabel_panels=multilabel_panels,
                ltr_panels=ltr_panels,
                quantile_panels=quantile_panels,
                base_path=plot_file,
                suptitle=(report_title + " " + model_name).strip(),
                # Authoritative target_type gate — prevents regression
                # targets that happen to carry group_ids from FTE
                # (grouped-CV pattern) from incorrectly triggering the
                # LTR panels branch, which used to render NDCG/MRR
                # nonsense for regression + paid 10-30s on 5M rows.
                target_type=target_type,
            )

    if show_fi:
        n_cols = n_features if n_features is not None else (len(columns) if columns is not None and len(columns) > 0 else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        with phase(
            "plot_feature_importances",
            model=type(model).__name__,
            n_cols=len(columns) if columns is not None and len(columns) > 0 else 0,
        ):
            # Lazy import: _reporting <- evaluation <- _reporting cycle (see comment at top of file).
            from .evaluation import plot_model_feature_importances
            # Thread df + targets through to power the permutation-FI
            # fallback for estimators without native ``feature_importances_``
            # / ``coef_`` (PyTorch-Lightning MLP, Keras nets, custom predict-
            # only wrappers). When the inner exposes a native source the
            # permutation path is skipped automatically.
            _fi_X = df[list(columns)] if (df is not None and columns is not None and len(columns) > 0) else None
            feature_importances = plot_model_feature_importances(
                model=model,
                columns=columns,
                model_name=(report_title + " " + model_name + f" [{nfeatures}{get_human_readable_set_size(len(preds))} rows]").strip(),
                plot_file=plot_file + "_fiplot.png" if plot_file else "",
                X=_fi_X,
                y=targets,
                **fi_kwargs,
            )
        if metrics is not None:
            metrics.update({"feature_importances": feature_importances})

    return preds, probs


# Wave 97 (2026-05-21): report_probabilistic_model_perf (~520 lines)
# moved to sibling file _reporting_probabilistic.py to drop this file
# below the 1k-line monolith threshold. Re-exported below so existing
# callers (`from .._reporting import report_probabilistic_model_perf`)
# keep working.
from ._reporting_probabilistic import report_probabilistic_model_perf  # noqa: F401, E402
# report_regression_model_perf (~650 lines) moved to sibling file
# _reporting_regression.py for the same 1k-line monolith threshold;
# re-exported below.
from ._reporting_regression import report_regression_model_perf  # noqa: F401, E402

