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

from mlframe.metrics.core import compute_fairness_metrics, fast_calibration_report, fast_mean_absolute_error, fast_max_error, fast_r2_score, fast_roc_auc, fast_root_mean_squared_error
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
            preds, probs = report_regression_model_perf(
                **common_params, plot_outputs=plot_outputs, plot_dpi=plot_dpi,
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
            feature_importances = plot_model_feature_importances(
                model=model,
                columns=columns,
                model_name=(report_title + " " + model_name + f" [{nfeatures}{get_human_readable_set_size(len(preds))} rows]").strip(),
                plot_file=plot_file + "_fiplot.png" if plot_file else "",
                **fi_kwargs,
            )
        if metrics is not None:
            metrics.update({"feature_importances": feature_importances})

    return preds, probs


def report_regression_model_perf(
    targets: np.ndarray | pd.Series,
    columns: Sequence[str],
    model_name: str,
    model: RegressorMixin | None,
    subgroups: dict[str, np.ndarray] | None = None,
    subset_index: np.ndarray | None = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    verbose: bool = False,
    preds: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    plot_marker: str = "o",
    plot_sample_size: int = DEFAULT_PLOT_SAMPLE_SIZE,
    metrics: dict[str, Any] | None = None,
    n_features: int | None = None,
    plot_outputs: str | None = None,
    plot_dpi: int | None = None,
) -> tuple[np.ndarray, None]:
    """
    Generate a detailed performance report for regression models.

    Computes and optionally displays MAE, RMSE, MaxError, R^2 metrics,
    scatter plots of predictions vs actuals, and fairness analysis.

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target values.
    columns : Sequence[str]
        Feature column names.
    model_name : str
        Name of the model for display.
    model : RegressorMixin or None
        Trained regression model. Can be None if preds are provided.
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
    verbose : bool, default=False
        Enable verbose output.
    preds : np.ndarray, optional
        Pre-computed predictions. If None, generated from model.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display performance charts.
    plot_file : str, default=""
        Path for saving the plot.
    plot_marker : str, default="o"
        Marker style for scatter plot.
    plot_sample_size : int, default=500
        Maximum number of points to plot (for performance).
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).

    Returns
    -------
    tuple
        (preds, None) - predictions and None (no probabilities for regression).
    """
    if preds is None:
        # Wrap in _predict_with_fallback so CatBoost's Polars fastpath
        # dispatcher misses ("No matching signature found") trigger a
        # symmetric pandas fallback -- same pattern as fit. Without this
        # wrap, a polars val/test DF + a CB model whose fit path fell
        # back to pandas would crash at predict time with the identical
        # opaque TypeError (prod incident).
        from mlframe.training.trainer import _predict_with_fallback
        preds = _predict_with_fallback(model, df, method="predict")

    if isinstance(targets, pd.Series):
        targets = targets.values

    # Numba fast helpers now cover full sklearn signature
    # (1-D / 2-D, sample_weight, multioutput) so all regression metric
    # call sites here go through the fast path. 6-23x faster than
    # sklearn at production size.
    targets_arr = np.asarray(targets)
    preds_arr = np.asarray(preds)
    if targets_arr.ndim > 1 and targets_arr.shape[1] > 1:
        # WARN-loud when this multioutput path
        # fires. Multilabel classification SHOULD route to
        # ``report_probabilistic_model_perf`` via the
        # ``is_classifier(model)`` dispatch upstream; (N, K) reaching the
        # regression path means a classifier-mixin wasn't recognised
        # (likely a model wrapped in something that strips
        # ClassifierMixin). Fast helpers now compute per-output
        # correctly, but the dispatch root cause still needs fixing.
        logger.warning(
            "report_regression_model_perf received a multioutput target "
            "(shape=%s); this almost certainly indicates an upstream "
            "is_classifier dispatch bug for a multilabel-wrapped "
            "estimator. The metrics will compute per-output but the "
            "dispatch should route to report_probabilistic_model_perf "
            "instead.",
            targets_arr.shape,
        )
    MAE = fast_mean_absolute_error(targets_arr, preds_arr)
    # 2-D max_error returns per-output array; the existing reporting
    # contract is a single scalar (overall max), so reduce explicitly.
    _max_err = fast_max_error(targets_arr, preds_arr)
    MaxError = float(np.max(_max_err)) if isinstance(_max_err, np.ndarray) else _max_err
    R2 = fast_r2_score(targets_arr, preds_arr)
    RMSE = fast_root_mean_squared_error(targets_arr, preds_arr)

    current_metrics = dict(
        MAE=MAE,
        MaxError=MaxError,
        R2=R2,
        RMSE=RMSE,
    )
    if metrics is not None:
        metrics.update(current_metrics)

    # Compute residual audit ONCE (used by both the chart and the print-report block; cheap thanks to internal sampling).
    # ``behavior_config.report_residual_audit`` is a LOG-ONLY toggle. When False we MUST still compute the audit so the chart's hist + resid-vs-pred panels stay populated -- only the multi-line verdict text in the log is suppressed. Multi-output targets still skip the audit (no scalar residuals to fit a distribution to).
    _residual_audit = None
    from .evaluation import _get_residual_audit_enabled  # lazy: breaks cycle with .evaluation
    _audit_log_enabled = bool(_get_residual_audit_enabled())
    if not (
        (targets_arr.ndim > 1 and targets_arr.shape[1] > 1)
        or (preds_arr.ndim > 1 and preds_arr.shape[1] > 1)
    ):
        try:
            from .regression_residual_audit import audit_residuals as _audit_residuals_fn
            _residual_audit = _audit_residuals_fn(targets, preds)
            if metrics is not None:
                metrics["residual_audit"] = _residual_audit.to_dict()
        except Exception as _audit_err:
            logger.warning(
                "residual_audit failed for '%s': %s. Continuing without diagnostics.",
                model_name, _audit_err,
            )

    # Short-circuit when there is NO consumer for the chart.
    # Same logic as ``mlframe.metrics.core.show_calibration_plot``: in a script /
    # CI / fuzz process (no IPython kernel, no ``sys.ps1``) the
    # ``show_perf_chart=True`` default renders a matplotlib figure that
    # nobody can see AND nothing is written to disk because ``plot_file``
    # is empty. The figure render is 100-200 ms / call and dominates
    # warm-state regression report wall.
    if show_perf_chart and not plot_file:
        try:
            _is_interactive_session = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            import sys as _sys
            _is_interactive_session = hasattr(_sys, "ps1")
        if not _is_interactive_session:
            show_perf_chart = False  # disable for the guards below

    if show_perf_chart or plot_file:
        # Split the long title into three pieces.
        # - ``header_str``: split / model_name + [features/rows] -> figure SUPTITLE
        # - ``metrics_str``: MAE / RMSE / MaxError / R2 -> scatter (left) title
        # - residual hypothesis (formerly tacked onto scatter) -> moved
        #   entirely to the histogram (middle) panel by
        #   ``plot_residual_diagnostics``.
        n_cols = n_features if n_features is not None else (len(columns) if columns is not None and len(columns) > 0 else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        header_str = (
            report_title + " " + model_name +
            f" [{nfeatures}{get_human_readable_set_size(len(targets))} rows]"
        )
        from ._format import format_metric as _fmt
        metrics_str = (
            f"MAE={_fmt(MAE, report_ndigits)}"
            f" RMSE={_fmt(RMSE, report_ndigits)}"
            f" MaxError={_fmt(MaxError, report_ndigits)}"
            f" R2={_fmt(R2, report_ndigits)}"
        )
        # ``title`` retained for the (deprecated) print-report path that still concatenates everything for stdout. Charts use the split.
        title = header_str + "\n " + metrics_str  # noqa: F841 -- see comment above

        # For (N, K) multilabel-as-regression
        # targets the scatter plot below would do
        # ``np.argsort(preds[idx])`` on a 2-D array (which sorts rows
        # element-wise instead of by-row), then ``plt.scatter`` would
        # emit K overlapping point clouds with no visual separation.
        # Skip the plot when targets are 2-D -- title metrics already
        # carry the per-output-aggregated MAE/RMSE/R2.
        _is_multioutput = (
            (targets_arr.ndim > 1 and targets_arr.shape[1] > 1)
            or (preds_arr.ndim > 1 and preds_arr.shape[1] > 1)
        )
        if _is_multioutput:
            if print_report:
                logger.info(
                    f"  [multioutput regression: target shape={targets_arr.shape}, "
                    f"skipping scatter plot -- per-output plotting would mix K clouds]"
                )
        else:
            from .regression_residual_audit import (
                plot_residual_diagnostics as _plot_residual_diagnostics,
            )
            _audit = _residual_audit  # reuse pre-computed audit

            # When ReportingConfig.plot_outputs is set
            # (default ``"plotly[html,png]"``), bypass the inline
            # matplotlib block and route through the FigureSpec / DSL
            # pipeline so plotly + matplotlib emit per the user's
            # config. The DSL builder lives at
            # ``mlframe/reporting/charts/regression.py``.
            if plot_outputs and plot_file and _audit is not None:
                _plot_residual_diagnostics(
                    targets, preds, audit=_audit,
                    plot_outputs=plot_outputs,
                    base_path=plot_file,
                    header_str=header_str,
                    metrics_str=metrics_str,
                    plot_sample_size=plot_sample_size,
                    seed=DEFAULT_RANDOM_SEED,
                    dpi=plot_dpi,
                )
            else:
                # Legacy matplotlib-only path. Kept for callers that
                # still want axes injection (e.g. notebooks composing
                # the chart into a larger figure).
                #
                # Local RNG -- do not pollute global numpy state. Cache
                # by (n, size, seed) so repeated reports on the same
                # prediction length reuse the sample.
                from .evaluation import _get_cached_plot_idx  # lazy: breaks cycle with .evaluation
                idx = _get_cached_plot_idx(len(preds), plot_sample_size, DEFAULT_RANDOM_SEED)
                idx = idx[np.argsort(preds[idx])]

                # Three-panel figure: scatter | residuals histogram | residuals vs predicted.
                # constrained_layout cached solver state -- ~13s saved
                # vs tight_layout per-chart on multi-chart reports.
                # Honour plot_dpi when caller set it.
                _reg_subplots_kwargs = dict(
                    figsize=(figsize[0] * 3 / 2, figsize[1]),
                    layout="constrained",
                )
                if plot_dpi is not None:
                    _reg_subplots_kwargs["dpi"] = plot_dpi
                fig, axes = plt.subplots(1, 3, **_reg_subplots_kwargs)
                ax_scatter, ax_hist, ax_resid = axes

                # y=1.02 puts the suptitle ABOVE the
                # axes region so constrained_layout auto-extends the
                # top margin. Previously y=0.995 placed the suptitle
                # inside the axes row, causing collision with the
                # multiline subplot titles (hist/resid panels carry
                # 2-line titles for hypothesis + heteroscedasticity).
                fig.suptitle(header_str, fontsize=11, y=1.02)

                ax_scatter.scatter(
                    preds[idx], targets[idx], marker=plot_marker, alpha=0.3,
                )
                ax_scatter.plot(
                    preds[idx], preds[idx], linestyle="--", color="green",
                    label="Perfect fit",
                )
                ax_scatter.set_xlabel("Predictions")
                ax_scatter.set_ylabel("True values")
                ax_scatter.set_title(metrics_str)
                ax_scatter.grid(True, alpha=0.3)
                ax_scatter.legend(loc="best", fontsize=8, framealpha=0.7)

                if _audit is not None:
                    _plot_residual_diagnostics(
                        targets, preds, audit=_audit,
                        ax_hist=ax_hist, ax_resid_vs_pred=ax_resid,
                    )
                else:
                    ax_hist.set_visible(False)
                    ax_resid.set_visible(False)

                if plot_file:
                    fig.savefig(plot_file)

                if show_perf_chart:
                    plt.ion()
                    plt.show()
                # Leak fix: close unless interactive (Jupyter inline).
                from mlframe.metrics.core import _close_unless_interactive
                _close_unless_interactive(fig, was_shown=show_perf_chart)

    if print_report:
        # Route through logger so file handlers (e.g.
        # pyutilz.logginglib.init_logging) capture the report block.
        # Pre-fix: bare print() bypassed the logging system entirely
        # → cell output ✓ but file handler ✗. Operators using
        # init_logging in jupyter notebooks lost the metric blocks
        # from on-disk logs.
        from ._format import format_metric as _fmt
        # Annotate composite-target reports as T-scale. Composite targets carry ``MTRESID=`` in the model_name (stamped by ``select_target``); this indicates the printed metrics live on the RESIDUAL scale, not the raw y-scale. The wrap pass separately emits y-scale numbers via ``[CompositeTargetEstimator] ... y-scale metrics:`` so the operator can compare apples-to-apples with raw-target reports.
        _is_t_scale_composite = "MTRESID=" in model_name
        _scale_note = (
            "  (T-scale residual; y-scale metrics in "
            "[CompositeTargetEstimator] log line)"
            if _is_t_scale_composite else ""
        )
        # One-line metrics in the log block.
        # Reuses the SAME ``metrics_str`` formula the chart title carries
        # (``MAE=... RMSE=... MaxError=... R2=...`` separated by spaces) so
        # the log line is immediately searchable / regex-friendly and the
        # chart + log show the same string. Previous layout emitted 4
        # separate ``MAE: ...`` lines, padding production logs needlessly.
        _metrics_one_line = (
            f"MAE={_fmt(MAE, report_ndigits)}"
            f" RMSE={_fmt(RMSE, report_ndigits)}"
            f" MaxError={_fmt(MaxError, report_ndigits)}"
            f" R2={_fmt(R2, report_ndigits)}"
        )
        _report_lines = [
            report_title + " " + model_name + _scale_note,
            _metrics_one_line,
        ]
        # Residual-audit VERDICT TEXT is gated on the suite flag.
        # The audit is still computed above (so the chart panels stay
        # populated); only the multi-line text block here is suppressed when
        # ``behavior_config.report_residual_audit=False``.
        if _residual_audit is not None and _audit_log_enabled:
            from .regression_residual_audit import (
                format_residual_audit_report as _fmt_residual_audit,
            )
            _report_lines.append(_fmt_residual_audit(_residual_audit))
        # Single multi-line logger.info call so the file handler picks
        # up one record per report block (vs N records for the original
        # per-line prints). Cell renderers join multiline messages
        # cleanly under one record.
        logger.info("\n".join(_report_lines))

    if subgroups:
        fairness_report = compute_fairness_metrics(
            subgroups=subgroups,
            subset_index=subset_index,
            y_true=targets,
            y_pred=preds,
            metrics={"MAE": fast_mean_absolute_error, "RMSE": fast_root_mean_squared_error},
            metrics_higher_is_better={"MAE": False, "RMSE": False},
        )
        if fairness_report is not None:
            if print_report:
                _maybe_display(fairness_report)
            if metrics is not None:
                metrics.update(dict(fairness_report=fairness_report))

    return preds, None


# Wave 97 (2026-05-21): report_probabilistic_model_perf (~520 lines)
# moved to sibling file _reporting_probabilistic.py to drop this file
# below the 1k-line monolith threshold. Re-exported below so existing
# callers (`from .._reporting import report_probabilistic_model_perf`)
# keep working.
from ._reporting_probabilistic import report_probabilistic_model_perf  # noqa: F401, E402

