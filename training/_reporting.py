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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.preprocessing import LabelEncoder

from .phases import phase
from .configs import DEFAULT_REPORT_NDIGITS, DEFAULT_FIGSIZE, DEFAULT_NBINS, DEFAULT_PLOT_SAMPLE_SIZE

logger = logging.getLogger(__name__)

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
    output is cast to ``int64`` via ``>= 0.5`` threshold (round-3
    audit A#22): downstream metrics expect ``{0, 1}`` indicators.

    Extracted from the inline canonicalization at evaluation.py
    (Session 2026-04-28) so ``mlframe.training.dummy_baselines`` and
    other consumers share one path.
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

    # Coerce float-indicator multilabel to int via threshold (round-3 A#22).
    if targets_arr.ndim == 2 and targets_arr.dtype.kind == "f":
        targets_arr = (targets_arr >= 0.5).astype(np.int64)

    return targets_arr


def report_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence[str],
    model_name: str,
    model: Optional[Union[ClassifierMixin, RegressorMixin]],
    subgroups: Optional[Dict[str, np.ndarray]] = None,
    subset_index: Optional[np.ndarray] = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = DEFAULT_CALIB_REPORT_NDIGITS,
    verbose: bool = False,
    classes: Optional[Sequence] = None,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    target_label_encoder: Optional[LabelEncoder] = None,
    nbins: int = DEFAULT_NBINS,
    print_report: bool = True,
    show_perf_chart: bool = True,
    show_fi: bool = True,
    fi_kwargs: Optional[Dict[str, Any]] = None,
    plot_file: str = "",
    custom_ice_metric: Optional[Callable] = None,
    custom_rice_metric: Optional[Callable] = None,
    metrics: Optional[Dict[str, Any]] = None,
    group_ids: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: Optional[Tuple[str, ...]] = None,
    multilabel_dispatch_config: Optional["MultilabelDispatchConfig"] = None,
    plot_outputs: Optional[str] = None,
    plot_dpi: Optional[int] = None,
    target_type: Optional[str] = None,
    multiclass_panels: Optional[str] = None,
    multilabel_panels: Optional[str] = None,
    ltr_panels: Optional[str] = None,
    quantile_panels: Optional[str] = None,
    quantile_alphas: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
            # 2026-05-09: thread plot_outputs explicitly so the regression
            # report can pick the plotly DSL path. Probabilistic path
            # uses plot_outputs differently (multi_target_panels at L277);
            # regression's only consumer is the residual-diagnostics chart.
            preds, probs = report_regression_model_perf(
                **common_params, plot_outputs=plot_outputs, plot_dpi=plot_dpi,
            )

    # 2026-05-08 PR2 wiring: render multiclass / multilabel / LTR panel
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
        n_cols = n_features if n_features is not None else (len(columns) if columns else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        with phase(
            "plot_feature_importances",
            model=type(model).__name__,
            n_cols=len(columns) if columns else 0,
        ):
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
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence[str],
    model_name: str,
    model: Optional[RegressorMixin],
    subgroups: Optional[Dict[str, np.ndarray]] = None,
    subset_index: Optional[np.ndarray] = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    verbose: bool = False,
    preds: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    plot_marker: str = "o",
    plot_sample_size: int = DEFAULT_PLOT_SAMPLE_SIZE,
    metrics: Optional[Dict[str, Any]] = None,
    n_features: Optional[int] = None,
    plot_outputs: Optional[str] = None,
    plot_dpi: Optional[int] = None,
) -> Tuple[np.ndarray, None]:
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
        # opaque TypeError (2026-04-19 prod incident).
        from mlframe.training.trainer import _predict_with_fallback
        preds = _predict_with_fallback(model, df, method="predict")

    if isinstance(targets, pd.Series):
        targets = targets.values

    # 2026-05-09: numba fast helpers now cover full sklearn signature
    # (1-D / 2-D, sample_weight, multioutput) so all regression metric
    # call sites here go through the fast path. 6-23x faster than
    # sklearn at production size.
    targets_arr = np.asarray(targets)
    preds_arr = np.asarray(preds)
    if targets_arr.ndim > 1 and targets_arr.shape[1] > 1:
        # 2026-04-28 (batch 4): WARN-loud when this multioutput path
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
    # 2026-05-12 (user clarification): ``behavior_config.report_residual_audit`` is a LOG-ONLY toggle. When False we MUST still compute the audit so the chart's hist + resid-vs-pred panels stay populated -- only the multi-line verdict text in the log is suppressed. Multi-output targets still skip the audit (no scalar residuals to fit a distribution to).
    _residual_audit = None
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

    # 2026-05-09: short-circuit when there is NO consumer for the chart.
    # Same logic as ``mlframe.metrics.show_calibration_plot``: in a script /
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
        # 2026-05-08 (user feedback): split the long title into three pieces.
        # - ``header_str``: split / model_name + [features/rows] -> figure SUPTITLE
        # - ``metrics_str``: MAE / RMSE / MaxError / R2 -> scatter (left) title
        # - residual hypothesis (formerly tacked onto scatter) -> moved
        #   entirely to the histogram (middle) panel by
        #   ``plot_residual_diagnostics``.
        n_cols = n_features if n_features is not None else (len(columns) if columns else 0)
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
        # ``title`` retained for the (deprecated) print-report path that
        # still concatenates everything for stdout. Charts use the split.
        title = header_str + "\n " + metrics_str

        # 2026-04-27 (batch 3): for (N, K) multilabel-as-regression
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

            # 2026-05-09: when ReportingConfig.plot_outputs is set
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
                idx = _get_cached_plot_idx(len(preds), plot_sample_size, DEFAULT_RANDOM_SEED)
                idx = idx[np.argsort(preds[idx])]

                # Three-panel figure: scatter | residuals histogram | residuals vs predicted.
                # constrained_layout cached solver state -- ~13s saved
                # vs tight_layout per-chart on multi-chart reports.
                # 2026-05-11: honour plot_dpi when caller set it.
                _reg_subplots_kwargs = dict(
                    figsize=(figsize[0] * 3 / 2, figsize[1]),
                    layout="constrained",
                )
                if plot_dpi is not None:
                    _reg_subplots_kwargs["dpi"] = plot_dpi
                fig, axes = plt.subplots(1, 3, **_reg_subplots_kwargs)
                ax_scatter, ax_hist, ax_resid = axes

                # 2026-05-09 fix: y=1.02 puts the suptitle ABOVE the
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
                from mlframe.metrics import _close_unless_interactive
                _close_unless_interactive(fig, was_shown=show_perf_chart)

    if print_report:
        # 2026-05-10: route through logger so file handlers (e.g.
        # pyutilz.logginglib.init_logging) capture the report block.
        # Pre-fix: bare print() bypassed the logging system entirely
        # → cell output ✓ but file handler ✗. Operators using
        # init_logging in jupyter notebooks lost the metric blocks
        # from on-disk logs.
        from ._format import format_metric as _fmt
        # C2 (2026-05-11): annotate composite-target reports as T-scale. Composite targets carry ``MTRESID=`` in the model_name (stamped by ``select_target``); this indicates the printed metrics live on the RESIDUAL scale, not the raw y-scale. The wrap pass separately emits y-scale numbers via ``[CompositeTargetEstimator] ... y-scale metrics:`` so the operator can compare apples-to-apples with raw-target reports.
        _is_t_scale_composite = "MTRESID=" in model_name
        _scale_note = (
            "  (T-scale residual; y-scale metrics in "
            "[CompositeTargetEstimator] log line)"
            if _is_t_scale_composite else ""
        )
        # 2026-05-12 (user request): one-line metrics in the log block.
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
        # 2026-05-12: residual-audit VERDICT TEXT is gated on the suite flag.
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
            metrics={"MAE": mean_absolute_error, "RMSE": root_mean_squared_error},
            metrics_higher_is_better={"MAE": False, "RMSE": False},
        )
        if fairness_report is not None:
            if print_report:
                display(fairness_report)
            if metrics is not None:
                metrics.update(dict(fairness_report=fairness_report))

    return preds, None


def report_probabilistic_model_perf(
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence[str],
    model_name: str,
    model: Optional[ClassifierMixin],
    subgroups: Optional[Dict[str, np.ndarray]] = None,
    subset_index: Optional[np.ndarray] = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = DEFAULT_CALIB_REPORT_NDIGITS,
    verbose: bool = False,
    classes: Optional[Sequence] = None,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    target_label_encoder: Optional[LabelEncoder] = None,
    nbins: int = DEFAULT_NBINS,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    custom_ice_metric: Optional[Callable] = None,
    custom_rice_metric: Optional[Callable] = None,
    metrics: Optional[Dict[str, Any]] = None,
    group_ids: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: Optional[Tuple[str, ...]] = None,
    multilabel_dispatch_config: Optional["MultilabelDispatchConfig"] = None,
    plot_dpi: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a detailed performance report for probabilistic classification models.

    Computes and displays classification metrics including ROC AUC, PR AUC,
    calibration metrics, Brier loss, log loss, and fairness analysis.

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target labels.
    columns : Sequence[str]
        Feature column names.
    model_name : str
        Name of the model for display.
    model : ClassifierMixin or None
        Trained classification model. Can be None if probs are provided.
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
        Class labels. If None, inferred from model or targets.
    preds : np.ndarray, optional
        Pre-computed class predictions.
    probs : np.ndarray, optional
        Pre-computed class probabilities. If None, generated from model.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    target_label_encoder : LabelEncoder, optional
        Encoder for converting numeric labels to string names.
    nbins : int, default=10
        Number of bins for calibration analysis.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display calibration and performance charts.
    plot_file : str, default=""
        Base path for saving plots.
    custom_ice_metric : Callable, optional
        Custom integral calibration error metric function.
    custom_rice_metric : Callable, optional
        Custom robust integral calibration error metric function.
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).
    group_ids : np.ndarray, optional
        Group identifiers for grouped calibration analysis.

    Returns
    -------
    tuple
        (preds, probs) - class predictions and probability arrays.
    """
    if probs is None:
        # Lazy import avoids circular: trainer.py already imports from
        # evaluation.py at module level.
        from mlframe.training.trainer import _predict_with_fallback
        try:
            # _predict_with_fallback handles the CatBoost Polars-fastpath
            # dispatcher miss ("No matching signature found") symmetrically
            # with fit's fallback. Any OTHER error (model has no
            # predict_proba, returns NotImplemented, or a non-CB TypeError)
            # bubbles to the outer except and hits the predict() fallback
            # path below -- with the same Polars fallback wrapping so we
            # don't retry into the same dispatcher miss (2026-04-19 bug).
            probs = _predict_with_fallback(model, df, method="predict_proba")
        except (AttributeError, TypeError, NotImplementedError) as e:
            logger.warning(f"predict_proba not available for {type(model).__name__}, using predict() instead: {e}")
            preds_fallback = _predict_with_fallback(model, df, method="predict")

            if hasattr(model, "classes_"):
                n_classes = len(model.classes_)
                class_indices = np.searchsorted(model.classes_, preds_fallback)
            else:
                n_classes = len(np.unique(preds_fallback))
                class_indices = preds_fallback.astype(int)

            probs = np.zeros((len(preds_fallback), n_classes))
            probs[np.arange(len(preds_fallback)), class_indices] = 1.0

    if preds is None:
        # 2026-04-24 Session 6: multilabel target -> (N, K) probs, threshold
        # each column independently; do NOT argmax (collapses to single class).
        # 2026-04-28: also treat object-dtype-of-arrays as 2-D (the
        # ``pl.List`` -> pandas roundtrip). Without this, preds was
        # computed via argmax (1-D class index) while targets stayed
        # multilabel-indicator (2-D), and ``classification_report`` raised
        # ``mix of multilabel-indicator and multiclass targets``.
        _targets_2d = (
            isinstance(targets, np.ndarray) and targets.ndim == 2
        ) or (
            isinstance(targets, pd.DataFrame)
        ) or (
            isinstance(targets, np.ndarray)
            and targets.dtype == object
            and targets.ndim == 1
            and targets.shape[0] > 0
            and (hasattr(targets[0], "shape") or (
                hasattr(targets[0], "__len__")
                and not isinstance(targets[0], (str, bytes))
            ))
        )
        if _targets_2d:
            # MultiOutputClassifier returns list[(N,2)] for predict_proba -- canonicalize to (N, K).
            from .helpers import _canonical_predict_proba_shape, _predict_from_probs
            from .configs import TargetTypes as _TT
            probs = _canonical_predict_proba_shape(probs)
            # Honour MultilabelDispatchConfig.per_label_thresholds when
            # supplied: per-column decision threshold tuned for label
            # imbalance (defaults to 0.5 across all labels otherwise).
            # ``_predict_from_probs`` already broadcasts a scalar 0.5 vs
            # accepts a (K,) vector -- same downstream shape (N, K).
            _per_label_thr = (
                multilabel_dispatch_config.per_label_thresholds
                if (multilabel_dispatch_config is not None
                    and multilabel_dispatch_config.per_label_thresholds is not None)
                else 0.5
            )
            preds = _predict_from_probs(
                probs, _TT.MULTILABEL_CLASSIFICATION, threshold=_per_label_thr,
            )
        elif probs.shape[1] == 2:
            # For binary classification, use threshold=0.5 on class 1 probability
            # This ensures consistency with calibration metrics in fast_calibration_report
            classes_ = model.classes_ if (model is not None and hasattr(model, "classes_")) else np.array([0, 1])
            preds = np.where(probs[:, 1] >= 0.5, classes_[1], classes_[0])
        else:
            preds = np.argmax(probs, axis=1)
            if model is not None and hasattr(model, "classes_"):
                preds = model.classes_[preds]

    if isinstance(targets, pd.Series):
        targets = targets.values

    brs = []
    calibs = []
    pr_aucs = []
    roc_aucs = []
    integral_errors = []
    log_losses = []
    robust_integral_errors = []

    # 2026-04-24 Session 6: detect multilabel from 2-D target shape. Each
    # column is an independent binary label; the per-class loop below uses
    # the column directly instead of `targets == class_name` (which would
    # broadcast a 2-D bool against a 1-D y_score and crash).
    # 2026-04-28: also detect object-dtype-of-arrays (the polars
    # ``pl.List(pl.Int8)`` -> pandas object roundtrip), stack to 2-D so
    # ``targets_arr[:, class_id]`` works in the multilabel branch.
    # Surfaced 3-way fuzz c0000 / c0008 (cb / multilabel target).
    # 2026-05-10: extracted to ``_canonical_multilabel_y`` helper so the
    # new ``mlframe.training.dummy_baselines`` module can reuse the same
    # canonicalization logic without duplication.
    targets_arr = _canonical_multilabel_y(targets)
    targets = targets_arr  # rebind so downstream uses the stacked form
    is_multilabel = targets_arr.ndim == 2

    integral_error = custom_ice_metric(y_true=targets, y_score=probs) if custom_ice_metric else 0.0
    robust_integral_error = None
    if custom_rice_metric and custom_rice_metric != custom_ice_metric:
        robust_integral_error = custom_rice_metric(y_true=targets, y_score=probs)

    if not classes:
        if is_multilabel:
            # K independent labels named 0..K-1
            classes = list(range(targets_arr.shape[1]))
        elif model is not None:
            if hasattr(model, "classes_"):
                classes = model.classes_
            else:
                classes = np.unique(targets)
        elif target_label_encoder:
            classes = np.arange(len(target_label_encoder.classes_)).tolist()
        else:
            classes = np.unique(targets)

    # GPU batch-AUC fastpath: when the suite has many classes (multiclass /
    # multilabel) and the row count is large enough, compute all K
    # (roc_auc, pr_auc) pairs in ONE batched GPU call instead of K serial
    # ``fast_aucs_per_group_optimized`` calls inside the per-class loop.
    # Auto-dispatched by ``compute_batch_aucs``: GPU when cupy + CUDA
    # visible AND N >= threshold, otherwise CPU loop (no behavior change).
    # Only valid when group_ids is None (per-group AUCs need the full
    # function path). Empirical wins documented in ``bench_gpu_metrics.py``:
    # at N=1M K=20 PR AUC alone, GPU = 170 ms vs CPU loop = 2016 ms.
    _precomputed_aucs_per_class: Optional[List[Optional[Tuple[float, float]]]] = None
    if group_ids is None and len(classes) >= 2:
        try:
            from mlframe.metrics import compute_batch_aucs
            # Build (N, K) score matrix and (N, K)/(N,) label matrix once.
            if is_multilabel:
                _y_true_NK = targets_arr  # already (N, K) binary
                _y_score_NK = probs       # (N, K)
            elif len(classes) == 2:
                # Binary: only class_id=1 is reported (loop skips id=0).
                # Single column, no batching benefit, but the dispatcher
                # auto-falls-back to CPU at small M anyway.
                _y_true_NK = (targets == classes[1]).astype(np.int8)[:, None]
                _y_score_NK = probs[:, [1]]
            else:
                # Multiclass: K columns, one-vs-rest.
                _y_true_NK = np.column_stack(
                    [(targets == c).astype(np.int8) for c in classes]
                )
                _y_score_NK = probs
            roc_batch, pr_batch = compute_batch_aucs(_y_true_NK, _y_score_NK)
            _precomputed_aucs_per_class = [
                (float(roc_batch[j]), float(pr_batch[j])) for j in range(_y_score_NK.shape[1])
            ]
        except Exception as e:
            # Any failure -> fall back to per-class fast_aucs path.
            logger.debug("compute_batch_aucs precompute failed (%s); using per-class path.", e)
            _precomputed_aucs_per_class = None

    true_classes = []
    for class_id, class_name in enumerate(classes):
        if str(class_name).isnumeric() and target_label_encoder:
            str_class_name = str(target_label_encoder.classes_[class_name])
        else:
            str_class_name = str(class_name)
        true_classes.append(str_class_name)

        # Multilabel: never skip class_id=0; every column is an independent label.
        if not is_multilabel and len(classes) == 2 and class_id == 0:
            continue

        if is_multilabel:
            y_true = targets_arr[:, class_id]
        else:
            y_true = (targets == class_name)
        y_score = probs[:, class_id]
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()

        title = report_title + " " + model_name
        if len(classes) != 2:
            title += "-" + str_class_name

        class_integral_error = custom_ice_metric(y_true=y_true, y_score=y_score) if custom_ice_metric else 0.0
        n_cols = n_features if n_features is not None else (len(columns) if columns else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        title += f" [{nfeatures}{get_human_readable_set_size(len(y_true))} rows]"
        if custom_rice_metric and custom_rice_metric != custom_ice_metric:
            class_robust_integral_error = custom_rice_metric(y_true=y_true, y_score=y_score)
            title += f", RICE={class_robust_integral_error:.{calib_report_ndigits}f}"

        # Build kwargs for fast_calibration_report. title_metrics_tokens is the
        # post-validation tuple from ReportingConfig - if None, the function's
        # own DEFAULT_TITLE_METRICS_TOKENS applies.
        _fcr_kwargs = dict(
            y_true=y_true,
            y_pred=y_score,
            use_weights=use_weights,
            nbins=nbins,
            group_ids=group_ids,
            title=title,
            figsize=figsize,
            # NOTE: plot_file and show_perf_chart are intentionally independent.
            # `plot_file` (derived from `data_dir`) controls whether plots are SAVED
            # to disk. `show_perf_chart` controls only interactive DISPLAY (plt.show).
            # Saving plots even when show_perf_chart=False is deliberate - users get
            # artifacts on disk without GUI popups. The Agg save-only fastpath in
            # show_calibration_plot handles this case without Qt overhead.
            plot_file=plot_file + "_perfplot.png" if plot_file else "",
            show_plots=show_perf_chart,
            ndigits=calib_report_ndigits,
            verbose=verbose,
            show_prob_histogram=show_prob_histogram,
            prob_histogram_yscale=prob_histogram_yscale,
            show_inline_population_labels=show_inline_population_labels,
            dpi=plot_dpi,
        )
        if title_metrics_tokens is not None:
            _fcr_kwargs["title_metrics_tokens"] = title_metrics_tokens

        # Inject precomputed (roc, pr) for THIS class id when the batched
        # GPU/CPU fastpath ran above. fast_calibration_report skips its
        # internal ``fast_aucs_per_group_optimized`` call when this is set.
        if _precomputed_aucs_per_class is not None:
            # Index alignment:
            #  - multilabel: matrix has K columns, class_id 0..K-1
            #  - binary: matrix has 1 column; we get here only for class_id=1
            #  - multiclass: matrix has K columns, class_id 0..K-1
            if not is_multilabel and len(classes) == 2:
                # Single-column matrix indexed at 0
                _fcr_kwargs["_precomputed_aucs"] = _precomputed_aucs_per_class[0]
            else:
                _fcr_kwargs["_precomputed_aucs"] = _precomputed_aucs_per_class[class_id]

        with phase("fast_calibration_report", class_id=str_class_name, n_rows=len(y_true)):
            (
                brier_loss, calibration_mae, calibration_std, calibration_coverage,
                ece, brier_reliability, brier_resolution, brier_uncertainty,
                roc_auc, pr_auc, ice, ll, precision, recall, f1,
                metrics_string, _fig,
            ) = fast_calibration_report(**_fcr_kwargs)

        if print_report:
            calibs.append(
                f"\t{str_class_name}: MAE{'W' if use_weights else ''}={calibration_mae * 100:.{calib_report_ndigits}f}%, STD={calibration_std * 100:.{calib_report_ndigits}f}%, COV={calibration_coverage * 100:.0f}%"
            )
            pr_aucs.append(f"{str_class_name}={'N/A' if np.isnan(pr_auc) else f'{pr_auc:.{report_ndigits}f}'}")
            roc_aucs.append(f"{str_class_name}={'N/A' if np.isnan(roc_auc) else f'{roc_auc:.{report_ndigits}f}'}")
            brs.append(f"{str_class_name}={brier_loss * 100:.{report_ndigits}f}%")
            integral_errors.append(f"{str_class_name}={ice:.{report_ndigits}f}")
            if ll is None:
                log_losses.append(f"{str_class_name}=None")
            else:
                log_losses.append(f"{str_class_name}={ll:.{report_ndigits}f}")
            if custom_rice_metric and custom_rice_metric != custom_ice_metric:
                robust_integral_errors.append(f"{str_class_name}={class_robust_integral_error:.{report_ndigits}f}")

        if metrics is not None:
            class_metrics = dict(
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                calibration_mae=calibration_mae,
                calibration_std=calibration_std,
                brier_loss=brier_loss,
                ece=ece,
                brier_reliability=brier_reliability,
                brier_resolution=brier_resolution,
                brier_uncertainty=brier_uncertainty,
                log_loss=ll,
                ice=ice,
                class_integral_error=class_integral_error,
                precision=precision,
                recall=recall,
                f1=f1,
            )
            if custom_rice_metric and custom_rice_metric != custom_ice_metric:
                class_metrics["class_robust_integral_error"] = class_robust_integral_error
            metrics.update({class_id: class_metrics})

    if print_report:
        # 2026-05-10: route through logger so file handlers (e.g.
        # pyutilz.logginglib.init_logging) capture the report block.
        # See sibling fix in report_regression_model_perf at line 659.
        # 2026-04-29: replace sklearn's ``classification_report`` with the
        # njit-backed ``format_classification_report``. cProfile of fuzz
        # c0014 traced 90ms (55 %) of the warm-path
        # ``report_probabilistic_model_perf`` to sklearn's
        # ``precision_recall_fscore_support`` + ``multilabel_confusion_matrix``
        # path, which is overkill for single-label classification. The
        # njit version computes the same numbers in ~1ms warm and formats
        # to the identical text shape.
        _cls_report_text = ""
        try:
            from mlframe.metrics import format_classification_report
            _y_true = np.asarray(targets).astype(np.int64) if not is_multilabel else None
            _y_pred = np.asarray(preds).astype(np.int64) if not is_multilabel else None
            if (
                _y_true is not None and _y_pred is not None
                and _y_true.ndim == 1 and _y_pred.ndim == 1
                and len(_y_true) == len(_y_pred)
            ):
                _nclasses = max(int(_y_true.max()) + 1, int(_y_pred.max()) + 1, 2) if len(_y_true) else 2
                _cls_report_text = format_classification_report(
                    _y_true, _y_pred, nclasses=_nclasses, digits=report_ndigits, zero_division=0,
                )
            else:
                _cls_report_text = classification_report(targets, preds, zero_division=0, digits=report_ndigits)
        except Exception:
            _cls_report_text = classification_report(targets, preds, zero_division=0, digits=report_ndigits)
        _report_lines = [
            report_title + " " + model_name,
            _cls_report_text,
            f"ROC AUCs: {', '.join(roc_aucs)}",
            f"PR AUCs: {', '.join(pr_aucs)}",
            f"CALIBRATIONs: \n{', '.join(calibs)}",
            f"BRIER LOSSes: \n\t{', '.join(brs)}",
            f"LOG_LOSSes: \n\t{', '.join(log_losses)}",
            f"ICEs: \n\t{', '.join(integral_errors)}",
        ]
        if custom_ice_metric != custom_rice_metric:
            _report_lines.append(f"RICEs: \n\t{', '.join(robust_integral_errors)}")
        logger.info("\n".join(_report_lines))

        logger.info(f"TOTAL INTEGRAL ERROR: {integral_error:.4f}")
        if robust_integral_error is not None:
            logger.info(f"TOTAL ROBUST INTEGRAL ERROR: {robust_integral_error:.4f}")

        # 2026-04-24 Session 4: pluggable multi-output metrics registry.
        # Dispatches hamming_loss / subset_accuracy / jaccard_score_multilabel
        # (registered in mlframe.training.metrics_registry) when the
        # report-caller context indicates a multilabel target. Additional
        # metrics can be registered externally via
        # ``register_metric(target_type, name, fn)`` -- no code change to
        # this report function required.
        try:
            from .metrics_registry import iter_extra_metrics
            # Heuristic inference: multilabel if targets is 2-D binary.
            if hasattr(targets, "ndim") and targets.ndim == 2:
                from .configs import TargetTypes
                extra = list(iter_extra_metrics(
                    TargetTypes.MULTILABEL_CLASSIFICATION, targets, probs, preds,
                ))
                if extra:
                    _ml_lines = ["MULTILABEL METRICS:"]
                    for name, val in extra:
                        try:
                            _ml_lines.append(f"\t{name}={val:.{report_ndigits}f}")
                        except Exception:
                            _ml_lines.append(f"\t{name}={val}")
                    logger.info("\n".join(_ml_lines))
        except Exception as e:
            # Never fail a report because of metrics-registry plumbing.
            logger.debug("multilabel metrics registry skipped: %s", e)

    if subgroups:
        subgroups_metrics = {"ICE": custom_ice_metric}
        metrics_higher_is_better = {"ICE": False}

        if probs.shape[1] == 2:
            subgroups_metrics["ROC AUC"] = fast_roc_auc
            metrics_higher_is_better["ROC AUC"] = True

        with phase("compute_fairness_metrics"):
            fairness_report = compute_fairness_metrics(
                subgroups=subgroups,
                subset_index=subset_index,
                y_true=targets,
                y_pred=probs,
                metrics=subgroups_metrics,
                metrics_higher_is_better=metrics_higher_is_better,
            )
        if fairness_report is not None:
            if print_report:
                display(fairness_report.style.set_caption("ML perf fairness by group"))
            if metrics is not None:
                metrics.update(dict(fairness_report=fairness_report))

    return preds, probs


