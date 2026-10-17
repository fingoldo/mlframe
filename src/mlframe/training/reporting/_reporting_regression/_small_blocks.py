"""Small verbatim-carved blocks of ``report_regression_model_perf`` (residual audit, interactive guard, fairness, point metrics)."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.metrics.core import (
    compute_fairness_metrics,
    fast_max_error,
    fast_mean_absolute_error,
    fast_r2_score,
    fast_root_mean_squared_error,
)

from .._reporting_common import _maybe_display

logger = logging.getLogger("mlframe.training.reporting._reporting_regression")


def _report_regression__targets_arr_ndim_targets(targets_arr, preds_arr, targets, preds, metrics, model_name, _residual_audit):
    """Block of report_regression_model_perf starting at ``if not ((targets_arr.ndim > 1 and targets_arr.shape[1] > 1) or (preds_``."""
    if not ((targets_arr.ndim > 1 and targets_arr.shape[1] > 1) or (preds_arr.ndim > 1 and preds_arr.shape[1] > 1)):
        try:
            from mlframe.training.targets import audit_residuals as _audit_residuals_fn

            _residual_audit = _audit_residuals_fn(targets, preds)
            if metrics is not None:
                metrics["residual_audit"] = _residual_audit.to_dict()
        except Exception as _audit_err:
            logger.warning(
                "residual_audit failed for '%s': %s. Continuing without diagnostics.",
                model_name,
                _audit_err,
            )
    return _residual_audit


def _report_regression__warm_state_regression_report(show_perf_chart, plot_file):
    """Block of report_regression_model_perf starting at ``if show_perf_chart and not plot_file:``."""
    if show_perf_chart and not plot_file:
        try:
            _is_interactive_session = bool(__IPYTHON__)  # type: ignore[name-defined]
        except NameError:
            import sys as _sys

            _is_interactive_session = hasattr(_sys, "ps1")
        if not _is_interactive_session:
            show_perf_chart = False  # disable for the guards below
    return show_perf_chart


def _report_regression__try(reporting_config, _reg_tokens):
    """Block of report_regression_model_perf starting at ``try:``."""
    try:
        _cfg_tokens = getattr(reporting_config, "regression_title_metrics_tokens", None)
        if _cfg_tokens:
            _reg_tokens = tuple(_cfg_tokens)
    except (AttributeError, TypeError):
        # reporting_config is now a real parameter (threaded through
        # _trainer_train_and_evaluate -> _compute_split_metrics ->
        # report_model_perf -> here). It may still be None when a caller
        # constructs the regression reporter directly without the full
        # suite context; the try/except keeps that path harmless.
        pass
    return _reg_tokens


def _report_regression__cleanly_under_one_record(subgroups, subset_index, targets, preds, print_report, metrics):
    """Block of report_regression_model_perf starting at ``if subgroups:``."""
    if subgroups:
        fairness_report = compute_fairness_metrics(
            subgroups=subgroups,
            subset_index=subset_index,  # compute_fairness_metrics handles None internally (see its subset_index is None branch)
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


def _report_regression_point_metrics(st, sample_weight):
    """Headline and extended regression metrics for the current (targets_arr, preds_arr) pair.

    Uses the fused 1-D kernel when there is no sample weight, otherwise the per-metric weighted kernels.
    Returns (MAE, RMSE, MaxError, R2, MBE, MAPE_mean, SMAPE, wMAPE, CV_RMSE, NSE, Pearson, ExplainedVariance).
    """
    if st.targets_arr.ndim == 1 and st.preds_arr.ndim == 1 and sample_weight is None:
        # audit batch: switched from fast_regression_metrics_block
        # (4 metrics: MAE/RMSE/MaxError/R2) to the EXTENDED block that lands
        # 12 metrics in the SAME 2 numba passes. 5.8-10.5x faster than the
        # equivalent 12 separate calls; see comment block in
        # ``mlframe.metrics._regression_extras`` for measured speedups.
        # Only reachable when sample_weight is None: the fused kernel has no weighted variant.
        from mlframe.metrics.core import fast_regression_metrics_block_extended

        _block = fast_regression_metrics_block_extended(st.targets_arr, st.preds_arr)
        MAE = _block["MAE"]
        RMSE = _block["RMSE"]
        MaxError = _block["MaxError"]
        R2 = _block["R2"]
        # Stash the rest on the locals so the chart-title token renderer
        # below can pick them up without recomputing.
        _ext_MBE = _block["MBE"]
        _ext_MAPE_mean = _block["MAPE_mean"]
        _ext_SMAPE = _block["SMAPE"]
        _ext_wMAPE = _block["wMAPE"]
        _ext_CV_RMSE = _block["CV_RMSE"]
        _ext_NSE = _block["NSE"]
        _ext_Pearson = _block["Pearson"]
        _ext_EV = _block["ExplainedVariance"]
    else:
        # Either a 2-D/multioutput target, OR sample_weight was supplied (the fused block has no
        # weighted variant). MAE/RMSE/R2 correctly honor sample_weight here via their own
        # fast_*(sample_weight=...) kernels; the 8 "extended" metrics have no weighted kernel
        # anywhere in the codebase and stay unweighted (NaN placeholders, matching the pre-existing
        # 2-D-path behavior) rather than silently mixing weighted and unweighted numbers in one report.
        _ext_MBE = _ext_MAPE_mean = _ext_SMAPE = _ext_wMAPE = np.nan
        _ext_CV_RMSE = _ext_NSE = _ext_Pearson = _ext_EV = np.nan
        MAE = fast_mean_absolute_error(st.targets_arr, st.preds_arr, sample_weight=sample_weight)
        # 2-D max_error returns per-output array; the existing reporting
        # contract is a single scalar (overall max), so reduce explicitly.
        # No weighted variant exists (matches sklearn's own unweighted max_error).
        _max_err = fast_max_error(st.targets_arr, st.preds_arr)
        MaxError = float(np.max(_max_err)) if isinstance(_max_err, np.ndarray) else _max_err
        R2 = fast_r2_score(st.targets_arr, st.preds_arr, sample_weight=sample_weight)
        RMSE = fast_root_mean_squared_error(st.targets_arr, st.preds_arr, sample_weight=sample_weight)
    return (MAE, RMSE, MaxError, R2, _ext_MBE, _ext_MAPE_mean, _ext_SMAPE, _ext_wMAPE, _ext_CV_RMSE, _ext_NSE, _ext_Pearson, _ext_EV)
