"""Stages of ``report_regression_model_perf``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import pandas as pd
from mlframe.training.reporting._reporting_regression import _report_regression_mo_step1_st_targets_arr
from mlframe.training.reporting._reporting_regression import logger
from mlframe.training.reporting._reporting_regression import select_quantile_point_estimate
from mlframe.training.reporting._reporting_regression._small_blocks import _report_regression__targets_arr_ndim_targets
from mlframe.training.reporting._reporting_regression._small_blocks import _report_regression_point_metrics
# --- end imports ---


def _report_regression_metric_stage(st, mase_naive_mae, model_name, MAE, MaxError, R2, RMSE, _ext_MBE, _ext_MAPE_mean, _ext_SMAPE, _ext_wMAPE, _ext_CV_RMSE, _ext_NSE, _ext_Pearson, _ext_EV, mase_seasonality, metrics, targets, preds):
    """Stage of ``report_regression_model_perf``."""
    _ext_RMSLE = np.nan
    _ext_MdAPE = np.nan
    _ext_Spearman = np.nan
    _ext_Kendall = np.nan
    _ext_Cindex = np.nan
    _ext_Huber = np.nan
    st._ext_MASE = np.nan
    st._ext_RMSPE = np.nan
    st._ext_LogCosh = np.nan
    _ext_Cindex, _ext_Huber, _ext_Kendall, _ext_MdAPE, _ext_RMSLE, _ext_Spearman = _report_regression_mo_step1_st_targets_arr(
        st, mase_naive_mae, model_name, _ext_Cindex, _ext_Huber, _ext_Kendall, _ext_MdAPE, _ext_RMSLE, _ext_Spearman
    )

    st.current_metrics = dict(
        MAE=MAE,
        MaxError=MaxError,
        R2=R2,
        RMSE=RMSE,
        # audit: 8 extras lifted from the fused extended block.
        # NSE numerically equals R2 (sklearn convention) but stamped
        # separately because hydrology / climate code expects the name.
        MBE=_ext_MBE,
        MAPE_mean=_ext_MAPE_mean,
        SMAPE=_ext_SMAPE,
        wMAPE=_ext_wMAPE,
        CV_RMSE=_ext_CV_RMSE,
        NSE=_ext_NSE,
        Pearson=_ext_Pearson,
        ExplainedVariance=_ext_EV,
        # Sort-based / log-based / parameterised metrics (separate kernels).
        RMSLE=_ext_RMSLE,
        MdAPE=_ext_MdAPE,
        Spearman=_ext_Spearman,
        Kendall=_ext_Kendall,
        ConcordanceIndex=_ext_Cindex,
        Huber=_ext_Huber,
        MASE=st._ext_MASE,
        MASE_seasonality=int(mase_seasonality) if mase_seasonality is not None else None,
        RMSPE=st._ext_RMSPE,
        LogCosh=st._ext_LogCosh,
    )
    if metrics is not None:
        metrics.update(st.current_metrics)

    # Compute residual audit ONCE (used by both the chart and the print-report block; cheap thanks to internal sampling).
    # ``behavior_config.report_residual_audit`` is a LOG-ONLY toggle. When False we MUST still compute the audit so the chart's hist + resid-vs-pred panels stay populated -- only the multi-line verdict text in the log is suppressed. Multi-output targets still skip the audit (no scalar residuals to fit a distribution to).
    st._residual_audit = None
    from mlframe.training.evaluation import _get_residual_audit_enabled  # lazy: breaks cycle with .evaluation

    st._audit_log_enabled = bool(_get_residual_audit_enabled())
    st._residual_audit = _report_regression__targets_arr_ndim_targets(st.targets_arr, st.preds_arr, targets, preds, metrics, model_name, st._residual_audit)
    return _ext_Cindex, _ext_Huber, _ext_Kendall, _ext_MdAPE, _ext_RMSLE, _ext_Spearman


def _report_regression_point_and_sensor_stage(st, sample_weight, model_name, y_train_min, y_train_max, y_train_std, MAE, MaxError, R2, RMSE):
    """Stage of ``report_regression_model_perf``."""
    MAE, RMSE, MaxError, R2, _ext_MBE, _ext_MAPE_mean, _ext_SMAPE, _ext_wMAPE, _ext_CV_RMSE, _ext_NSE, _ext_Pearson, _ext_EV = _report_regression_point_metrics(
        st, sample_weight
    )

    # Prediction-collapse / extrapolation / mean-shift / out-of-envelope sensor (carved to _sensors.py). Logs a HARD
    # WARNING when predictions look pathological so operators see it in the run log instead of eyeballing scatters.
    from mlframe.training.reporting._reporting_regression._sensors import run_collapse_sensor

    run_collapse_sensor(
        st.preds_arr,
        st.targets_arr,
        R2,
        model_name=model_name,
        y_train_min=y_train_min,
        y_train_max=y_train_max,
        y_train_std=y_train_std,
    )
    return MAE, MaxError, R2, RMSE, _ext_CV_RMSE, _ext_EV, _ext_MAPE_mean, _ext_MBE, _ext_NSE, _ext_Pearson, _ext_SMAPE, _ext_wMAPE


def _report_regression_clip_stage(st, y_train_min, y_train_max, y_train_std, model_name, report_title, preds):
    """Stage of ``report_regression_model_perf``."""
    # eval-fallback) target range BEFORE metrics + chart; catches catastrophic extrapolation on group-aware splits.
    from mlframe.training.reporting._reporting_regression._sensors import apply_prediction_envelope_clip

    st.preds_arr = apply_prediction_envelope_clip(
        st.preds_arr,
        st.targets_arr,
        y_train_min=y_train_min,
        y_train_max=y_train_max,
        y_train_std=y_train_std,
        model_name=model_name,
        report_title=report_title,
    )
    # Single source of truth from here on: the clip is documented as landing "BEFORE metrics + chart", but
    # only preds_arr (feeding the numeric metrics) was reassigned -- the residual audit, the DSL chart
    # path, the legacy matplotlib chart, and this function's own return value all kept reading the
    # ORIGINAL, unclipped `preds`. During the exact catastrophic-extrapolation scenario the clip exists to
    # guard against, the title/metrics would show a plausible clipped R2 while the chart/audit/returned
    # predictions (later stored as entry.test_preds and consumed throughout the rest of the reporting
    # pipeline) silently showed the raw, unclipped values.
    preds = st.preds_arr
    if st.targets_arr.ndim > 1 and st.targets_arr.shape[1] > 1:
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
            st.targets_arr.shape,
        )
    return preds


def _report_regression_quantile_stage(preds, model, df, targets, st, quantile_alphas):
    """Stage of ``report_regression_model_perf``."""
    if preds is None:
        # Wrap in _predict_with_fallback so CatBoost's Polars fastpath
        # dispatcher misses ("No matching signature found") trigger a
        # symmetric pandas fallback -- same pattern as fit. Without this
        # wrap, a polars val/test DF + a CB model whose fit path fell
        # back to pandas would crash at predict time with the identical
        # opaque TypeError (prod incident).
        from mlframe.training.trainer import _predict_with_fallback

        preds = np.asarray(_predict_with_fallback(model, df, method="predict"))

    if isinstance(targets, pd.Series):
        targets = targets.values

    # The underlying mlframe.metrics.regression fast_* kernels cover the full sklearn signature
    # (1-D / 2-D, sample_weight, multioutput), but the FUSED block this call site uses on the
    # common 1-D path (fast_regression_metrics_block_extended, below) does not thread
    # sample_weight through -- see that branch for which metrics are actually weight-aware here.
    st.targets_arr = np.asarray(targets)
    st.preds_arr = np.asarray(preds)
    # Quantile regression: a single-column target scored against (N, K) per-alpha predictions. The headline
    # point metrics and the 1-D chart/audit/sensor stages need one point estimate per row, which is the median.
    # The full (N, K) matrix is still what this function returns, for the quantile panels and stored predictions.
    st.full_quantile_preds = None
    _point = select_quantile_point_estimate(st.targets_arr, st.preds_arr, quantile_alphas)
    if _point is not st.preds_arr:
        st.full_quantile_preds = st.preds_arr
        st.preds_arr = _point
        preds = _point
    return preds, targets
