"""Multi-target-regression (MTR) report block.

Carved out of ``report_regression_model_perf`` to keep the package ``__init__`` below the 1k-line monolith threshold.
Handles (N, K) targets/preds: aggregated metrics into the metrics dict + one per-target chart file per column. The
1-D-only paths (residual audit, fairness subgroups, MASE, prediction-envelope clip) stay skipped here by design.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def render_mtr_report(
    targets_arr: np.ndarray,
    preds_arr: np.ndarray,
    model_name: str,
    *,
    metrics: dict[str, Any] | None,
    print_report: bool,
    plot_outputs: str | None,
    plot_file: str,
    figsize: tuple[int, int],
    plot_sample_size: int,
    plot_dpi: int | None,
    report_title: str,
    verbose: bool,
) -> tuple[np.ndarray, None]:
    """Aggregate MTR metrics + render per-target charts. Returns ``(preds_arr, None)`` (regression has no probabilities)."""
    from ...configs import TargetTypes
    from ...metrics_registry import iter_extra_metrics
    _mtr_extra = dict(iter_extra_metrics(
        TargetTypes.MULTI_TARGET_REGRESSION,
        targets_arr, None, preds_arr,
    ))
    if metrics is None:
        metrics = {}
    metrics.update(_mtr_extra)
    if print_report:
        try:
            _msg_lines = [
                f"MULTI_TARGET_REGRESSION [{model_name}] " f"shape=(N={targets_arr.shape[0]}, K={targets_arr.shape[1]}):",
            ]
            for _k, _v in _mtr_extra.items():
                _msg_lines.append(f"  {_k} = {_v:+.4f}")
            logger.info("\n".join(_msg_lines))
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _mtr.py:51: %s", e)
            pass
    # Per-target K-grid charts: one chart file per target column at ``{plot_file_base}_target{k}{ext}``, through the
    # same build_regression_panel_spec -> render_and_save pipeline as the single-target report.
    if plot_outputs and plot_file:
        try:
            from mlframe.metrics.core import (
                fast_mean_absolute_error as _mae_for_chart,
                fast_root_mean_squared_error as _rmse_for_chart,
                fast_r2_score as _r2_for_chart,
            )
            from mlframe.reporting.charts.regression import build_regression_panel_spec
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save
            from ...targets import audit_residuals
            _render_config = parse_plot_output_dsl(plot_outputs)
            # render_and_save appends the format extension; strip a trailing image extension the caller may have passed
            # so we don't produce "file_target0.png.png".
            _base = plot_file
            for _ext_strip in (".png", ".pdf", ".svg", ".jpg", ".jpeg", ".html"):
                if _base.lower().endswith(_ext_strip):
                    _base = _base[: -len(_ext_strip)]
                    break
            _K = int(targets_arr.shape[1])
            for _k_idx in range(_K):
                _yt_k = targets_arr[:, _k_idx].astype(np.float64).ravel()
                _yp_k = preds_arr[:, _k_idx].astype(np.float64).ravel()
                _mask_k = np.isfinite(_yt_k) & np.isfinite(_yp_k)
                if int(_mask_k.sum()) < 5:
                    logger.warning(
                        "MTR per-target chart: target %d has <5 finite " "(true, pred) pairs; skipping chart for this column.",
                        _k_idx,
                    )
                    continue
                _audit_k = audit_residuals(_yt_k[_mask_k], _yp_k[_mask_k], seed=42)
                _mae_k = float(_mae_for_chart(_yt_k[_mask_k], _yp_k[_mask_k]))
                _rmse_k = float(_rmse_for_chart(_yt_k[_mask_k], _yp_k[_mask_k]))
                _r2_k = float(_r2_for_chart(_yt_k[_mask_k], _yp_k[_mask_k]))
                _metrics_str = f"target {_k_idx}: R^2={_r2_k:+.4f} " f"RMSE={_rmse_k:.4f} MAE={_mae_k:.4f}"
                _spec = build_regression_panel_spec(
                    _yt_k, _yp_k, audit=_audit_k,
                    header_str=f"{report_title or model_name} target {_k_idx}",
                    metrics_str=_metrics_str,
                    figsize=figsize, plot_sample_size=plot_sample_size,
                )
                if plot_dpi is not None:
                    import dataclasses as _dc
                    _spec = _dc.replace(_spec, dpi=plot_dpi)
                _per_target_path = f"{_base}_target{_k_idx}"
                render_and_save(_spec, _render_config, _per_target_path)
            logger.info(
                "MTR per-target charts: rendered %d chart base paths at "
                "%s_target0 ... %s_target%d (renderer appends format ext).",
                _K, _base, _base, _K - 1,
            )
        except Exception as _chart_err:
            logger.warning(
                "MTR per-target chart generation failed (%s); " "aggregated metrics still stamped into metrics dict.",
                _chart_err,
            )
    if verbose:
        logger.warning(
            "MULTI_TARGET_REGRESSION report path: residual audit + "
            "fairness subgroup + MASE skipped (1-D-only). "
            "Aggregated metrics in metrics dict; per-target charts "
            "rendered to {plot_file}_target{k}{ext} when plot_outputs "
            "+ plot_file are set."
        )
    return preds_arr, None


__all__ = ["render_mtr_report"]
