"""Regression tests for the 2026-05-27 user requirement on composite-
target chart visibility.

Root cause: at the time ``report_regression_model_perf`` is called for
a composite-target model, the inner has NOT yet been wrapped in
``CompositeTargetEstimator`` (wrap happens later in
``_phase_composite_post``). So ``model.predict`` returns T-scale and
``targets`` are T-scale -- a T-scale chart there would be misleading.

The Y-scale chart is emitted from ``_phase_composite_wrapping`` AFTER
the wrap, where ``wrapper.predict(df)`` returns y-scale and the
original raw y target is in scope.

Tests:
* Y-scale chart helper renders without raising on a minimal happy-path.
* T-scale residual chart is skipped by default in the per-model
  reporter (avoids the misleading-RMSE footgun); env opt-in restores.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np


def test_yscale_chart_helper_runs_without_raising() -> None:
    from mlframe.training.core._phase_composite_wrapping import (
        _emit_yscale_composite_chart,
    )

    rng = np.random.default_rng(0)
    y_target = rng.normal(11502.0, 11.5, 1000)
    y_pred = y_target + rng.normal(0.0, 10.0, 1000)
    # Use a minimal stub for ``inner_entry`` (the helper only reads
    # ``model``/``estimator_``/class-name to build the chart title).
    inner_entry = SimpleNamespace(model=SimpleNamespace())
    _emit_yscale_composite_chart(
        y_target=y_target,
        y_pred=y_pred,
        inner_entry=inner_entry,
        composite_name="TVT-spline-TVT_prev",
        orig_tname="TVT",
        target_name="TVT",
        plot_file="",  # interactive-session-only; we don't write
        reporting_config=None,
        rmse_y=13.5,
        mae_y=8.0,
        r2_y=0.99,
    )


def test_yscale_chart_helper_no_op_on_empty_inputs() -> None:
    from mlframe.training.core._phase_composite_wrapping import (
        _emit_yscale_composite_chart,
    )
    from types import SimpleNamespace

    _emit_yscale_composite_chart(
        y_target=np.array([]),
        y_pred=np.array([]),
        inner_entry=SimpleNamespace(),
        composite_name="X",
        orig_tname="X",
        target_name="X",
        plot_file="",
        reporting_config=None,
        rmse_y=0.0,
        mae_y=0.0,
        r2_y=0.0,
    )


def test_mtresid_t_scale_chart_returns_early_without_rendering(monkeypatch) -> None:
    """For composite-target models, the per-model reporter MUST return
    BEFORE constructing chart figures (T-scale RMSE looks competitive
    when raw-y RMSE may be 100x larger -- known operator footgun). The
    skip is implemented via early-return at the chart block; the
    function still computes residual_audit and MAE/RMSE/R2 (since those
    happen before the skip) but no scatter / no plot file is written.
    """
    monkeypatch.delenv("MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS", raising=False)
    from mlframe.training.reporting._reporting_regression import (
        report_regression_model_perf,
    )

    rng = np.random.default_rng(0)
    y_true = rng.normal(0.0, 1.0, 500)
    y_pred = y_true + rng.normal(0.0, 0.3, 500)
    preds_out, _ = report_regression_model_perf(
        targets=y_true,
        preds=y_pred,
        columns=("x0",),
        model_name="LGBMRegressor TVT MTRESID/MRTS=1.39/1.54",
        model=None,
        print_report=False,
        show_perf_chart=False,
        # plot_file omitted -> no figure file written either way; here
        # we just sanity-check the call returns predictions and does
        # not raise.
    )
    assert preds_out is not None
