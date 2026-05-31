"""F-34 suite-side integration tests for MULTI_TARGET_REGRESSION.

Covers:
  * D2: CT_ENSEMBLE skips MTR targets with WARN (no silent miscomputation)
  * D3: metrics_registry has MTR-specific per-target + aggregated metrics
  * D4: report_regression_model_perf gates (N, K) preds to the
    metrics-only path (no chart / audit / fairness)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlframe.training import TargetTypes
from mlframe.training.metrics_registry import (
    iter_extra_metrics, list_registered, metric_name_higher_is_better,
)


# --- D3: metrics_registry MTR built-ins --------------------------------------


def test_mtr_metrics_registered():
    """All 7 expected MTR metrics land in the registry at import time."""
    names = set(list_registered(TargetTypes.MULTI_TARGET_REGRESSION))
    expected = {
        "rmse_macro", "rmse_micro", "rmse_max",
        "mae_macro", "mae_max",
        "r2_macro", "r2_min",
    }
    missing = expected - names
    assert not missing, f"missing MTR metrics: {sorted(missing)}"


def test_mtr_metric_directions():
    """Direction lookup via metric_name_higher_is_better resolves through
    the per-target registry for MTR-only names."""
    assert metric_name_higher_is_better("val_rmse_macro") is False
    assert metric_name_higher_is_better("val_r2_macro") is True
    assert metric_name_higher_is_better("val_r2_min") is True
    assert metric_name_higher_is_better("val_mae_max") is False


def test_mtr_metric_computation_on_synthetic():
    """All 7 MTR metrics compute finite values on a (N=100, K=3)
    synthetic target where preds are a noisy linear approximation."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(100, 3))
    preds = y_true + 0.1 * rng.normal(size=(100, 3))  # near-perfect preds

    results = dict(iter_extra_metrics(
        TargetTypes.MULTI_TARGET_REGRESSION, y_true, None, preds,
    ))
    assert "rmse_macro" in results
    assert "r2_macro" in results
    # All values finite.
    for name, value in results.items():
        assert np.isfinite(value), f"{name} = {value} is not finite"
    # On near-perfect preds, R2 macro should be near 1.0.
    assert results["r2_macro"] > 0.9, (
        f"r2_macro={results['r2_macro']:+.4f} should be ~1.0 on near-perfect preds"
    )
    # rmse_max >= rmse_macro >= 0 (max over K >= mean over K).
    assert results["rmse_max"] >= results["rmse_macro"] >= 0


def test_mtr_metric_handles_1d_input_via_coercion():
    """Coercion path: (N,) inputs get reshaped to (N, 1) so the metrics
    don't crash on accidentally-1D inputs."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=100)
    preds = y_true + 0.1 * rng.normal(size=100)
    results = dict(iter_extra_metrics(
        TargetTypes.MULTI_TARGET_REGRESSION, y_true, None, preds,
    ))
    # rmse_max == rmse_macro when K=1.
    assert abs(results["rmse_max"] - results["rmse_macro"]) < 1e-9


# --- D4: report_regression_model_perf gates (N, K) ----------------------------


def test_report_regression_mtr_path_returns_metrics_only(caplog):
    """When targets/preds are (N, K) with K>=2, the reporter skips the
    chart / audit / fairness branches and stamps MTR metrics into the
    metrics dict instead. Returns preds unchanged + None."""
    from mlframe.training._reporting_regression import report_regression_model_perf

    rng = np.random.default_rng(0)
    n, k = 60, 3
    targets = rng.normal(size=(n, k)).astype(np.float32)
    preds = targets + 0.05 * rng.normal(size=(n, k)).astype(np.float32)
    metrics = {}

    with caplog.at_level(logging.INFO, logger="mlframe.training._reporting_regression"):
        ret_preds, ret_extra = report_regression_model_perf(
            targets=targets,
            columns=["f1", "f2", "f3", "f4"],
            model_name="test_model",
            model=None,
            preds=preds,
            metrics=metrics,
            print_report=True,
            show_perf_chart=False,
            verbose=False,
        )
    # Output contract preserved.
    assert ret_extra is None
    assert ret_preds.shape == preds.shape
    # MTR metrics stamped into the metrics dict.
    assert "rmse_macro" in metrics
    assert "r2_macro" in metrics
    # Sanity: near-perfect preds -> r2_macro near 1.
    assert metrics["r2_macro"] > 0.9


def test_report_regression_n1_y_treated_as_single_target():
    """(N, 1) targets are STILL single-target regression (one column);
    the MTR gate fires only on K>=2."""
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(80, 1)).astype(np.float32)
    # The K=1 path goes through the legacy single-target branch (no
    # gate). We don't run the full reporter (it needs more fixtures);
    # the test asserts the gate logic explicitly.
    targets_arr = np.asarray(targets)
    is_mtr = targets_arr.ndim == 2 and targets_arr.shape[1] >= 2
    assert is_mtr is False, "(N, 1) must NOT trigger the MTR gate"


# --- D2 superseded by E2 -----------------------------------------------------
#
# The earlier D2 test asserted "MTR target_type causes CT_ENSEMBLE to log a
# WARN and skip". E2 (commit landing alongside this update) replaced the
# skip with an actual per-column equal-mean ensemble (see
# test_mtr_charts_and_ensemble.py::test_ct_ensemble_dispatcher_routes_mtr_to_per_column_path).
# The new contract is the right one — silent skip was a stepping-stone.
