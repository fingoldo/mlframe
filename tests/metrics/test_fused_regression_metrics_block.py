"""Regression coverage for ``fast_regression_metrics_block`` -- the fused
single-pass kernel that replaces 4 separate ``fast_*`` calls in the
regression-reporting block.

User asked 2026-05-22 whether we can compute all regression metrics in one
numpy / numba pass. The bench at
``mlframe.metrics._benchmarks.bench_fused_regression_metrics`` measures
2.3-3.4x speedup across 10k/500k/5M sizes at <1e-12 absolute drift vs the
sklearn-equivalent fast_* helpers. This file pins:

- Numerical equivalence (drop-in replacement contract).
- Edge cases: empty arrays, constant y_true, zero residuals, large-mean y
  (the catastrophic-cancellation regime that broke the naive identity
  formulation).
- Multidim inputs raise (the fused kernel is 1-D only).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import (
    fast_max_error,
    fast_mean_absolute_error,
    fast_r2_score,
    fast_regression_metrics_block,
    fast_root_mean_squared_error,
)


def _ref_block(y_true, y_pred):
    """Reference dict computed via the 4 separate ``fast_*`` kernels."""
    _max_err = fast_max_error(y_true, y_pred)
    return {
        "MAE": float(fast_mean_absolute_error(y_true, y_pred)),
        "RMSE": float(fast_root_mean_squared_error(y_true, y_pred)),
        "MaxError": float(np.max(_max_err)) if isinstance(_max_err, np.ndarray) else float(_max_err),
        "R2": float(fast_r2_score(y_true, y_pred)),
    }


@pytest.mark.parametrize("n", [10_000, 100_000, 500_000])
@pytest.mark.parametrize("y_mean,y_std", [
    (0.0, 1.0),       # standard-normal
    (11500.0, 645.0), # TVT-2026-05-22 shape (large-mean, the catastrophic-cancellation regime)
    (-50.0, 0.01),    # near-constant target (sensitive R^2 denom)
])
def test_numerical_equivalence_vs_separate_kernels(n, y_mean, y_std):
    rng = np.random.default_rng(0)
    y_true = rng.normal(y_mean, y_std, n).astype(np.float64)
    y_pred = y_true + rng.normal(0, max(0.01 * y_std, 1e-6), n).astype(np.float64)

    fused = fast_regression_metrics_block(y_true, y_pred)
    ref = _ref_block(y_true, y_pred)

    for k in ("MAE", "RMSE", "MaxError", "R2"):
        diff = abs(fused[k] - ref[k])
        tol = max(1e-9, abs(ref[k]) * 1e-9)
        assert diff < tol, (
            f"fused {k} = {fused[k]}, ref = {ref[k]}, diff = {diff} (tol = {tol}) "
            f"at n={n}, y_mean={y_mean}, y_std={y_std}"
        )


def test_empty_input_returns_zero_dict():
    out = fast_regression_metrics_block(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    assert out == {"MAE": 0.0, "RMSE": 0.0, "MaxError": 0.0, "R2": 0.0}


def test_constant_y_true_returns_r2_zero_when_perfect():
    """sklearn convention: constant y_true with zero residuals -> R^2 = 0.0."""
    y_true = np.full(100, 5.0, dtype=np.float64)
    y_pred = y_true.copy()
    out = fast_regression_metrics_block(y_true, y_pred)
    assert out["R2"] == 0.0
    assert out["MAE"] == 0.0
    assert out["RMSE"] == 0.0
    assert out["MaxError"] == 0.0


def test_multidim_input_raises():
    y_true_2d = np.zeros((100, 2), dtype=np.float64)
    y_pred_2d = np.zeros((100, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="1-D"):
        fast_regression_metrics_block(y_true_2d, y_pred_2d)


def test_large_mean_target_stable_r2():
    """The naive single-pass identity ``sum_y_sq - n*y_mean^2`` catastrophically
    cancels when y has a large mean. The fused 2-pass formulation must produce
    a stable R^2 in that regime."""
    rng = np.random.default_rng(11)
    n = 1_000_000
    # y_mean=1e6, y_std=10 -> sum_y_sq is ~1e18, n*y_mean^2 is ~1e18,
    # difference is ~1e8. Float64 keeps ~15-16 digits; the single-pass
    # identity loses ~10 digits to cancellation.
    y_true = rng.normal(1_000_000.0, 10.0, n).astype(np.float64)
    y_pred = y_true + rng.normal(0.0, 1.0, n).astype(np.float64)
    out = fast_regression_metrics_block(y_true, y_pred)
    ref = _ref_block(y_true, y_pred)
    # R^2 ~ 0.99 on this signal; abs diff < 1e-9 confirms 2-pass stability.
    assert abs(out["R2"] - ref["R2"]) < 1e-9
