"""Regression tests for the serial/parallel gate on ``_nanminmax_cols`` in compute_naive_outlier_score (PERF LOW, 2026-07).

Small frames (n*d below the measured ~20k crossover) route to a serial njit sweep that avoids the parallel kernel's thread-spawn + reduce-buffer
overhead. The serial kernel must be bit-identical to the parallel one (including the all-NaN-column -> NaN collapse), and the outlier score must be
unchanged whether the small-frame gate fires or not.
"""

import numpy as np
import pytest

from mlframe.preprocessing import outliers as _o
from mlframe.preprocessing.outliers import (
    _nanminmax_cols,
    _nanminmax_cols_serial,
    compute_naive_outlier_score,
)

pytest.importorskip("numba")


@pytest.mark.parametrize("n,d", [(1, 1), (3, 4), (200, 4), (5000, 8), (50000, 30)])
def test_serial_matches_parallel_nanminmax(n, d):
    rng = np.random.default_rng(0)
    X = rng.random((n, d))
    # Sprinkle NaNs, and force one fully-NaN column to exercise the empty-slice collapse.
    X[X < 0.02] = np.nan
    if d > 1:
        X[:, d - 1] = np.nan
    mp, xp = _nanminmax_cols(X)
    ms, xs = _nanminmax_cols_serial(X)
    np.testing.assert_array_equal(np.isnan(mp), np.isnan(ms))
    np.testing.assert_array_equal(np.isnan(xp), np.isnan(xs))
    finite = ~np.isnan(mp)
    np.testing.assert_array_equal(mp[finite], ms[finite])
    np.testing.assert_array_equal(xp[finite], xs[finite])


def test_score_identical_across_gate_boundary(monkeypatch):
    rng = np.random.default_rng(1)
    X_train = rng.random((300, 5))
    X_test = rng.random((120, 5))
    # Below the gate -> serial path.
    assert X_train.shape[0] * X_train.shape[1] < _o._NANMINMAX_PARALLEL_MIN_ELEMS
    score_gated = compute_naive_outlier_score(X_train, X_test)
    # Force the parallel path by dropping the threshold to 0.
    monkeypatch.setattr(_o, "_NANMINMAX_PARALLEL_MIN_ELEMS", 0)
    score_parallel = compute_naive_outlier_score(X_train, X_test)
    np.testing.assert_array_equal(score_gated, score_parallel)
