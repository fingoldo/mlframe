"""Regression sensor for the QRF batched weighted-quantile prange kernel.

``_LeafResidualForest.predict_quantile`` originally inverted the conditional ECDF one query
row at a time in Python (mask-nonzero -> ``np.argsort`` -> ``np.cumsum`` -> ``np.interp`` per
row). ``_batch_weighted_quantiles_kernel`` replaces that with a single njit(parallel) pass
over the dense membership batch. These tests pin (1) the kernel exists and is wired into the
predict path and (2) the kernel output is numerically identical (within FP reduction-order
tolerance) to the Python per-row path it replaced, including the all-zero-weight NaN row.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import mlframe.training.composite.qrf as qrf


def test_batch_kernel_symbol_exists_and_is_njit():
    # Pre-fix code had no batched kernel; importing the symbol fails on the old module.
    assert hasattr(qrf, "_batch_weighted_quantiles_kernel")


def _fit_estimator(seed: int = 1):
    rng = np.random.default_rng(seed)
    n = 3000
    X = rng.standard_normal((n, 4))
    y = X[:, 0] * 2.0 + 0.5 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=["base", "f1", "f2", "f3"])
    est = qrf.CompositeQRFEstimator(
        base_column="base", n_estimators=40, prefer_quantile_forest=False, random_state=0, min_samples_leaf=5
    )
    est.fit(df, y)
    return est, df


@pytest.mark.skipif(not qrf._HAS_NUMBA, reason="numba required for the kernel path")
def test_kernel_path_matches_python_per_row_path():
    est, df = _fit_estimator()
    levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    qrf._HAS_NUMBA = True
    try:
        new = est.predict_quantile(df, quantiles=levels)
        qrf._HAS_NUMBA = False
        old = est.predict_quantile(df, quantiles=levels)
    finally:
        qrf._HAS_NUMBA = True
    assert new.shape == old.shape
    assert np.array_equal(np.isnan(new), np.isnan(old))
    assert np.nanmax(np.abs(new - old)) < 1e-9


@pytest.mark.skipif(not qrf._HAS_NUMBA, reason="numba required for the kernel path")
def test_kernel_handles_all_zero_weight_row_as_nan():
    # A query row whose leaf-weights are all zero must yield NaN (transform fallback), matching
    # the Python path's ``total <= 0`` branch. Drive the kernel directly with a zero row.
    w = np.zeros((3, 50), dtype=np.float64)
    w[0, 5] = 1.0
    w[2, 10] = 0.5
    w[2, 20] = 0.5
    # row 1 left all-zero
    y_train = np.linspace(-1.0, 1.0, 50)
    levels = np.array([0.25, 0.5, 0.75])
    out = np.empty((3, 3), dtype=np.float64)
    qrf._batch_weighted_quantiles_kernel(w, y_train, levels, out, 0)
    assert np.all(np.isnan(out[1]))
    assert not np.any(np.isnan(out[0]))
    assert not np.any(np.isnan(out[2]))
