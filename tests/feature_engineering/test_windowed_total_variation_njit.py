"""Regression for rolling_total_variation after njit vectorization (iter107).

The per-segment numpy multi-pass (``np.abs(np.diff(wins,axis=1)).sum(axis=1)`` -- a (n_windows, K-1) diff alloc
+ abs alloc + a separate sum pass, plus two more full-matrix reductions when ``normalize``) moved to an
``@njit(parallel)`` per-window single-pass kernel (~1.74x no-norm / ~2.45x normalize e2e at 10M). These tests
pin (a) the kernel symbol exists and the function routes through it on finite input with K>=2, (b) numerical
equivalence against the numpy reference (~1e-15 reduction-order tolerance; numpy sums pairwise, the kernel
left-to-right) across even/odd K and continuous / tied / discrete windows for both ``normalize`` settings,
(c) NaN-bearing segments fall back to numpy bit-for-bit, and (d) K=1 (no diffs) yields all-zero TV.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering import windowed_shape as ws
from mlframe.feature_engineering.grouped import per_group_sliding_window


def _ref(values, window_K, normalize):
    out = np.full(values.size, np.nan, dtype=np.float64)
    g = np.zeros(values.size, dtype=np.int64)
    for _si, wins, wi in per_group_sliding_window(values, g, window_K=window_K):
        tv = np.abs(np.diff(wins, axis=1)).sum(axis=1)
        if normalize:
            tv = tv / ((wins.max(axis=1) - wins.min(axis=1)) + 1e-12)
        out[wi] = tv
    return out


def test_kernel_symbol_present_and_routed_on_finite_input():
    assert hasattr(ws, "_total_variation_kernel")
    calls = {"n": 0}
    orig = ws._total_variation_kernel

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    ws._total_variation_kernel = spy
    try:
        v = np.random.default_rng(0).standard_normal(500)
        g = np.zeros(500, dtype=np.int64)
        ws.rolling_total_variation(v, g, window_K=20)
    finally:
        ws._total_variation_kernel = orig
    assert calls["n"] >= 1


@pytest.mark.parametrize("window_K", [2, 5, 6, 20, 21])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kind", ["continuous", "tied_lowcard", "discrete_int"])
def test_matches_numpy_reference_finite(window_K, normalize, kind):
    rng = np.random.default_rng(7)
    n = 4000
    if kind == "continuous":
        v = rng.standard_normal(n).cumsum()
    elif kind == "tied_lowcard":
        v = rng.integers(0, 4, n).astype(np.float64)
    else:
        v = rng.integers(-3, 4, n).astype(np.float64)
    g = np.zeros(n, dtype=np.int64)
    got = ws.rolling_total_variation(v, g, window_K=window_K, normalize=normalize)
    exp = _ref(v, window_K, normalize)
    fin = ~np.isnan(exp)
    np.testing.assert_allclose(got[fin], exp[fin], rtol=1e-12, atol=1e-12)


def test_nan_segment_falls_back_to_numpy_exact():
    rng = np.random.default_rng(3)
    v = rng.standard_normal(3000)
    v[100] = np.nan
    g = np.zeros(3000, dtype=np.int64)
    got = ws.rolling_total_variation(v, g, window_K=20)
    exp = _ref(v, 20, False)
    np.testing.assert_array_equal(got, exp)


def test_window_k_one_is_all_zero_tv():
    v = np.random.default_rng(5).standard_normal(1000)
    g = np.zeros(1000, dtype=np.int64)
    got = ws.rolling_total_variation(v, g, window_K=1)
    assert np.all(got == 0.0)
