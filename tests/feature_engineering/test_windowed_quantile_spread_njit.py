"""Regression for rolling_quantile_spread after njit vectorization.

The per-segment ``np.quantile(wins, [q_low, q_high], axis=1)`` (which partitions
the whole ``(n_windows, K)`` matrix + allocates a copy + interpolates in separate
passes, ~6.4s/10M rows) moved to an ``@njit(parallel)`` per-window sort kernel
(~2.7x e2e at 10M). These tests pin (a) the presence of the kernel and that the
function routes through it on finite input, and (b) bit-for-bit equivalence
against numpy ``method='linear'`` across even/odd K, every quantile pair, and
continuous / tied / discrete / NaN-bearing windows (NaN segments fall back to
numpy, preserving sort-NaN-to-end semantics).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from mlframe.feature_engineering import windowed_shape as ws


def _ref(values, window_K, q_low, q_high):
    out = np.full(values.size, np.nan, dtype=np.float64)
    wins = sliding_window_view(values, window_K)
    q = np.quantile(wins, [q_low, q_high], axis=1)
    out[window_K - 1 :] = q[1] - q[0]
    return out


def test_kernel_symbol_present_and_routed():
    assert hasattr(ws, "_quantile_spread_kernel")
    calls = {"n": 0}
    orig = ws._quantile_spread_kernel

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    ws._quantile_spread_kernel = spy
    try:
        v = np.random.default_rng(0).standard_normal(500)
        g = np.zeros(500, dtype=np.int64)
        ws.rolling_quantile_spread(v, g, window_K=20)
    finally:
        ws._quantile_spread_kernel = orig
    assert calls["n"] >= 1


@pytest.mark.parametrize("window_K", [5, 7, 20, 50])
@pytest.mark.parametrize("qs", [(0.1, 0.9), (0.25, 0.75), (0.0, 1.0), (0.05, 0.95), (0.5, 0.95)])
@pytest.mark.parametrize("kind", ["continuous", "tied_lowcard", "discrete_int"])
def test_matches_numpy_reference_finite(window_K, qs, kind):
    rng = np.random.default_rng(7)
    n = 4000
    if kind == "continuous":
        v = rng.standard_normal(n)
    elif kind == "tied_lowcard":
        v = rng.integers(0, 4, n).astype(np.float64)
    else:
        v = rng.integers(0, 50, n).astype(np.float64)
    v = v.astype(np.float64)
    g = np.zeros(n, dtype=np.int64)
    q_low, q_high = qs
    got = ws.rolling_quantile_spread(v, g, window_K=window_K, q_low=q_low, q_high=q_high)
    exp = _ref(v, window_K, q_low, q_high)
    # Bit-for-bit equal to numpy method='linear' on finite windows.
    fin = ~np.isnan(exp)
    np.testing.assert_array_equal(got[fin], exp[fin])


def test_nan_segment_falls_back_to_numpy():
    rng = np.random.default_rng(3)
    v = rng.standard_normal(3000)
    v[100] = np.nan
    g = np.zeros(3000, dtype=np.int64)
    got = ws.rolling_quantile_spread(v, g, window_K=20)
    exp = _ref(v, 20, 0.1, 0.9)
    np.testing.assert_array_equal(got, exp)  # equal_nan via NaN==NaN positions
