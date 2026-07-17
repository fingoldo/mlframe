"""Regression for rolling_zero_crossings after njit vectorization (iter106).

The per-segment numpy multi-pass (``np.sign`` + the ``s[:,1:]*s[:,:-1] < 0`` product/boolean allocs +
``sum(axis=1)`` + ``astype``, ~2.0s/10M rows) moved to an ``@njit(parallel)`` per-window walk kernel
(~3.2x e2e at 10M) for the default ``center='zero'``. These tests pin (a) the kernel symbol exists and the
function routes through it on finite ``center='zero'`` input, (b) bit-for-bit equivalence against the numpy
reference across even/odd K, every center, and continuous / tied / discrete windows, (c) NaN-bearing
segments fall back to numpy, and (d) ``center='median'``/``'mean'`` do NOT route through the kernel (their
center is itself a reduction whose FP order would differ and could flip a near-zero sign -- gated out).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering import windowed_shape as ws
from mlframe.feature_engineering.grouped import per_group_sliding_window


def _ref(values, window_K, center):
    """Helper: Ref."""
    out = np.full(values.size, np.nan, dtype=np.float64)
    g = np.zeros(values.size, dtype=np.int64)
    for _si, wins, wi in per_group_sliding_window(values, g, window_K=window_K):
        if center == "median":
            c = np.median(wins, axis=1, keepdims=True)
        elif center == "mean":
            c = wins.mean(axis=1, keepdims=True)
        else:
            c = 0.0
        s_sign = np.sign(wins - c)
        cross = (s_sign[:, 1:] * s_sign[:, :-1]) < 0
        out[wi] = cross.sum(axis=1).astype(np.float64)
    return out


def test_kernel_symbol_present_and_routed_on_zero_center():
    """Kernel symbol present and routed on zero center."""
    assert hasattr(ws, "_zero_crossings_kernel")
    calls = {"n": 0}
    orig = ws._zero_crossings_kernel

    def spy(*a, **k):
        """Spy."""
        calls["n"] += 1
        return orig(*a, **k)

    ws._zero_crossings_kernel = spy
    try:
        v = np.random.default_rng(0).standard_normal(500)
        g = np.zeros(500, dtype=np.int64)
        ws.rolling_zero_crossings(v, g, window_K=20, center="zero")
    finally:
        ws._zero_crossings_kernel = orig
    assert calls["n"] >= 1


@pytest.mark.parametrize("center", ["median", "mean"])
def test_median_mean_centers_do_not_route_through_kernel(center):
    """Median mean centers do not route through kernel."""
    calls = {"n": 0}
    orig = ws._zero_crossings_kernel

    def spy(*a, **k):
        """Spy."""
        calls["n"] += 1
        return orig(*a, **k)

    ws._zero_crossings_kernel = spy
    try:
        v = np.random.default_rng(1).standard_normal(500)
        g = np.zeros(500, dtype=np.int64)
        ws.rolling_zero_crossings(v, g, window_K=20, center=center)
    finally:
        ws._zero_crossings_kernel = orig
    assert calls["n"] == 0


@pytest.mark.parametrize("window_K", [5, 6, 7, 20, 21])
@pytest.mark.parametrize("center", ["zero", "median", "mean"])
@pytest.mark.parametrize("kind", ["continuous", "tied_lowcard", "discrete_int", "with_zeros"])
def test_matches_numpy_reference_finite(window_K, center, kind):
    """Matches numpy reference finite."""
    rng = np.random.default_rng(7)
    n = 4000
    if kind == "continuous":
        v = rng.standard_normal(n)
    elif kind == "tied_lowcard":
        v = rng.integers(0, 4, n).astype(np.float64)
    elif kind == "discrete_int":
        v = rng.integers(-3, 4, n).astype(np.float64)
    else:
        v = rng.integers(-2, 3, n).astype(np.float64)  # many exact zeros after center=zero
    g = np.zeros(n, dtype=np.int64)
    got = ws.rolling_zero_crossings(v, g, window_K=window_K, center=center)
    exp = _ref(v, window_K, center)
    fin = ~np.isnan(exp)
    np.testing.assert_array_equal(got[fin], exp[fin])


def test_nan_segment_falls_back_to_numpy():
    """Nan segment falls back to numpy."""
    rng = np.random.default_rng(3)
    v = rng.standard_normal(3000)
    v[100] = np.nan
    g = np.zeros(3000, dtype=np.int64)
    got = ws.rolling_zero_crossings(v, g, window_K=20, center="zero")
    exp = _ref(v, 20, "zero")
    np.testing.assert_array_equal(got, exp)
