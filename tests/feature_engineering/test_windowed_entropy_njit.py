"""Regression for rolling_shannon_entropy_binned after njit vectorization.

The per-window Shannon-entropy scan moved from a pure-Python loop (one
``np.quantile`` + ``np.unique`` + ``np.histogram`` + ``np.log`` dispatch per
window, ~190s/1M rows) to an ``@njit(parallel)`` kernel (~670x at 1M). These
tests pin (a) the presence of the kernel and that the function routes through
it, and (b) bit-for-bit-modulo-ULP equivalence against an independent
reimplementation of the prior per-window numpy loop, across both bin
strategies and continuous / tied / discrete / NaN-bearing windows.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering import windowed_shape as ws
from mlframe.feature_engineering.grouped import per_group_sliding_window


def _ref(values, group_ids, window_K, n_bins, bin_strategy):
    """Independent reimplementation of the pre-njit per-window numpy loop."""
    out = np.full(values.size, np.nan, dtype=np.float64)
    for _si, wins, wi in per_group_sliding_window(values, group_ids, window_K=window_K):
        n_wins, _K = wins.shape
        ent = np.full(n_wins, np.nan, dtype=np.float64)
        for r in range(n_wins):
            w = wins[r]
            wf = w[np.isfinite(w)]
            if wf.size < 2:
                continue
            if bin_strategy == "quantile":
                edges = np.unique(np.quantile(wf, np.linspace(0, 1, n_bins + 1)))
                if edges.size < 2:
                    ent[r] = 0.0
                    continue
                counts, _ = np.histogram(wf, bins=edges)
            else:
                lo, hi = float(wf.min()), float(wf.max())
                if hi - lo < 1e-12:
                    ent[r] = 0.0
                    continue
                counts, _ = np.histogram(wf, bins=n_bins, range=(lo, hi + 1e-12))
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent[r] = float(-np.sum(probs * np.log(probs)))
        out[wi] = ent
    return out


def test_kernel_symbol_present_and_routed():
    # Pre-fix code had no kernel and used a Python loop; this guards the regression.
    assert hasattr(ws, "_shannon_entropy_binned_kernel")
    calls = {"n": 0}
    orig = ws._shannon_entropy_binned_kernel

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    ws._shannon_entropy_binned_kernel = spy
    try:
        v = np.random.default_rng(0).standard_normal(500)
        g = np.zeros(500, dtype=np.int64)
        ws.rolling_shannon_entropy_binned(v, g, window_K=20)
    finally:
        ws._shannon_entropy_binned_kernel = orig
    assert calls["n"] >= 1


@pytest.mark.parametrize("bin_strategy", ["quantile", "uniform"])
@pytest.mark.parametrize("n_bins", [4, 8, 16])
@pytest.mark.parametrize(
    "kind", ["continuous", "tied_lowcard", "discrete_int", "with_nan"]
)
def test_matches_numpy_reference(bin_strategy, n_bins, kind):
    rng = np.random.default_rng(7)
    n = 8000
    if kind == "continuous":
        v = rng.standard_normal(n)
    elif kind == "tied_lowcard":
        v = rng.integers(0, 5, n).astype(np.float64)
    elif kind == "discrete_int":
        v = rng.integers(0, 50, n).astype(np.float64)
    else:
        v = np.where(rng.random(n) < 0.1, np.nan, rng.standard_normal(n))
    v = v.astype(np.float64)
    g = (np.arange(n) // 2000).astype(np.int64)
    got = ws.rolling_shannon_entropy_binned(
        v, g, window_K=20, n_bins=n_bins, bin_strategy=bin_strategy
    )
    exp = _ref(v, g, window_K=20, n_bins=n_bins, bin_strategy=bin_strategy)
    # NaN positions identical.
    np.testing.assert_array_equal(np.isnan(got), np.isnan(exp))
    fin = ~np.isnan(got)
    # Entropy values: ULP-level (summation-order) agreement.
    np.testing.assert_allclose(got[fin], exp[fin], rtol=0, atol=1e-12)
