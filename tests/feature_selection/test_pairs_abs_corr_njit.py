"""FP-equivalence of the one-pass njit |corr| kernel (noise-wrap gate) vs numpy corrcoef.

``_abs_corr_finite_njit`` replaces the isfinite-mask + boolean-index copies + two np.std + 2x2 np.corrcoef
in ``_safe_abs_corr``. It must match the numpy result to ~1e-13 (selection-safe for the 0.30/0.5 |corr| gate)
and reproduce the degenerate-column / too-few-rows short-circuits to 0.0.
"""

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _abs_corr_finite_njit


def _numpy_ref(a, y, yfin):
    """Numpy ref."""
    m = np.isfinite(a) & yfin
    if int(m.sum()) < 8:
        return 0.0
    av, yv = a[m], y[m]
    if float(np.std(av)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return 0.0
    return abs(float(np.corrcoef(av, yv)[0, 1]))


def test_abs_corr_njit_matches_numpy_with_nan():
    """Abs corr njit matches numpy with nan."""
    rng = np.random.default_rng(0)
    for n in (500, 2000, 5000):
        y = rng.standard_normal(n)
        a = 0.6 * y + 0.8 * rng.standard_normal(n)
        a[::9] = np.nan
        y[::13] = np.nan
        yfin = np.isfinite(y)
        ref = _numpy_ref(a, y, yfin)
        got = float(_abs_corr_finite_njit(np.ascontiguousarray(a), np.ascontiguousarray(y), yfin))
        assert abs(ref - got) <= 1e-13, (n, ref, got)


def test_abs_corr_njit_short_circuits():
    """Abs corr njit short circuits."""
    n = 1000
    y = np.linspace(0, 1, n)
    yfin = np.isfinite(y)
    # constant column -> 0.0
    const = np.full(n, 3.0)
    assert float(_abs_corr_finite_njit(const, y, yfin)) == 0.0
    # too few joint-finite rows -> 0.0
    a = np.full(n, np.nan)
    a[:5] = np.arange(5, dtype=np.float64)
    assert float(_abs_corr_finite_njit(a, y, yfin)) == 0.0


def test_abs_corr_njit_perfect_correlation():
    """Abs corr njit perfect correlation."""
    n = 1000
    y = np.linspace(-2, 2, n)
    a = -3.0 * y + 1.0  # |corr| == 1
    yfin = np.isfinite(y)
    assert abs(float(_abs_corr_finite_njit(a, y, yfin)) - 1.0) <= 1e-12
