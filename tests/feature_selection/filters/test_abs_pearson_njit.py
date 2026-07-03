"""FP-equivalence of the one-pass njit |Pearson| kernel behind abs_pearson vs the numpy form.

The usability |corr| distinguisher feeds WIDE-margin gates (min_corr 0.6; tail-concentration gap ~0.99 vs ~0.06),
so ~1e-13 FP agreement is selection-safe. Pins the njit kernel against a numpy reference incl NaN + degenerate.
"""
import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import abs_pearson, _abs_pearson_njit


def _numpy_ref(y, v):
    m = np.isfinite(y) & np.isfinite(v)
    if int(m.sum()) < 2:
        return 0.0
    yy = y[m]; vv = v[m]
    ys = float(yy.std()); vs = float(vv.std())
    if ys <= 0.0 or vs <= 0.0:
        return 0.0
    c = float(np.mean((yy - yy.mean()) * (vv - vv.mean())) / (ys * vs))
    return abs(c) if np.isfinite(c) else 0.0


def test_abs_pearson_matches_numpy_incl_nan():
    rng = np.random.default_rng(0)
    for n in (500, 2000, 20000):
        y = rng.standard_normal(n)
        v = 0.6 * y + 0.8 * rng.standard_normal(n)
        y[::13] = np.nan
        v[::17] = np.nan
        ref = _numpy_ref(y, v)
        got = abs_pearson(y, v)
        assert abs(ref - got) <= 1e-13, (n, ref, got)


def test_abs_pearson_short_circuits():
    n = 1000
    y = np.linspace(0, 1, n)
    assert float(_abs_pearson_njit(np.full(n, 3.0), y)) == 0.0  # constant side -> 0
    a = np.full(n, np.nan)
    a[0] = 1.0
    assert float(_abs_pearson_njit(a, y)) == 0.0  # <2 finite -> 0


def test_abs_pearson_perfect_and_sign():
    n = 1000
    y = np.linspace(-2, 2, n)
    assert abs(abs_pearson(y, -3.0 * y + 1.0) - 1.0) <= 1e-12  # |corr| == 1, sign folded
