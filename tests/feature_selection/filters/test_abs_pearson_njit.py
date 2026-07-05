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


def test_abs_pearson_drops_nonfinite_rows_exactly():
    """Regression guard for the branchless + reassoc-fastmath kernel (2026-07, ~2.5x): the fastmath set MUST keep
    ``nnan``/``ninf`` so NaN/inf rows are dropped EXACTLY. A tempting full ``fastmath=True`` lets LLVM assume
    no-NaN and drop the ``isfinite`` test, silently admitting the poisoned rows and collapsing |corr| toward 0 --
    a selection-BREAKING ~1e-2 error. This test FAILS on that unsafe variant (got ~0 while ref >= 0.9) and passes
    on both the pre-fix fastmath=False kernel and the shipped safe-fastmath one (they drop the rows identically)."""
    rng = np.random.default_rng(3)
    n = 20000
    y = rng.standard_normal(n)
    v = 0.9 * y + 0.1 * rng.standard_normal(n)  # strong true corr on the finite rows
    bad = rng.choice(n, int(n * 0.3), replace=False)  # poison 30% of v with NaN / inf
    v[bad[0::2]] = np.nan
    v[bad[1::2]] = np.inf
    ref = _numpy_ref(y, v)  # numpy masked reference: drops the poisoned rows
    got = abs_pearson(y, v)
    assert ref >= 0.9, ref  # sanity: the surviving finite rows carry the strong signal
    assert abs(ref - got) <= 1e-9, (ref, got)  # exact row-drop; a no-NaN fastmath would return ~0 here


def test_abs_pearson_reassoc_delta_is_selection_safe():
    """The reassoc-fastmath reduction reorders the sums, so the result differs from a strict left-to-right numpy
    reference by at most a few ULP (~1e-13) -- far below every usability gate margin (min_corr 0.6; the
    tail-concentration gap is ~0.99 vs ~0.06). Pins that the divergence stays in the selection-safe band."""
    rng = np.random.default_rng(7)
    for n in (600, 5000, 30000):
        y = rng.standard_normal(n).astype(np.float32)
        v = (0.4 * y + 0.9 * rng.standard_normal(n).astype(np.float32)).astype(np.float32)
        assert abs(_numpy_ref(y.astype(np.float64), v.astype(np.float64)) - abs_pearson(y, v)) <= 1e-12, n
