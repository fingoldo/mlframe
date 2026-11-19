"""The float64 radix routes of ``unique_with_counts`` and ``sort_xy`` give what ``np.unique`` / ``np.argsort`` give, and the MDLP edges do not change."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import supervised_binning as sb
from mlframe.feature_selection.filters.discretization import _quantile_radix as qr


def _columns():
    """Float64 columns of 20000 rows: continuous, heavily tied, signed, with exact zeros of both signs."""
    r = np.random.default_rng(0)
    cont = r.standard_normal(20000)
    tied = np.round(r.standard_normal(20000) * 6) / 6
    signed = np.concatenate([-np.abs(r.standard_normal(10000)) * 1e-300, np.abs(r.standard_normal(10000)) * 1e300])
    zeros = np.where(r.random(20000) < 0.3, 0.0, r.standard_normal(20000))
    zeros[::7] = -0.0
    return {"continuous": cont, "tied": tied, "extreme": signed, "zeros": zeros}


@pytest.mark.parametrize("name", ["continuous", "tied", "extreme", "zeros"])
def test_unique_counts_equal_numpy(name):
    """Same values and counts as ``np.unique`` (an exact zero of either sign counts as one value)."""
    x = _columns()[name]
    want_v, want_c = np.unique(x, return_counts=True)
    got_v, got_c = qr.unique_with_counts(x)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_c, want_c)


@pytest.mark.parametrize("name", ["continuous", "tied", "extreme", "zeros"])
def test_sort_xy_equals_stable_argsort(name):
    """Sorted x equals ``np.sort``; every row keeps its label (the pairs are those of a stable argsort, ties in input order)."""
    x = _columns()[name]
    y = np.random.default_rng(1).integers(0, 3, x.size).astype(np.int8)
    got = qr.sort_xy(x, y)
    assert got is not None
    xs, ys = got
    order = np.argsort(x, kind="stable")
    np.testing.assert_array_equal(xs, x[order])
    # the labels of every group of equal x agree as multisets (the radix order puts -0.0 before +0.0, equal values, so the order inside such a tie can differ)
    starts = np.flatnonzero(np.concatenate([[True], xs[1:] != xs[:-1], [True]]))
    want_ys = y[order]
    for lo, hi in zip(starts[:-1], starts[1:]):
        assert np.array_equal(np.sort(ys[lo:hi]), np.sort(want_ys[lo:hi]))
    if name in ("continuous", "extreme"):  # no ties at all: the order is exactly the stable one
        np.testing.assert_array_equal(ys, want_ys)


def test_nan_input_keeps_the_numpy_route():
    """A column with NaN is not taken by the radix routes."""
    x = np.random.default_rng(2).standard_normal(10000)
    x[5] = np.nan
    assert qr.sort_xy(x, np.zeros(x.size, dtype=np.int8)) is None
    v, c = qr.unique_with_counts(x)
    wv, wc = np.unique(x, return_counts=True)
    np.testing.assert_array_equal(c, wc)
    np.testing.assert_array_equal(v[:-1], wv[:-1])


@pytest.mark.parametrize("name", ["continuous", "tied", "extreme"])
def test_mdlp_edges_do_not_change(name, monkeypatch):
    """The validated MDLP edges of a float64 column are exactly those of the numpy route."""
    x = _columns()[name]
    y = (np.random.default_rng(3).random(x.size) < 0.5 + 0.25 * np.tanh(x / (np.std(x) or 1.0))).astype(np.int8)
    got = sb.mdlp_bin_edges(x, y)
    monkeypatch.setattr(qr, "_UNIQUE_COUNTS_MIN_SIZE", 10**9)
    want = sb.mdlp_bin_edges(x, y)
    np.testing.assert_array_equal(got, want)
