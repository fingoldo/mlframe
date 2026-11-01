"""The presorted masked Spearman gives the version-1 values exactly, for NaN in either column, ties, constant columns, pairs with too few shared rows and a cached sort order."""

import numpy as np
from numba.typed import List

from mlframe.feature_selection.filters._mrmr_fit_impl import _eng_dedup_batch_corr as B


def _columns(n, seed=0):
    """A candidate with NaN and a mix of kept columns."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[rng.random(n) < 0.1] = np.nan
    base = rng.normal(size=n)
    kept = [
        base + 0.05 * rng.normal(size=n),
        np.round(rng.normal(size=n), 1),
        np.full(n, 3.0),
        np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
        np.where(np.arange(n) < 5, 1.0, np.nan),
        a * 2.0 + 1.0,
    ]
    kept[0][rng.random(n) < 0.05] = np.nan
    return a, kept


def test_presorted_equals_version_one():
    """Same |rho| (or the -1 skip mark) for every kept column, on several sizes."""
    for n in (31, 500, 6001):
        a, kept = _columns(n, seed=n)
        fin = np.isfinite(a)
        others, orders = List(), List()
        for c in kept:
            col = np.ascontiguousarray(c)
            others.append(col)
            orders.append(B.finite_sort_order(col))
        want = np.empty(len(kept))
        B.masked_abs_spearman_many(np.ascontiguousarray(a), fin, others, 1e-9, want)
        got = np.empty(len(kept))
        B.masked_abs_spearman_presorted(np.ascontiguousarray(a), fin, B.finite_sort_order(np.ascontiguousarray(a), cache=False), others, orders, 1e-9, got)
        np.testing.assert_array_equal(got, want)


def test_sort_order_cache_is_per_object():
    """A cached order is returned for the same array object and rebuilt for another with equal values."""
    col = np.random.default_rng(0).normal(size=100)
    col[7] = np.nan
    first = B.finite_sort_order(col)
    assert B.finite_sort_order(col) is first
    assert B.finite_sort_order(col.copy()) is not first
    assert 7 not in first and np.all(np.diff(col[first]) >= 0)
