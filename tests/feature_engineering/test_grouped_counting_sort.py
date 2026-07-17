"""Regression tests for the O(n) integer counting-sort fast path in iter_group_segments.

The fast path is bit-identical to ``np.argsort(kind="stable")`` segmentation but skips the
O(n log n) sort for bounded-span integer group ids. These tests pin (a) that the fast path is
actually taken for integer keys (FAILS on pre-fix code which always called np.argsort) and
(b) that it produces output identical to the argsort path on ties / negatives / single group.
"""

import numpy as np
import pytest

from mlframe.feature_engineering import grouped
from mlframe.feature_engineering.grouped import (
    iter_group_segments,
    per_group_shift,
    per_group_cum_reduce,
    per_group_rank,
)


def _argsort_segments(group_ids):
    g = np.ascontiguousarray(group_ids)
    n = g.size
    sort_idx = np.argsort(g, kind="stable")
    g_sorted = g[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], bnd)).astype(np.intp)
    ends = np.concatenate((bnd, [n])).astype(np.intp)
    return sort_idx, starts, ends


@pytest.mark.parametrize(
    "gids",
    [
        np.array([2, 0, 1, 0, 2, 1, 1, 0], dtype=np.int64),  # ties, unordered
        np.array([-3, -1, -3, 5, -1, 5, 5], dtype=np.int64),  # negatives
        np.array([7, 7, 7, 7], dtype=np.int64),  # single group
        np.arange(50, dtype=np.int64),  # all distinct
        np.zeros(10, dtype=np.int64),  # one big group
    ],
)
def test_counting_sort_bit_identical_to_argsort(gids):
    si, st, en = iter_group_segments(gids)
    bsi, bst, ben = _argsort_segments(gids)
    assert np.array_equal(si, bsi)
    assert np.array_equal(st, bst)
    assert np.array_equal(en, ben)


def test_integer_path_skips_argsort(monkeypatch):
    """Integer gids must NOT call np.argsort -- the counting-sort path handles them.

    Pre-fix code always called np.argsort(kind='stable'); this spy trips on it.
    """
    gids = np.array([3, 1, 2, 1, 3, 2, 1], dtype=np.int64)
    called = {"n": 0}
    real_argsort = np.argsort

    def spy(*a, **k):
        called["n"] += 1
        return real_argsort(*a, **k)

    monkeypatch.setattr(grouped.np, "argsort", spy)
    iter_group_segments(gids)
    assert called["n"] == 0, "integer gids should use counting sort, not np.argsort"


def test_huge_span_falls_back_to_argsort(monkeypatch):
    """Sparse integer keys (span >> n) must keep the argsort path to stay RAM-safe."""
    gids = np.array([0, 10**12, 5 * 10**11], dtype=np.int64)
    called = {"n": 0}
    real_argsort = np.argsort

    def spy(*a, **k):
        called["n"] += 1
        return real_argsort(*a, **k)

    monkeypatch.setattr(grouped.np, "argsort", spy)
    si, _st, _en = iter_group_segments(gids)
    assert called["n"] >= 1
    assert np.array_equal(si, _argsort_segments(gids)[0])


def test_per_group_helpers_identity_via_fast_path():
    rng = np.random.default_rng(1)
    n = 5000
    gids = rng.integers(0, 200, size=n).astype(np.int64)
    vals = rng.standard_normal(n)

    a = per_group_shift(vals, gids, 1)
    b_si, b_st, b_en = _argsort_segments(gids)
    out = np.full(n, np.nan)
    for s, e in zip(b_st, b_en):
        seg = b_si[s:e]
        if seg.size > 1:
            out[seg[1:]] = vals[seg[:-1]]
    assert np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(out, nan=-9e9))

    cum = per_group_cum_reduce(vals, gids, "sum")
    rank = per_group_rank(vals, gids)
    assert np.isfinite(cum).all()
    assert np.isfinite(rank).all()
