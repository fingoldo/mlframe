"""Sorting the uncached kept columns of the dedup scan on the pool returns the orders the serial calls give."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl import _eng_dedup_batch_corr as B


def _cols(k=12, n=3000):
    """Columns with NaN, +-inf and heavy ties, so the stable order matters."""
    rng = np.random.default_rng(9)
    out = []
    for j in range(k):
        c = np.round(rng.standard_normal(n), 1).astype(np.float32)
        c[:: 7 + j] = np.nan
        if j % 3 == 0:
            c[5::50] = np.inf
        out.append(np.ascontiguousarray(c))
    return out


def test_pooled_orders_equal_the_serial_ones_cached_and_uncached():
    """Fresh columns, then a second call that is served from the cache, then a mix of cached and fresh: all equal ``finite_sort_order``."""
    B.clear_finite_sort_order_cache()
    cols = _cols()
    want = [B.finite_sort_order(c, cache=False) for c in cols]
    got = B.finite_sort_orders(cols)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
    again = B.finite_sort_orders(cols)
    for g, w in zip(again, want):
        np.testing.assert_array_equal(g, w)
    mixed = cols[:4] + _cols(3)
    for g, c in zip(B.finite_sort_orders(mixed), mixed):
        np.testing.assert_array_equal(g, B.finite_sort_order(c, cache=False))
    B.clear_finite_sort_order_cache()


def test_candidate_order_prefetch_equals_the_direct_sort():
    """Every candidate's prefetched order equals ``finite_sort_order`` of its column at the criterion dtype, across several prefetch blocks."""
    import pandas as pd

    from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
    from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_scan import _CandidateOrderPrefetch

    cols = _cols(k=70, n=1500)
    X = pd.DataFrame({f"c{i}": c for i, c in enumerate(cols)})
    pre = _CandidateOrderPrefetch(X, list(X.columns))
    for name in X.columns:
        want = B.finite_sort_order(np.ascontiguousarray(X[name].to_numpy(dtype=_crit_np_dtype())), cache=False)
        np.testing.assert_array_equal(pre.get(name), want)
    assert pre.get("absent") is None
