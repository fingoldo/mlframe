"""partial_fit API + equivalence coverage for MRMR.

Gap addressed: no existing test pins the documented partial_fit contract -- that with
``partial_fit_decay=0`` a streamed two-batch ``partial_fit`` recompute equals the single
``fit`` over the concatenated data (the buffer just accumulates rows, decay=0 weights
every row equally), that ``partial_fit`` returns ``self``, and that batches below
``partial_fit_min_recompute`` defer the refit (support_ unchanged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _data(n: int = 200, m: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int32)
    return pd.DataFrame(X, columns=list("abcd")[:m]), y


def _fast(**kw):
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1, random_seed=4)
    base.update(kw)
    return MRMR(**base)


def test_partial_fit_decay0_equals_concatenated_fit():
    """decay=0, min_recompute=1: two-batch partial_fit recompute == fit on concatenated data."""
    X, y = _data(seed=11)

    MRMR._FIT_CACHE.clear()
    full = _fast().fit(X, y)
    sup_full = np.sort(np.asarray(full.support_))

    MRMR._FIT_CACHE.clear()
    pf = _fast(partial_fit_decay=0.0, partial_fit_min_recompute=1)
    pf.partial_fit(X.iloc[:100], y[:100])
    pf.partial_fit(X.iloc[100:], y[100:])
    sup_pf = np.sort(np.asarray(pf.support_))

    np.testing.assert_array_equal(sup_full, sup_pf)


def test_partial_fit_returns_self():
    X, y = _data(seed=12)
    MRMR._FIT_CACHE.clear()
    pf = _fast(partial_fit_min_recompute=1)
    assert pf.partial_fit(X, y) is pf


def test_partial_fit_first_call_fits():
    """The first partial_fit call is equivalent to fit on that batch (support populated)."""
    X, y = _data(seed=13)
    MRMR._FIT_CACHE.clear()
    pf = _fast(partial_fit_min_recompute=1)
    pf.partial_fit(X, y)
    assert hasattr(pf, "support_") and np.asarray(pf.support_).size >= 1


def test_partial_fit_below_min_recompute_defers_refit():
    """A second batch smaller than min_recompute does NOT trigger a refit -> support_ unchanged."""
    X, y = _data(seed=14)
    MRMR._FIT_CACHE.clear()
    pf = _fast(partial_fit_min_recompute=10_000)
    pf.partial_fit(X.iloc[:100], y[:100])
    sup_before = np.sort(np.asarray(pf.support_))
    pf.partial_fit(X.iloc[100:110], y[100:110])  # only 10 new rows, well below threshold
    sup_after = np.sort(np.asarray(pf.support_))
    np.testing.assert_array_equal(sup_before, sup_after)


def test_partial_fit_rejects_mismatched_lengths():
    X, y = _data(seed=15)
    MRMR._FIT_CACHE.clear()
    pf = _fast()
    with pytest.raises(ValueError):
        pf.partial_fit(X.iloc[:50], y[:40])


def test_partial_fit_rejects_empty_batch():
    X, _y = _data(seed=16)
    MRMR._FIT_CACHE.clear()
    pf = _fast()
    with pytest.raises(ValueError):
        pf.partial_fit(X.iloc[:0], np.asarray([], dtype=np.int32))
