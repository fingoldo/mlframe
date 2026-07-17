"""Regression: the cat-FE streaming marginal-MI cache must key on the TARGET.

The cached value is ``MI(X; Y)``. ``_column_signature`` captures only X's
bincount, so before the fix two fits with an identical X distribution but a
different / relabelled Y collided and the stale cached MI was reused -- the
column was then mis-pruned by ``marginal_floor``. The fix folds a content
signature of the (discretized) target into the cache and refuses reuse when it
changes. These tests exercise the real ``_restore_cached_marginal_mis`` /
``_target_signature`` helpers.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.cat_interactions import (
    _column_signature,
    _restore_cached_marginal_mis,
    _target_signature,
)


def _build_cache(X: np.ndarray, y: np.ndarray, cached_mi: float) -> dict:
    sig = _column_signature(X, 2)
    return {
        "target_sig": _target_signature(y),
        "col_signatures": {0: sig},
        "marginal_mis": {0: cached_mi},
    }


def test_changed_target_invalidates_cached_marginal_mi():
    """Same X distribution, different Y -> stale MI must NOT be reused.

    Pre-fix this returned ``reusable=True`` and served the cached MI=0.0,
    silently dropping a column that perfectly predicts the new target.
    """
    X = np.array([0, 1] * 50, dtype=np.int64)
    fd = X.reshape(-1, 1)
    nbins = np.array([2], dtype=np.int64)
    y_old = np.zeros(100, dtype=np.int64)  # X independent of y_old -> MI 0
    y_new = X.copy()  # X now perfectly predicts the target -> MI high

    cache = _build_cache(X, y_old, cached_mi=0.0)
    mask, _mi, _ = _restore_cached_marginal_mis(
        fd,
        np.array([0]),
        nbins,
        cache,
        kl_threshold=0.01,
        target_sig=_target_signature(y_new),
    )
    assert not bool(mask[0]), "changed Y must invalidate the cached marginal MI"


def test_same_target_and_dist_still_reuses():
    """Unchanged X distribution AND unchanged Y -> cache reuse still works."""
    X = np.array([0, 1] * 50, dtype=np.int64)
    fd = X.reshape(-1, 1)
    nbins = np.array([2], dtype=np.int64)
    y = np.zeros(100, dtype=np.int64)

    cache = _build_cache(X, y, cached_mi=0.1234)
    mask, mi, _ = _restore_cached_marginal_mis(
        fd,
        np.array([0]),
        nbins,
        cache,
        kl_threshold=0.01,
        target_sig=_target_signature(y),
    )
    assert bool(mask[0])
    assert mi[0] == 0.1234


def test_missing_target_sig_disables_reuse():
    """A legacy cache with no ``target_sig`` (or unknown current Y) must not reuse."""
    X = np.array([0, 1] * 50, dtype=np.int64)
    fd = X.reshape(-1, 1)
    nbins = np.array([2], dtype=np.int64)
    legacy = {"col_signatures": {0: _column_signature(X, 2)}, "marginal_mis": {0: 0.0}}
    mask, _, _ = _restore_cached_marginal_mis(
        fd,
        np.array([0]),
        nbins,
        legacy,
        kl_threshold=0.01,
        target_sig=None,
    )
    assert not bool(mask[0])
