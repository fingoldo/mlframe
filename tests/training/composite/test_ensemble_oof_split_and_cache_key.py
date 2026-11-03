"""The ensemble's outer OOF split keeps groups whole even with fewer groups than folds, and its cache key sees the whole group / time arrays."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.ensemble import _oof_full_cache_key, _outer_oof_split


def test_three_groups_five_folds_never_split_a_group():
    """kfold=5 over 3 groups used to fall back to a shuffled K-fold; every group must now sit wholly in one side of each fold."""
    g = np.repeat(np.arange(3), 20)
    folds = list(_outer_oof_split(60, 5, 0, None, g, False))
    assert len(folds) == 3
    for tr, te in folds:
        assert not set(g[tr]) & set(g[te])


def test_cache_key_differs_when_only_the_interior_of_the_groups_differs():
    """Two group arrays with equal first and last 64 labels but different interiors must not share an entry."""
    a = np.zeros(300, dtype=int)
    b = a.copy()
    b[150] = 1
    ka = _oof_full_cache_key("k", 5, 0, 0.2, None, None, None, a)
    kb = _oof_full_cache_key("k", 5, 0, 0.2, None, None, None, b)
    assert ka != kb


def test_cache_key_differs_for_different_time_orders():
    """Two different time orderings of equal length are different splits."""
    t1, t2 = np.arange(50), np.arange(50)[::-1]
    assert _oof_full_cache_key("k", 5, 0, 0.2, t1, None, None, None) != _oof_full_cache_key("k", 5, 0, 0.2, t2, None, None, None)
