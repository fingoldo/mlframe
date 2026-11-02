"""Tied dropped raw columns resolve to the first one in column order, whatever the set iterates in (tracker 2026-11-02 wave 8, RNG8-2)."""

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._assign_support_tail import _post_drop_raw_floor_pick


class _ReversedSet:
    """Set-like whose iteration order is the reverse of column order, the order a hash seed could produce."""

    def __init__(self, names):
        self._names = list(names)

    def __iter__(self):
        return iter(reversed(self._names))


def test_tied_relevance_picks_the_first_column():
    """Two identical columns tie on MI with the target; the lower column index wins regardless of iteration order."""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 3, 600)
    col = (target + rng.integers(0, 2, 600)) % 3
    data = np.column_stack([col, col, target]).astype(np.int32)
    nbins = np.array([3, 3, 3], dtype=np.int64)
    got = _post_drop_raw_floor_pick(_ReversedSet(["a", "b"]), {"a": 0, "b": 1}, data, np.array([2]), nbins)
    assert got == "a"
