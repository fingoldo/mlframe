"""The collinearity confirmation keeps its unit-column cache within a byte budget, the verdicts do not depend on the cache size, and the streaming detector finds exact
scaled and sign-flipped copies at float32 (a 1e-9 tolerance under float32 rounding used to miss them)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _mrmr_degenerate as M


def _frame(n=3000, seed=0):
    """Twelve columns with three exact collinear pairs (scaled and sign-flipped copies) among noise."""
    rng = np.random.default_rng(seed)
    base = {f"c{i}": rng.standard_normal(n) for i in range(9)}
    base["d0"] = 3.0 * base["c1"] + 2.0
    base["d1"] = -base["c4"]
    base["d2"] = 0.5 * base["c7"] - 1.0
    return pd.DataFrame(base)


def test_verdicts_equal_with_a_tiny_cache(monkeypatch):
    """A budget of two unit columns gives the same collinear hits as the default."""
    X = _frame()
    live = list(X.columns)
    good_a, hits_a = M._collinear_hits_streaming(X, live, len(X))
    monkeypatch.setattr(M, "_UNIT_CACHE_BYTES", 1)
    good_b, hits_b = M._collinear_hits_streaming(X, live, len(X))
    assert (good_a == good_b).all() and hits_a == hits_b
    assert {live[j] for j in hits_a} == {"d0", "d1", "d2"}


def test_cache_size_follows_the_row_count(monkeypatch):
    """The number of cached unit columns never exceeds the byte budget's share (at least two)."""
    sizes = []
    from collections import OrderedDict

    class Spy(OrderedDict):
        """An OrderedDict that records its largest size."""

        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            sizes.append(len(self))

    import collections

    monkeypatch.setattr(collections, "OrderedDict", Spy)
    monkeypatch.setattr(M, "_UNIT_CACHE_BYTES", 3 * 3000 * 4)
    X = _frame()
    M._collinear_hits_streaming(X, list(X.columns), len(X))
    assert sizes and max(sizes) <= 3 + 1
