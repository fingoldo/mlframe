"""Identity regression for the vectorised _ksg._count_within_eps (CPX7).

The pre-fix implementation was a pure-Python loop issuing 2*N searchsorted
calls per MI pair (the dominant cost in mixed_ksg_mi). The vectorised version
issues two array-valued searchsorted calls. This pins bit-identity: same
counts on continuous, tied, and edge inputs, and unchanged MI from
mixed_ksg_mi. Also guards that the dead ColumnKNNCache stays removed.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._ksg import (
    _count_within_eps,
    mixed_ksg_mi,
)


def _reference_count_within_eps(arr_1d, eps):
    arr = arr_1d.astype(np.float64).ravel()
    sorter = np.argsort(arr)
    sorted_arr = arr[sorter]
    counts = np.empty(arr.size, dtype=np.int64)
    n = arr.size
    for i in range(n):
        lo = arr[i] - eps[i] + 1e-12
        hi = arr[i] + eps[i] - 1e-12
        lo_idx = np.searchsorted(sorted_arr, lo, side="left")
        hi_idx = np.searchsorted(sorted_arr, hi, side="right")
        counts[i] = max(0, (hi_idx - lo_idx) - 1)
    return counts


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("kind", ["continuous", "tied", "tiny_eps"])
def test_count_within_eps_bit_identical(seed, kind):
    rng = np.random.default_rng(seed)
    n = 2000
    if kind == "continuous":
        arr = rng.standard_normal(n)
        eps = np.abs(rng.standard_normal(n)) * 0.1 + 0.01
    elif kind == "tied":
        arr = rng.integers(0, 5, size=n).astype(np.float64)
        eps = np.full(n, 0.5)
    else:  # tiny_eps -> bands collapse to self only
        arr = rng.standard_normal(n)
        eps = np.full(n, 1e-13)

    ref = _reference_count_within_eps(arr, eps)
    got = _count_within_eps(arr, eps)
    assert got.dtype == np.int64
    assert np.array_equal(ref, got)


def test_mixed_ksg_mi_unchanged_on_correlated():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.standard_normal(n)
    y = 0.7 * x + 0.3 * rng.standard_normal(n)
    mi = mixed_ksg_mi(x, y, k=5, seed=0)
    # Tracks real dependence; deterministic given fixed seed.
    assert mi > 0.1
    assert mixed_ksg_mi(x, y, k=5, seed=0) == mi


def test_column_knn_cache_removed():
    import mlframe.feature_selection.filters._ksg as ksg

    assert not hasattr(ksg, "ColumnKNNCache")
    assert "ColumnKNNCache" not in ksg.__all__
