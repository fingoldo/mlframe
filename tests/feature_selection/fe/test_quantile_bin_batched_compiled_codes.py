"""``quantile_bin_batched`` takes the compiled quantile-codes block for columns with distinct edges and keeps the numpy path for tied ones, with the same codes."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_shared import quantile_bin_batched


def _reference(arr, nbins):
    """The numpy edges, dedup and searchsorted convention."""
    n, k = arr.shape
    out = np.zeros((n, k), np.int64)
    edges = np.quantile(arr, np.linspace(0, 1, nbins + 1), axis=0)
    for j in range(k):
        u = np.unique(edges[:, j])
        if u.size <= 2:
            if u.size == 2:
                out[:, j] = arr[:, j] >= u[1]
            continue
        out[:, j] = np.searchsorted(u[1:-1], arr[:, j], side="right")
    return out


def test_mixed_distinct_and_tied_columns_match_the_numpy_convention():
    """Continuous, rounded, few-level and float32 columns give the numpy codes."""
    r = np.random.default_rng(0)
    a = np.column_stack([r.normal(size=5000), np.round(r.normal(size=5000), 1), r.integers(0, 4, 5000).astype(float), r.normal(size=5000), np.zeros(5000)])
    assert np.array_equal(quantile_bin_batched(a, 10), _reference(a, 10))
    b = r.normal(size=(3000, 4)).astype(np.float32)
    assert np.array_equal(quantile_bin_batched(b, 10), _reference(b, 10))
