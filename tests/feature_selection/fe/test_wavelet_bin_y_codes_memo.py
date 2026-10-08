"""``_bin_y_codes`` is memoised on the content of ``y`` and keeps its documented binning."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _wavelet_basis_fe as W


def _reference(y, nbins=10):
    """The pre-memo implementation, verbatim."""
    y = np.asarray(y).ravel()
    if np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 20:
        return y.astype(np.int64)
    if np.unique(y).size <= 20:
        uy = np.unique(y)
        return np.searchsorted(uy, y)
    edges_y = np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
    return np.asarray(np.digitize(y, edges_y))


def test_memoised_codes_equal_the_reference_for_every_branch():
    """Integer labels, few distinct floats and a continuous target all match, on first call and on a memo hit."""
    rng = np.random.default_rng(0)
    cases = [rng.integers(0, 5, 3000), rng.integers(0, 4, 3000).astype(np.float64) * 0.5, rng.standard_normal(5000)]
    for y in cases:
        for _ in range(2):  # miss, then hit
            np.testing.assert_array_equal(W._bin_y_codes(y), _reference(y))


def test_a_memo_hit_does_not_share_a_mutable_array_and_distinguishes_content():
    """Mutating a returned array cannot corrupt the memo; a different y (same shape) gets its own codes."""
    rng = np.random.default_rng(1)
    y1, y2 = rng.standard_normal(4000), rng.standard_normal(4000)
    first = W._bin_y_codes(y1)
    first[:] = -1
    np.testing.assert_array_equal(W._bin_y_codes(y1), _reference(y1))
    np.testing.assert_array_equal(W._bin_y_codes(y2), _reference(y2))


def test_distinct_count_shortcut_agrees_with_the_full_unique():
    """A prefix with many distinct values decides the answer; a discrete column still gets the full count, on both sides of the limit."""
    from mlframe.feature_selection.filters._y_encoding import _has_more_distinct_than

    rng = np.random.default_rng(3)
    cont = rng.standard_normal(200_000)
    few = rng.integers(0, 5, 200_000).astype(np.float64)
    late = np.zeros(200_000)
    late[150_000:] = np.arange(50_000)  # the prefix is constant, the distinct values only appear later
    for arr, limit in ((cont, 32), (few, 32), (few, 3), (late, 32), (late, 60_000)):
        assert _has_more_distinct_than(arr, limit) == (np.unique(arr).size > limit)
