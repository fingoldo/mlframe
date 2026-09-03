"""Pytest port of legacy unittest_arrays.py.

Validity tests only. Timing/perf comparisons removed (flaky). For microbenchmarks,
use pytest-benchmark via the ``@pytest.mark.benchmark`` marker.
"""

import numpy as np
import pytest

import mlframe.core.arrays as m

MIN_ELEM = 50
MAX_ELEM = 1000
ARR_SIZE = 100_000  # reduced from 1_000_000 for test speed


@pytest.fixture
def rng():
    """Returns ``np.random.default_rng(0)``."""
    return np.random.default_rng(0)


def _baseline_argsort(vals):
    """Returns ``np.argsort(vals)``."""
    return np.argsort(vals)


def _baseline_argsort_indexed(vals, indices):
    """Returns ``indices[np.argsort(fr)]`` (after 1 setup step)."""
    fr = vals[indices]
    return indices[np.argsort(fr)]


def test_arrayMinMax(rng):
    """ArrayMinMax."""
    np.random.seed(0)
    assert m.arrayMinMax(np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)) == (MIN_ELEM, MAX_ELEM - 1)
    assert m.arrayMinMax(np.arange(20), 10, 15) == (10, 14)


def test_arrayMinMaxParallel(rng):
    """ArrayMinMaxParallel."""
    np.random.seed(0)
    assert m.arrayMinMaxParallel(np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)) == (MIN_ELEM, MAX_ELEM - 1)
    assert m.arrayMinMaxParallel(np.arange(20), 10, 15) == (10, 14)


def test_arrayCountingSort(rng):
    """ArrayCountingSort."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    assert (m.arrayCountingSort(x, MAX_ELEM) == np.sort(x)).all()


def test_arrayCountingArgSort_whole(rng):
    """ArrayCountingArgSort whole."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    assert (x[m.arrayCountingArgSort(x, MAX_ELEM)] == x[np.argsort(x)]).all()


def test_arrayCountingArgSort_indexed(rng):
    """ArrayCountingArgSort indexed."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    indices = np.random.choice(x, ARR_SIZE // 5, replace=False)
    assert (x[m.arrayCountingArgSort(x, MAX_ELEM, indices)] == x[indices[np.argsort(x[indices])]]).all()


def test_arrayCountingArgSortThreaded_whole(rng):
    """ArrayCountingArgSortThreaded whole."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    assert (x[m.arrayCountingArgSortThreaded(x, MAX_ELEM)] == x[np.argsort(x)]).all()


def test_arrayCountingArgSortThreaded_indexed(rng):
    """ArrayCountingArgSortThreaded indexed."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    indices = np.random.choice(x, ARR_SIZE // 5, replace=False)
    assert (x[m.arrayCountingArgSortThreaded(x, MAX_ELEM, indices)] == x[indices[np.argsort(x[indices])]]).all()


def test_arrayCountingArgSortAndUniqueValues_whole(rng):
    """ArrayCountingArgSortAndUniqueValues whole."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    assert (x[m.arrayCountingArgSortAndUniqueValues(x, MAX_ELEM)[2]] == x[np.argsort(x)]).all()


def test_arrayCountingArgSortAndUniqueValues_indexed(rng):
    """ArrayCountingArgSortAndUniqueValues indexed."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    indices = np.random.choice(x, ARR_SIZE // 5, replace=False)
    assert (x[m.arrayCountingArgSortAndUniqueValues(x, MAX_ELEM, indices)[2]] == x[indices[np.argsort(x[indices])]]).all()


def test_arrayCountingArgSortAndUniqueValuesThreaded_whole(rng):
    """Threaded unique-values variant matches the serial reference (argsorted output + unique values/offsets)."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    ref_unique, ref_offsets, ref_argsorted = m.arrayCountingArgSortAndUniqueValues(x, MAX_ELEM)
    unique, offsets, argsorted = m.arrayCountingArgSortAndUniqueValuesThreaded(x, MAX_ELEM)
    assert (unique == ref_unique).all()
    assert (offsets == ref_offsets).all()
    assert (x[argsorted] == x[ref_argsorted]).all()


def test_arrayCountingArgSortAndUniqueValuesThreaded_indexed(rng):
    """Threaded unique-values variant, mask-restricted, matches the serial reference."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE)
    indices = np.random.choice(x, ARR_SIZE // 5, replace=False)
    ref_unique, ref_offsets, ref_argsorted = m.arrayCountingArgSortAndUniqueValues(x, MAX_ELEM, indices)
    unique, offsets, argsorted = m.arrayCountingArgSortAndUniqueValuesThreaded(x, MAX_ELEM, indices)
    assert (unique == ref_unique).all()
    assert (offsets == ref_offsets).all()
    assert (x[argsorted] == x[ref_argsorted]).all()


def test_npnbArrayMinMax(rng):
    """npnbArrayMinMax matches plain numpy min/max on finite input."""
    np.random.seed(0)
    x = np.random.randint(MIN_ELEM, MAX_ELEM, ARR_SIZE).astype(np.float64)
    assert m.npnbArrayMinMax(x) == (x.min(), x.max())


class TestTopkByPartition:
    """mlframe.core.arrays.topk_by_partition -- pure numpy, no njit, previously untested."""

    def test_descending_default_matches_full_sort(self):
        """Default (ascending=False): top-k largest, sorted descending, matches a full argsort reference."""
        rng = np.random.default_rng(1)
        arr = rng.standard_normal(200)
        k = 10
        ind, val = m.topk_by_partition(arr, k)
        ref_ind = np.argsort(-arr)[:k]
        assert (ind == ref_ind).all()
        assert np.allclose(val, arr[ref_ind])
        assert list(val) == sorted(val, reverse=True)

    def test_ascending_smallest(self):
        """ascending=True: top-k smallest, sorted ascending."""
        rng = np.random.default_rng(2)
        arr = rng.standard_normal(200)
        k = 10
        ind, val = m.topk_by_partition(arr, k, ascending=True)
        ref_ind = np.argsort(arr)[:k]
        assert (ind == ref_ind).all()
        assert np.allclose(val, arr[ref_ind])
        assert list(val) == sorted(val)

    def test_does_not_mutate_input(self):
        """The caller's array must be untouched (the old implementation did an in-place `arr *= -1`)."""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        original = arr.copy()
        m.topk_by_partition(arr, 3)
        assert np.array_equal(arr, original)

    def test_k_larger_than_array_clamps(self):
        """k larger than the array size clamps to the full array, not an out-of-bounds partition."""
        arr = np.array([3.0, 1.0, 4.0])
        ind, val = m.topk_by_partition(arr, 10)
        assert len(ind) == 3
        assert list(val) == sorted(arr, reverse=True)

    def test_k_zero_returns_empty(self):
        """k=0 returns empty index/value arrays, not an error."""
        arr = np.array([3.0, 1.0, 4.0])
        ind, val = m.topk_by_partition(arr, 0)
        assert len(ind) == 0
        assert len(val) == 0

    def test_axis_parameter_2d(self):
        """axis=1: top-k per row, matching the per-row full-sort reference."""
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((5, 20))
        k = 4
        ind, val = m.topk_by_partition(arr, k, axis=1)
        assert ind.shape == (5, k)
        assert val.shape == (5, k)
        for row in range(5):
            ref_ind = np.argsort(-arr[row])[:k]
            assert (ind[row] == ref_ind).all()

    def test_axis_none_flattens(self):
        """axis=None: top-k over the flattened array, matching a flat-argsort reference."""
        rng = np.random.default_rng(4)
        arr = rng.standard_normal((4, 5))
        k = 6
        ind, val = m.topk_by_partition(arr, k, axis=None)
        ref_ind = np.argsort(-arr.ravel())[:k]
        assert (ind == ref_ind).all()
        assert np.allclose(val, arr.ravel()[ref_ind])
