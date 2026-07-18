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
    """Helper that rng."""
    return np.random.default_rng(0)


def _baseline_argsort(vals):
    """Helper that baseline argsort."""
    return np.argsort(vals)


def _baseline_argsort_indexed(vals, indices):
    """Helper that baseline argsort indexed."""
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
