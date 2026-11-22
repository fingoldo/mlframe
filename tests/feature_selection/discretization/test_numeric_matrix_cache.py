"""The whole-matrix cache of the unsupervised numeric path: a repeat call on the same frame returns the same codes, as an independent copy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization import _discretization_dataset as D
from mlframe.feature_selection.filters.discretization import categorize_dataset


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Every test starts and ends with empty code caches."""
    D.clear_numeric_code_cache()
    yield
    D.clear_numeric_code_cache()


def _frame(seed=0, n=3000, p=12):
    """A float32 frame of ``p`` numeric columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, p)).astype(np.float32), columns=[f"c{i}" for i in range(p)])


def test_a_repeat_call_returns_equal_codes_from_the_cache():
    """The second call on the same frame is a cache hit (one stored matrix) and equal to the first."""
    X = _frame()
    first = categorize_dataset(X, nbins_strategy=None, dtype=np.int16)
    assert len(D._NUMERIC_MATRIX_CACHE) == 1
    second = categorize_dataset(X, nbins_strategy=None, dtype=np.int16)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])


def test_the_returned_matrix_is_a_copy_the_caller_may_write_to():
    """Writing into a result never changes what the next call returns."""
    X = _frame()
    first = categorize_dataset(X, nbins_strategy=None, dtype=np.int16)[0]
    reference = first.copy()
    first[:] = 7
    again = categorize_dataset(X, nbins_strategy=None, dtype=np.int16)[0]
    np.testing.assert_array_equal(again, reference)
    again[:] = 9
    np.testing.assert_array_equal(categorize_dataset(X, nbins_strategy=None, dtype=np.int16)[0], reference)


def test_other_parameters_and_other_values_are_other_entries():
    """A different bin count, a different dtype or one changed value is not served from the matrix of another call."""
    X = _frame()
    four = categorize_dataset(X, nbins_strategy=None, n_bins=4, dtype=np.int16)[0]
    eight = categorize_dataset(X, nbins_strategy=None, n_bins=8, dtype=np.int16)[0]
    assert int(eight.max()) > int(four.max())
    low_row = int(np.flatnonzero(four[:, 0] == 0)[0])
    Y = X.copy()
    Y.iloc[low_row, 0] = np.float32(1e6)
    changed = categorize_dataset(Y, nbins_strategy=None, n_bins=4, dtype=np.int16)[0]
    assert four[low_row, 0] == 0 and changed[low_row, 0] == int(four[:, 0].max())  # the edited value left the lowest bin: not the cached matrix of X


def test_a_matrix_over_the_budget_is_not_cached(monkeypatch):
    """With a budget below the matrix size nothing is stored and the result is still right."""
    monkeypatch.setattr(D, "_NUMERIC_MATRIX_CACHE_MAX_BYTES", 1000)
    X = _frame()
    out = categorize_dataset(X, nbins_strategy=None, dtype=np.int16)[0]
    assert len(D._NUMERIC_MATRIX_CACHE) == 0
    np.testing.assert_array_equal(categorize_dataset(X, nbins_strategy=None, dtype=np.int16)[0], out)


def test_only_the_byte_budget_of_matrices_is_kept(monkeypatch):
    """Two frames of 72 kB each under a 100 kB budget: the older matrix is evicted, the bytes counter follows."""
    monkeypatch.setattr(D, "_NUMERIC_MATRIX_CACHE_MAX_BYTES", 100_000)
    for seed in (1, 2):
        categorize_dataset(_frame(seed=seed, n=3000, p=12), nbins_strategy=None, dtype=np.int16)
    assert len(D._NUMERIC_MATRIX_CACHE) == 1
    assert D._NUMERIC_MATRIX_CACHE_BYTES == 3000 * 12 * 2
