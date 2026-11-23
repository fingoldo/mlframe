"""A hit of the basis memo costs no memory under copy-on-write and still cannot be corrupted through (PLAN-10).

Every FE family that scores the basis expansion asks ``generate_univariate_basis_features`` for the same columns of the same frame (17 calls, 4 distinct inputs in an every-family fit); each hit used to be a
``deep=True`` copy (687 MB held at 100000 x 1000). Under copy-on-write a shallow copy shares the blocks until a caller writes, and the write copies that block only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _orthogonal_univariate_fe as U
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_basis_memo as M


def _frame(n: int = 800, seed: int = 0):
    """Four Gaussian columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, 4)), columns=list("abcd"))


@pytest.fixture(autouse=True)
def _clean_memo():
    """An empty memo before and after each test."""
    M._BASIS_MEMO.clear()
    yield
    M._BASIS_MEMO.clear()


def test_a_hit_returns_the_same_values_and_a_write_does_not_reach_the_memo():
    """Two calls give equal frames; scribbling on the first hit changes neither the stored columns nor the next hit."""
    X = _frame()
    first = U.generate_univariate_basis_features(X, degrees=(2, 3), basis="hermite")
    assert M._BASIS_MEMO, "the expansion must have been memoised for this test to mean anything"
    hit = U.generate_univariate_basis_features(X, degrees=(2, 3), basis="hermite")
    pd.testing.assert_frame_equal(first, hit)
    col = hit.columns[0]
    expected = first[col].to_numpy().copy()
    hit.loc[:, col] = -1.0e9  # a caller editing its result in place
    again = U.generate_univariate_basis_features(X, degrees=(2, 3), basis="hermite")
    assert np.array_equal(again[col].to_numpy(), expected), "an edit of one hit must not reach the stored columns"
    assert np.array_equal(first[col].to_numpy(), expected)


def test_under_copy_on_write_a_hit_shares_the_blocks():
    """The point of the change: the hit is a view of the stored data (no copy) when copy-on-write is on."""
    if not U._copy_on_write_active():
        pytest.skip("pandas without copy-on-write keeps the deep copy by design")
    X = _frame()
    U.generate_univariate_basis_features(X, degrees=(2, 3), basis="hermite")
    stored = next(iter(M._BASIS_MEMO.values()))
    hit = U.generate_univariate_basis_features(X, degrees=(2, 3), basis="hermite")
    assert np.shares_memory(stored.iloc[:, 0].to_numpy(), hit.iloc[:, 0].to_numpy())


def test_a_write_to_the_first_result_does_not_reach_the_stored_columns():
    """The stored frame is a shallow copy under copy-on-write too: editing the FIRST (computed) result in place must leave the memo, and so the next hit, intact."""
    X = _frame()
    first = U.generate_univariate_basis_features(X, degrees=(2, 3), basis="hermite")
    col = first.columns[0]
    expected = first[col].to_numpy().copy()
    first.loc[:, col] = -1.0e9
    again = U.generate_univariate_basis_features(X, degrees=(2, 3), basis="hermite")
    assert np.array_equal(again[col].to_numpy(), expected)
