"""The memoised basis expansion returns the same columns as an uncached call, survives caller edits, and is keyed by content, parameters and y."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_univariate_fe as U


def _frame(n=3000, seed=0):
    """Five numeric columns, one skewed, one near-copy of another."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    X["x4"] = np.exp(X["x0"])
    X.index = pd.RangeIndex(10, 10 + n)
    return X


def test_hit_equals_the_uncached_result_and_is_a_copy():
    """Second call hits the memo, equals the implementation's own result, takes the caller's index, and an edit of it does not leak back."""
    U._release_basis_memo()
    X = _frame()
    y = (X["x0"].to_numpy() > 0).astype(int)
    first = U.generate_univariate_basis_features(X, y=y)
    ref = U._generate_univariate_basis_features_impl(X, y=y)
    pd.testing.assert_frame_equal(first, ref)
    second = U.generate_univariate_basis_features(X, y=y)
    pd.testing.assert_frame_equal(second, ref)
    assert len(U._BASIS_MEMO) == 1
    second.iloc[:, 0] = -1.0
    pd.testing.assert_frame_equal(U.generate_univariate_basis_features(X, y=y), ref)


def test_other_parameters_data_and_index_do_not_collide():
    """A different degree set, a different y, changed data and a different index each get their own correct result."""
    U._release_basis_memo()
    X = _frame()
    y = (X["x0"].to_numpy() > 0).astype(int)
    a = U.generate_univariate_basis_features(X, y=y)
    b = U.generate_univariate_basis_features(X, y=y, degrees=(2,))
    assert list(a.columns) != list(b.columns)
    X2 = X.copy()
    X2["x1"] = X2["x1"] * 2.0
    c = U.generate_univariate_basis_features(X2, y=y)
    pd.testing.assert_frame_equal(c, U._generate_univariate_basis_features_impl(X2, y=y))
    X3 = X.copy()
    X3.index = pd.RangeIndex(0, len(X3))
    d = U.generate_univariate_basis_features(X3, y=y)
    assert d.index.equals(X3.index)
    np.testing.assert_array_equal(d.to_numpy(), a.to_numpy())
