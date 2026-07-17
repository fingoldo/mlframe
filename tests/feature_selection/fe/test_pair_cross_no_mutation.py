"""Regression: generate_pair_cross_basis_features must NOT mutate the caller's DataFrame.

Found by the hidden-flaw audit (2026-06-22): the NaN mean-fill used
``np.asarray(X[col].to_numpy(), float64)`` (an alias of the DataFrame's float64 block) + ``np.copyto``,
which overwrote the caller's NaNs in place -- corrupting any downstream missingness-FE. Fixed by copying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import (
    generate_pair_cross_basis_features,
)


def test_pair_cross_does_not_mutate_caller_nans():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    a[::37] = np.nan  # genuine missingness the caller may key downstream FE on
    b[::53] = np.nan
    X = pd.DataFrame({"a": a.astype(np.float64), "b": b.astype(np.float64)})
    before_a = X["a"].to_numpy(copy=True)
    before_b = X["b"].to_numpy(copy=True)
    before_nan_a = int(X["a"].isna().sum())
    before_nan_b = int(X["b"].isna().sum())

    generate_pair_cross_basis_features(X, [("a", "b")], max_degree=2)

    # The caller's NaNs must survive untouched.
    assert int(X["a"].isna().sum()) == before_nan_a > 0
    assert int(X["b"].isna().sum()) == before_nan_b > 0
    np.testing.assert_array_equal(X["a"].to_numpy(), before_a)
    np.testing.assert_array_equal(X["b"].to_numpy(), before_b)
