"""``quantile_bin_columns`` returns, per column, the codes ``_quantile_bin`` gives it (continuous, tied, two-valued, constant and NaN-bearing columns)."""

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin, quantile_bin_columns


def test_columns_equal_the_scalar_binner_per_column():
    """Every column of a mixed frame gets the scalar binner's codes; the NaN column takes the scalar path."""
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.normal(size=(n, 9)).astype(np.float32)
    X[:, 1] = np.round(X[:, 1], 1)
    X[:, 2] = 3.0
    X[:, 3] = (X[:, 3] > 0).astype(np.float32)
    X[:, 4] = rng.integers(0, 4, n).astype(np.float32)
    X[17, 5] = np.nan
    X[:, 6] = np.exp(X[:, 6])
    got = quantile_bin_columns(X, 10)
    assert len(got) == X.shape[1]
    for j in range(X.shape[1]):
        np.testing.assert_array_equal(got[j], _quantile_bin(np.ascontiguousarray(X[:, j]), nbins=10), err_msg=f"column {j}")
