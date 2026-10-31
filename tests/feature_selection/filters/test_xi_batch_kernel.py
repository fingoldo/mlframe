"""The parallel Xi batch equals the sequential lexsort + ``_xi_from_order`` reference with the same generator stream."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_xi_fe import _xi_from_order, xi_correlation_batch


def _reference(X, y, seed):
    """Xi per finite column with one generator advanced in column order."""
    rng = np.random.default_rng(seed)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        if not np.isfinite(col).all():
            continue
        perm = rng.permutation(len(col))
        out[j] = _xi_from_order(y[np.lexsort((perm, col))])
    return out


def test_batch_equals_reference_with_ties_and_nan():
    """Continuous, heavily tied, binary and NaN-bearing columns, a discrete target."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 9))
    X[:, 2] = np.round(X[:, 2], 1)
    X[:, 4] = (X[:, 4] > 0).astype(float)
    X[7, 5] = np.nan
    X[:, 6] = 1.0
    y = ((X[:, 0] + rng.normal(size=n)) > 0).astype(np.int64)
    np.testing.assert_array_equal(xi_correlation_batch(X, y, random_state=5), _reference(X, y.astype(np.float64), 5))


def test_batch_with_more_columns_than_one_block():
    """More finite columns than one block: the generator stream still runs in column order."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 70))
    y = (X[:, 3] > 0).astype(np.int64)
    np.testing.assert_array_equal(xi_correlation_batch(X, y, random_state=2), _reference(X, y.astype(np.float64), 2))
