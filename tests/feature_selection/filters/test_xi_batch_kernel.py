"""The parallel Xi batch equals the numpy ``lexsort`` + ``_xi_from_order`` reference with the counter-hash tie keys."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_xi_fe import xi_correlation_batch
from tests.feature_selection._xi_reference import xi_reference


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
    Xc = X.astype(np.float32)  # the batch reads its columns at the criterion dtype
    np.testing.assert_array_equal(xi_correlation_batch(X, y, random_state=5), xi_reference(Xc, y.astype(np.float64), 5))


def test_batch_with_more_columns_than_one_block():
    """More finite columns than one block: the tie keys follow the column number."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 70))
    X[:, 10] = np.round(X[:, 10])
    y = (X[:, 3] > 0).astype(np.int64)
    Xc = X.astype(np.float32)
    np.testing.assert_array_equal(xi_correlation_batch(X, y, random_state=2), xi_reference(Xc, y.astype(np.float64), 2))
