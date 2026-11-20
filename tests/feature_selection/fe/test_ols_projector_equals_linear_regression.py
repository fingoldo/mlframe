"""The cached SVD-projector residual of the pure-form non-separability gate equals the ``LinearRegression`` residual it replaced, with a rank-deficient basis too."""

import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters import _fe_pure_form_retention as fpr


@pytest.mark.parametrize("deficient", [False, True])
def test_projector_residual_equals_linear_regression(deficient):
    """Several targets against one basis: the residuals agree to 1e-9, and a duplicated basis column (rank deficiency) is handled as the lstsq rank cut handles it."""
    rng = np.random.default_rng(0)
    n = 800
    X = rng.standard_normal((n, 9))
    if deficient:
        X[:, 4] = X[:, 1] * 2.0
        X[:, 7] = X[:, 0] - X[:, 2]
    Xs = StandardScaler().fit_transform(X)
    basis = fpr._ols_projector_basis(Xs)
    assert basis.shape[1] == (7 if deficient else 9)
    for seed in range(5):
        y = np.random.default_rng(seed).standard_normal(n) + 0.3 * X[:, 0] ** 2
        lr = LinearRegression().fit(Xs, y)
        want = y - lr.predict(Xs)
        yc = y - y.mean()
        got = yc - basis @ (basis.T @ yc)
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=1e-9)
