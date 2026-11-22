"""``ksg_mi_columns`` returns what sklearn's ``mutual_info_classif`` / ``mutual_info_regression`` return for the same arguments, so the call sites that dropped sklearn keep their numbers."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._ksg_dispatch import ksg_mi_columns


def _data(n, p, seed):
    """Continuous columns with a dependence on both a class label and a continuous target."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    score = X[:, 0] * X[:, 1] + 0.5 * X[:, 2]
    return X, (score + 0.4 * rng.standard_normal(n) > 0).astype(np.int64), score + 0.3 * rng.standard_normal(n)


@pytest.mark.parametrize("n,p,k", [(300, 4, 3), (1500, 7, 3), (1500, 7, 5)])
def test_classification_matches_sklearn(n, p, k):
    """Discrete target, a batch of columns."""
    from sklearn.feature_selection import mutual_info_classif

    X, yc, _ = _data(n, p, seed=n + k)
    expected = mutual_info_classif(X, yc, n_neighbors=k, random_state=3, discrete_features=False)
    np.testing.assert_allclose(ksg_mi_columns(X, yc, discrete_target=True, n_neighbors=k, random_state=3), expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("n,p,k", [(300, 4, 3), (1500, 7, 3), (1500, 7, 5)])
def test_regression_matches_sklearn(n, p, k):
    """Continuous target, a batch of columns."""
    from sklearn.feature_selection import mutual_info_regression

    X, _, yr = _data(n, p, seed=n + k)
    expected = mutual_info_regression(X, yr, n_neighbors=k, random_state=3, discrete_features=False)
    np.testing.assert_allclose(ksg_mi_columns(X, yr, discrete_target=False, n_neighbors=k, random_state=3), expected, rtol=0, atol=1e-9)


def test_a_single_column_and_a_pair_work_like_sklearn():
    """The one-column and two-column shapes of the call sites (``x.reshape(-1, 1)``, ``column_stack([a, b])``)."""
    from sklearn.feature_selection import mutual_info_regression

    X, _, yr = _data(800, 3, seed=1)
    for cols in ([0], [0, 1]):
        expected = mutual_info_regression(X[:, cols], yr, n_neighbors=3, random_state=7, discrete_features=False)
        np.testing.assert_allclose(ksg_mi_columns(X[:, cols], yr, discrete_target=False, n_neighbors=3, random_state=7), expected, rtol=0, atol=1e-9)
    assert ksg_mi_columns(X[:, 0], yr, discrete_target=False, n_neighbors=3, random_state=7).shape == (1,)
