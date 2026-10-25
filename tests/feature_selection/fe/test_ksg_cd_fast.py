"""The numba Ross estimator agrees with sklearn's mutual_info_classif for continuous columns."""
import numpy as np
from sklearn.feature_selection import mutual_info_classif

from mlframe.feature_selection.filters._ksg_cd_fast import ksg_mi_discrete_target


def test_matches_sklearn_with_ties_constant_column_and_singleton_class():
    """Rounded (tied) column, constant column and a one-row class all reproduce sklearn's values."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 5))
    X[:, 3] = np.round(X[:, 3] * 3)
    X[:, 4] = 0.0
    y = rng.integers(0, 4, n)
    y[0] = 99
    ref = mutual_info_classif(X, y, discrete_features=False, n_neighbors=3, random_state=5)
    got = ksg_mi_discrete_target(X, y, n_neighbors=3, random_state=5)
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_detects_planted_dependence():
    """A column that determines the class scores far above an independent one."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    y = (x > 0).astype(int)
    got = ksg_mi_discrete_target(np.c_[x, rng.normal(size=4000)], y, n_neighbors=3, random_state=0)
    assert got[0] > 0.5 and got[1] < 0.05
