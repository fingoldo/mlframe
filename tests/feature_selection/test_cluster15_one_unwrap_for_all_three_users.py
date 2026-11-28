"""Cluster 15 regression: family detection, signed-coef extraction and the 'auto' getter unwrap wrappers through one helper (audit 2026-11-20, finding 15-F13)."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from mlframe.feature_selection.wrappers._helpers_importance_agg import detect_estimator_family, get_signed_linear_coef, unwrap_estimator
from mlframe.feature_selection.wrappers._importance_methods import _unwrap_estimator


def _fitted_search():
    """A fitted GridSearchCV around Ridge, where the signed coefficient of column 0 is negative."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = -3.0 * X[:, 0] + 0.1 * rng.normal(size=80)
    return GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3).fit(X, y), X


def test_all_three_users_see_through_a_fitted_search_cv():
    """The family is 'linear', the signed coefficient is found (negative for column 0), and both unwrap entry points return the inner Ridge."""
    search, X = _fitted_search()
    assert unwrap_estimator(search) is search.best_estimator_
    assert _unwrap_estimator(search) is search.best_estimator_
    assert detect_estimator_family(search) == "linear"
    coefs = get_signed_linear_coef(search, ["a", "b", "c"], train_data=X)
    assert coefs is not None and coefs["a"] < 0
