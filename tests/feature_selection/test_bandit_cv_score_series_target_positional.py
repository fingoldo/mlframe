"""Regression test: the bandit CV scorer must slice a pandas target positionally, not by label."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from mlframe.feature_selection.stochastic_bandit_selection import _cv_score


def _neg_mse(y_true, y_pred):
    """Higher-is-better score."""
    return -float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def test_cv_score_series_target_with_shuffled_index_matches_ndarray():
    """A Series target with a non-positional index must give the same CV score as the same values as an ndarray."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = X["a"].to_numpy() * 2.0 + rng.normal(scale=0.1, size=n)
    perm = rng.permutation(n)
    y_series = pd.Series(y, index=perm)
    cv = KFold(n_splits=3, shuffle=True, random_state=1)
    ref = _cv_score(LinearRegression(), X, y, cv, _neg_mse)
    got = _cv_score(LinearRegression(), X, y_series, cv, _neg_mse)
    assert got == ref
