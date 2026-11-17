"""The pool-size sweep of the Ridge prefilter must not let the held-out rows choose their own features."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.ridge_forward_prefilter import nested_pool_cv_scores, ridge_coefficient_prefilter


def _noise(n=80, p=300, seed=0):
    """A pure-noise regression problem: no feature carries signal."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng.normal(size=n)


def test_small_pool_cv_score_on_noise_is_not_inflated_by_ranking_on_all_rows():
    """Ranking on all rows then cross-validating the top pool scores a noise target clearly above zero; the nested sweep does not."""
    X, y = _noise()
    Xs = StandardScaler().fit_transform(X)
    top = np.argsort(-np.abs(Ridge(alpha=1.0).fit(Xs, y).coef_))[:8]
    leaky = float(np.mean(cross_val_score(Ridge(alpha=1.0), Xs[:, top], y, cv=3, scoring="r2")))
    honest = nested_pool_cv_scores(X, y, [8], 3, False, 1.0, 0)[8]
    assert leaky > 0.05
    assert honest < 0.0


def test_prefilter_on_noise_runs_and_returns_known_names():
    """The sweep still returns a non-empty prefix of the feature names."""
    X, y = _noise(seed=1)
    names = [f"f{i}" for i in range(X.shape[1])]
    out = ridge_coefficient_prefilter(X, y, names, candidate_sizes=[8, 64, 300], cv=3)
    assert out and set(out) <= set(names)
