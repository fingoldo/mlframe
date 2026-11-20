"""Regression tests for the forward_select NaN-score pick and the greedy_backward_elimination repeat-fold policy."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_selection.forward_select import forward_select
from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination


def test_forward_select_never_picks_a_nan_scored_candidate():
    """A candidate whose CV score is NaN must not win the round, whatever its position."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"bad": rng.normal(size=200), "good": rng.normal(size=200)})
    y = X["good"].to_numpy() * 2.0 + 0.01 * rng.normal(size=200)

    def scorer(est, Xs, ys):
        """NaN whenever the column ``bad`` is in the subset, else R2."""
        if "bad" in getattr(Xs, "columns", []):
            return float("nan")
        return float(est.score(Xs, ys))

    sel = forward_select(X, y, LinearRegression, scoring=scorer, cv=3, max_features=1)
    assert sel == ["good"]


class _SpyRegressor(LinearRegression):
    """LinearRegression that records the max train index seen in each fit."""

    seen: list = []

    def fit(self, X, y, sample_weight=None):
        """Record the train/test layout, then fit."""
        _SpyRegressor.seen.append(len(X))
        return super().fit(X, y, sample_weight)


def test_greedy_backward_repeats_do_not_replace_a_time_series_split():
    """With a non-reshufflable cv (TimeSeriesSplit) n_repeats must not silently swap in shuffled KFold folds."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=120)
    _SpyRegressor.seen = []
    greedy_backward_elimination(_SpyRegressor(), X, y, r2_score, cv=TimeSeriesSplit(n_splits=3), min_features=3, n_repeats=4)
    sizes = sorted(set(_SpyRegressor.seen))
    assert sizes == [30, 60, 90]
