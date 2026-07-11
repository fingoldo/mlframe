"""biz_value test for ``feature_selection.zero_importance_pruning.iterative_zero_importance_pruning``.

Source: dd_2nd_pover-t-tests.md -- "The `find_exclude` method iteratively retrains a model, drops zero-
importance features, and re-evaluates 5-fold CV log loss, repeating until no further features can be dropped,
then keeps the feature-exclusion set with lowest CV log loss." On a small regression with many pure-noise
columns, a tree ensemble's native importances should assign near-zero importance to most noise columns;
batch-dropping them each round should shrink the feature set substantially and improve held-out R2 versus the
full feature set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from mlframe.feature_selection.zero_importance_pruning import iterative_zero_importance_pruning


def _make_overparameterized_regression(n: int, n_signal: int, n_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    X_signal = rng.normal(size=(n, n_signal))
    beta = rng.normal(size=n_signal) * 3.0
    y = X_signal @ beta + rng.normal(scale=0.3, size=n)
    X_noise = rng.normal(size=(n, n_noise))
    columns = [f"s{i}" for i in range(n_signal)] + [f"n{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([X_signal, X_noise]), columns=columns)
    return X, y


def test_biz_val_zero_importance_pruning_shrinks_features_and_improves_r2():
    X, y = _make_overparameterized_regression(n=150, n_signal=3, n_noise=40, seed=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    cv = KFold(n_splits=4, shuffle=True, random_state=0)

    estimator = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0)
    survivors = iterative_zero_importance_pruning(estimator, Xtr, ytr, scoring=r2_score, cv=cv, importance_threshold=0.002)

    assert len(survivors) < X.shape[1] * 0.5, f"expected substantial pruning of {X.shape[1]} features, got {len(survivors)} survivors"

    model_full = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0).fit(Xtr, ytr)
    model_selected = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0).fit(Xtr[survivors], ytr)
    r2_full = float(r2_score(yte, model_full.predict(Xte)))
    r2_selected = float(r2_score(yte, model_selected.predict(Xte[survivors])))

    assert r2_selected > r2_full, f"expected pruning to improve held-out R2, got selected={r2_selected:.4f} full={r2_full:.4f}"


def test_zero_importance_pruning_never_returns_empty_feature_set():
    X, y = _make_overparameterized_regression(n=100, n_signal=2, n_noise=10, seed=2)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    estimator = RandomForestRegressor(n_estimators=50, random_state=0)
    survivors = iterative_zero_importance_pruning(estimator, X, y, scoring=r2_score, cv=cv, importance_threshold=1.0)
    assert len(survivors) >= 1


def test_zero_importance_pruning_keeps_all_when_nothing_is_zero_importance():
    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X["a"] * 2 + X["b"] * 2 + X["c"] * 2 + rng.normal(scale=0.05, size=n)).to_numpy()
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    estimator = RandomForestRegressor(n_estimators=50, random_state=0)
    survivors = iterative_zero_importance_pruning(estimator, X, y, scoring=r2_score, cv=cv, importance_threshold=0.0)
    assert set(survivors) == {"a", "b", "c"}
