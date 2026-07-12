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
from sklearn.inspection import permutation_importance
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


def test_zero_importance_pruning_omitted_importance_fn_is_bit_identical_to_native():
    """``importance_fn`` is strictly opt-in: omitting it must reproduce the pre-existing native-importance path exactly."""
    X, y = _make_overparameterized_regression(n=120, n_signal=2, n_noise=15, seed=4)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    estimator = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=0)

    survivors_default = iterative_zero_importance_pruning(estimator, X, y, scoring=r2_score, cv=cv, importance_threshold=0.01)
    survivors_explicit_none = iterative_zero_importance_pruning(
        estimator, X, y, scoring=r2_score, cv=cv, importance_threshold=0.01, importance_fn=None
    )

    assert survivors_default == survivors_explicit_none


def _make_cardinality_bias_regression(n: int, seed: int):
    """Two genuinely informative low-cardinality (binary) features plus one high-cardinality pure-noise column.

    Random-forest native (MDI) importance is documented to over-rate high-cardinality/continuous columns
    relative to low-cardinality ones -- ``noise_hc`` offers many split thresholds to overfit small-sample
    noise even though it carries zero true signal, so it can out-score genuinely predictive binary features.
    """
    rng = np.random.default_rng(seed)
    b1 = rng.integers(0, 2, size=n).astype(float)
    b2 = rng.integers(0, 2, size=n).astype(float)
    y = 2.0 * b1 + 2.0 * b2 + rng.normal(scale=1.0, size=n)
    noise_hc = rng.normal(size=n) * 1000.0 + np.arange(n) * 1e-3  # near-unique continuous values, no signal
    X = pd.DataFrame({"b1": b1, "b2": b2, "noise_hc": noise_hc})
    return X, y


def _held_out_permutation_importance_fn(Xval: pd.DataFrame, yval: np.ndarray):
    """Build an ``importance_fn`` scoring permutation importance on a fixed held-out set (avoids the
    training-set overfit artifact that also inflates permutation importance for high-cardinality noise)."""

    def _importance_fn(fitted_estimator, X_round: pd.DataFrame, y_round: np.ndarray) -> np.ndarray:
        result = permutation_importance(fitted_estimator, Xval[X_round.columns], yval, n_repeats=30, random_state=0)
        return np.asarray(result.importances_mean)

    return _importance_fn


def test_biz_val_zero_importance_pruning_permutation_importance_fn_beats_native_on_high_cardinality_noise():
    X, y = _make_cardinality_bias_regression(n=300, seed=7)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, random_state=0)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    estimator = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=0)
    threshold = 0.1

    survivors_native = iterative_zero_importance_pruning(estimator, Xtr, ytr, scoring=r2_score, cv=cv, importance_threshold=threshold)
    survivors_permutation = iterative_zero_importance_pruning(
        estimator,
        Xtr,
        ytr,
        scoring=r2_score,
        cv=cv,
        importance_threshold=threshold,
        importance_fn=_held_out_permutation_importance_fn(Xval, yval),
    )

    assert "noise_hc" in survivors_native, "native MDI importance is expected to over-rate the high-cardinality noise column here"
    assert "noise_hc" not in survivors_permutation, "held-out permutation importance should correctly prune the pure-noise column"
    assert set(survivors_permutation) == {"b1", "b2"}
    assert len(survivors_permutation) < len(survivors_native)
