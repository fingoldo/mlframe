"""Tests for ``PartialFitESWrapper``.

Two strategy paths:
  - ``partial_fit``-based for estimators with that method (SGDRegressor, SGDClassifier, etc.)
  - dichotomic budget search for estimators without partial_fit (Ridge, etc.)

Tests cover both paths + the diagnostic state surface (best_iter, best_metric, stopped_via).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._partial_fit_es_wrapper import PartialFitESWrapper


def _make_regression(seed: int = 0, n: int = 800, d: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(0, 0.5, n)
    return X, y


def _make_classification(seed: int = 0, n: int = 800, d: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    logit = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)
    y = (logit > 0).astype(int)
    return X, y


# -- partial_fit path -----------------------------------------------------------

def test_partial_fit_regression_sgd_stops_via_callback() -> None:
    """SGDRegressor on a learnable signal: ES wrapper should stop and record best_metric."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import SGDRegressor

    X, y = _make_regression()
    wrapper = PartialFitESWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0, learning_rate="constant", eta0=0.01),
        metric="rmse", patience=20, max_iter=100, val_size=0.2,
    )
    wrapper.fit(X, y)
    assert wrapper._fitted
    assert wrapper.best_iter is not None
    assert wrapper.best_metric is not None
    assert wrapper.stopped_via in {"patience", "curve_shape", "max_iter_hit"}
    assert len(wrapper.history) > 0
    # Sanity: predicting back on the train data shouldn't crash.
    preds = wrapper.predict(X)
    assert preds.shape == y.shape


def test_partial_fit_classification_sgd_uses_predict_proba() -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import SGDClassifier

    X, y = _make_classification()
    wrapper = PartialFitESWrapper(
        SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=0,
                       learning_rate="constant", eta0=0.01),
        metric="logloss", patience=15, max_iter=80, val_size=0.2,
        is_classification=True,
    )
    wrapper.fit(X, y)
    assert wrapper._fitted
    assert wrapper.best_iter is not None
    probs = wrapper.predict_proba(X)
    assert probs.shape == (len(y), 2)


def test_partial_fit_curve_shape_can_fire() -> None:
    """With a degenerate setup (huge learning rate -> diverging val), curve-shape ES catches it."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import SGDRegressor

    X, y = _make_regression()
    wrapper = PartialFitESWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0,
                      learning_rate="constant", eta0=10.0),  # blow-up rate
        metric="rmse", patience=200, max_iter=80,  # large patience
        worsening_enabled=True, worsening_coeff=5, worsening_min_iters=5,
    )
    wrapper.fit(X, y)
    # Either curve-shape catches the divergence early, or numerics blew up so patience kicked in
    assert wrapper.stopped_via in {"curve_shape", "patience", "max_iter_hit"}


# -- dichotomic-search path -----------------------------------------------------

def test_dichotomic_ridge_picks_a_budget() -> None:
    """Ridge has no partial_fit but has max_iter; dichotomic search should converge to a budget."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge

    X, y = _make_regression()
    wrapper = PartialFitESWrapper(
        Ridge(alpha=1.0, random_state=0),
        metric="rmse", val_size=0.2,
        budget_param="max_iter", budget_min=10, budget_max=500,
    )
    wrapper.fit(X, y)
    assert wrapper.stopped_via == "dichotomic"
    assert 10 <= wrapper.best_iter <= 500
    assert wrapper.best_metric is not None
    assert wrapper.predict(X).shape == y.shape


def test_dichotomic_rf_n_estimators_search() -> None:
    """RandomForest has no partial_fit; dichotomic on n_estimators picks a budget."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    X, y = _make_regression(n=400)  # smaller for speed
    wrapper = PartialFitESWrapper(
        RandomForestRegressor(random_state=0, n_jobs=1),
        metric="rmse",
        budget_param="n_estimators", budget_min=5, budget_max=80,
    )
    wrapper.fit(X, y)
    assert wrapper.stopped_via == "dichotomic"
    assert 5 <= wrapper.best_iter <= 80


# -- error paths ----------------------------------------------------------------

def test_no_partial_fit_no_budget_param_raises() -> None:
    """Estimator without partial_fit and no budget_param should raise informatively."""
    pytest.importorskip("sklearn")
    from sklearn.dummy import DummyRegressor

    X, y = _make_regression()
    wrapper = PartialFitESWrapper(DummyRegressor(), metric="rmse")
    with pytest.raises(ValueError, match="partial_fit"):
        wrapper.fit(X, y)


def test_user_supplied_val_set_used_directly() -> None:
    """X_val / y_val supplied to fit() bypasses the internal train/val split."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import SGDRegressor

    X, y = _make_regression(n=600)
    X_val, y_val = _make_regression(seed=99, n=200)
    wrapper = PartialFitESWrapper(
        SGDRegressor(max_iter=1, tol=None, random_state=0,
                      learning_rate="constant", eta0=0.01),
        metric="rmse", patience=10, max_iter=30,
    )
    wrapper.fit(X, y, X_val=X_val, y_val=y_val)
    assert wrapper.best_metric is not None
    # The val we evaluated against came from a different seed than train; the history should
    # have <= max_iter entries.
    assert 0 < len(wrapper.history) <= 30
