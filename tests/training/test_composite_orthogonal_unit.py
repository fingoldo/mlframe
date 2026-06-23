"""Unit tests for OrthogonalizedCompositeEstimator (double-ML / FWL composite).

Cover: cross-fitting leakage-free, FWL identity == plain OLS on no-confounder data,
predict shape, and sklearn clone."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge

from mlframe.training.composite.orthogonal import (
    OrthogonalizedCompositeEstimator,
    _cross_fitted_oof,
)
from sklearn.model_selection import KFold


def _make_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    base = rng.normal(size=n)  # independent of features -> no confounding
    y = 2.0 * base + 1.5 * f0 - 0.7 * f1 + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"base": base, "f0": f0, "f1": f1})
    return X, y


def test_predict_shape():
    X, y = _make_frame()
    est = OrthogonalizedCompositeEstimator(
        base_column="base", inner_estimator=LinearRegression(), n_folds=4, random_state=0
    )
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()


def test_clone_preserves_params():
    est = OrthogonalizedCompositeEstimator(
        base_column="base",
        inner_estimator=Ridge(alpha=2.0),
        n_folds=3,
        random_state=7,
    )
    cl = clone(est)
    assert cl.base_column == "base"
    assert cl.n_folds == 3
    assert cl.random_state == 7
    assert isinstance(cl.inner_estimator, Ridge)
    # clone must not carry fitted state
    assert not hasattr(cl, "base_coef_")


def test_cross_fitting_is_leakage_free():
    # Each OOF prediction must come from a model that NEVER saw that row in training.
    # We verify by intercepting: a model memorizing its train rows would yield ~0 OOF
    # error on a pure-noise target only if it leaked. With a constant-mean predictor the
    # OOF prediction for a held-out fold must equal the mean of the OTHER folds, never
    # including the held-out row's own value.
    rng = np.random.default_rng(3)
    n = 60
    X = pd.DataFrame({"f0": rng.normal(size=n)})
    target = rng.normal(size=n)
    kf = KFold(n_splits=3, shuffle=False)

    from sklearn.dummy import DummyRegressor

    oof = _cross_fitted_oof(DummyRegressor(strategy="mean"), X, target, kf)
    # For fold k, prediction == mean(target on the other two folds), independent of fold k.
    fold = n // 3
    for k, (s, e) in enumerate([(0, fold), (fold, 2 * fold), (2 * fold, n)]):
        others = np.concatenate([target[:s], target[e:]])
        expected = others.mean()
        assert np.allclose(oof[s:e], expected), f"fold {k} leaked its own rows"


def test_fwl_identity_equals_plain_ols_on_no_confounder():
    # With base INDEPENDENT of X, partialling-out must leave the base coefficient
    # essentially equal to the plain joint-OLS base coefficient (FWL identity).
    X, y = _make_frame(n=2000, seed=1)
    est = OrthogonalizedCompositeEstimator(
        base_column="base",
        inner_estimator=LinearRegression(),
        base_nuisance_estimator=LinearRegression(),
        y_nuisance_estimator=LinearRegression(),
        n_folds=5,
        random_state=1,
    )
    est.fit(X, y)

    # Plain joint OLS of y on [base, f0, f1]; its base coefficient is the FWL reference.
    joint = LinearRegression().fit(X[["base", "f0", "f1"]].to_numpy(), y)
    ols_base = float(joint.coef_[0])

    assert abs(est.base_coef_ - ols_base) < 0.05, (
        f"FWL base_coef_={est.base_coef_:.4f} should match OLS {ols_base:.4f} on no-confounder data"
    )
    # And both near the true causal coefficient 2.0.
    assert abs(est.base_coef_ - 2.0) < 0.1


def test_n_folds_validation():
    X, y = _make_frame(n=100)
    est = OrthogonalizedCompositeEstimator(base_column="base", inner_estimator=Ridge(), n_folds=1)
    with pytest.raises(ValueError):
        est.fit(X, y)


def test_missing_base_column_raises():
    X, y = _make_frame()
    est = OrthogonalizedCompositeEstimator(base_column="nope", inner_estimator=Ridge(), n_folds=3)
    with pytest.raises(KeyError):
        est.fit(X, y)


def test_predict_before_fit_raises():
    X, _ = _make_frame()
    est = OrthogonalizedCompositeEstimator(base_column="base", inner_estimator=Ridge())
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        est.predict(X)
