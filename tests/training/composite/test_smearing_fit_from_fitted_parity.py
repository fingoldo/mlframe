"""Audit 2026-11-20 finding 35-F1: a wrapper built with from_fitted_inner must predict what fit() predicts.

Before the fix from_fitted_inner never stamped ``smearing_quantiles``, so a log_y composite built on the suite path returned
the plain geometric-mean inverse while fit() returned the smeared conditional mean.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator._estimator import CompositeTargetEstimator


def _right_skewed_fixture(n=600, seed=0):
    """Right-skewed log-normal target with one feature that drives its location."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = np.exp(0.5 * X["x"].to_numpy() + rng.normal(scale=0.8, size=n))
    return X, y


def test_from_fitted_inner_stamps_the_same_smearing_quantiles_as_fit():
    """The wrapper stamps the same residual quantiles fit() stamps on the same data."""
    X, y = _right_skewed_fixture()
    cte = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="log_y", base_column="x")
    cte.fit(X, y)
    assert cte.fitted_params_.get("smearing_quantiles") is not None

    wrapped = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=cte.estimator_,
        transform_name="log_y",
        base_column="x",
        transform_fitted_params=dict(cte.fitted_params_),
        y_train=y,
        base_train=X["x"].to_numpy(),
        X_train=X,
    )
    np.testing.assert_allclose(
        wrapped.fitted_params_["smearing_quantiles"], cte.fitted_params_["smearing_quantiles"], rtol=1e-6,
    )


def test_from_fitted_inner_predicts_the_smeared_mean_like_fit():
    """Predictions from the from_fitted wrapper equal fit()'s predictions (the smeared mean)."""
    X, y = _right_skewed_fixture()
    cte = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="log_y", base_column="x")
    cte.fit(X, y)
    wrapped = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=cte.estimator_,
        transform_name="log_y",
        base_column="x",
        transform_fitted_params=dict(cte.fitted_params_),
        y_train=y,
        base_train=X["x"].to_numpy(),
        X_train=X,
    )
    np.testing.assert_allclose(wrapped.predict(X), cte.predict(X), rtol=1e-6)


def test_from_fitted_inner_without_train_features_keeps_plain_inverse():
    """Callers that cannot supply the train features get no smearing stamp, as before (no silent new behaviour)."""
    X, y = _right_skewed_fixture()
    cte = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="log_y", base_column="x")
    cte.fit(X, y)
    wrapped = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=cte.estimator_,
        transform_name="log_y",
        base_column="x",
        transform_fitted_params={k: v for k, v in cte.fitted_params_.items() if k != "smearing_quantiles"},
        y_train=y,
        base_train=X["x"].to_numpy(),
    )
    assert wrapped.fitted_params_.get("smearing_quantiles") is None
