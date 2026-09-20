"""Smearing makes a log/cbrt composite predict the conditional MEAN of y, so it can compete on y-scale RMSE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetEstimator


def _data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.exp(1.0 + 0.5 * x + rng.normal(scale=1.0, size=n))  # lognormal noise: geometric mean << mean
    return pd.DataFrame({"x": x}), y


@pytest.mark.parametrize("transform", ["log_y", "cbrt_y"])
def test_smeared_prediction_is_mean_unbiased(transform):
    from sklearn.linear_model import LinearRegression

    X, y = _data()
    Xt, yt = _data(seed=1)
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name=transform).fit(X, y)
    pred = est.predict(Xt)
    assert abs(pred.mean() / yt.mean() - 1.0) < 0.05
    params = dict(est.fitted_params_)
    params["smearing_quantiles"] = None
    est.fitted_params_ = params
    plain = est.predict(Xt)
    if transform == "log_y":
        assert plain.mean() < 0.8 * yt.mean()  # the geometric-mean bias the smearing removes
    assert np.sqrt(np.mean((pred - yt) ** 2)) < np.sqrt(np.mean((plain - yt) ** 2))
