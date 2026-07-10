"""biz_value test for ``training.DirectHorizonBucketForecaster``.

The win: a genuinely non-autoregressive process (target driven by calendar features -- trend + day-of-week
seasonality -- not by its own recent history) is forecast far more accurately by independent direct models
per horizon bucket using calendar features than by a naive recursive forecaster that feeds its own noisy
prediction forward as a lag feature -- the recursive approach's error visibly GROWS with horizon distance
(error accumulation), while the direct approach's error stays flat near the noise floor across the whole
horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.training._direct_horizon_bucket_forecaster import DirectHorizonBucketForecaster


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _true_y(day_idx: np.ndarray) -> np.ndarray:
    weekday_effect = np.array([0, 1, 2, 1, 0, -1, -2])[day_idx % 7]
    return 10 + 0.05 * day_idx + weekday_effect


def test_biz_val_direct_horizon_bucket_forecaster_beats_recursive_forecasting():
    rng = np.random.default_rng(0)
    n_days_hist = 200
    horizon = 28

    hist_days = np.arange(n_days_hist)
    hist_y = _true_y(hist_days) + rng.normal(0, 0.5, n_days_hist)

    # recursive baseline: a model predicts y_t from y_{t-1} only, then iteratively forecasts forward by
    # feeding its own prior prediction back in as the next step's input -- error compounds with horizon.
    ar_model = LinearRegression().fit(hist_y[:-1].reshape(-1, 1), hist_y[1:])
    recursive_preds = []
    last_val = hist_y[-1]
    for _ in range(horizon):
        pred = ar_model.predict(np.array([[last_val]]))[0]
        recursive_preds.append(pred)
        last_val = pred
    recursive_preds = np.array(recursive_preds)

    future_days = n_days_hist + np.arange(horizon)
    true_future = _true_y(future_days) + rng.normal(0, 0.5, horizon)
    recursive_rmse = _rmse(true_future, recursive_preds)

    # direct: independent per-bucket models using only calendar features known at forecast time (no
    # self-referential lag) -- day_offset, day_of_week, and the forecast origin's own day index.
    buckets = [(1, 7), (8, 14), (15, 21), (22, 28)]
    rows, targets = [], []
    for origin in range(0, n_days_hist - horizon):
        for h in range(1, horizon + 1):
            day_idx = origin + h
            rows.append({"day_offset": h, "day_of_week": day_idx % 7, "origin_day_idx": origin, "grp": 0})
            targets.append(_true_y(np.array([day_idx]))[0] + rng.normal(0, 0.5))
    X_direct = pd.get_dummies(pd.DataFrame(rows), columns=["day_of_week"])
    y_direct = np.array(targets)
    horizon_day_direct = X_direct["day_offset"].to_numpy()

    forecaster = DirectHorizonBucketForecaster(buckets, model_factory=lambda: LinearRegression(), group_col="grp")
    forecaster.fit(X_direct, y_direct, horizon_day_direct)

    X_test = pd.DataFrame({"day_offset": np.arange(1, horizon + 1), "day_of_week": future_days % 7, "origin_day_idx": n_days_hist, "grp": 0})
    X_test = pd.get_dummies(X_test, columns=["day_of_week"]).reindex(columns=X_direct.columns, fill_value=0)
    direct_preds = forecaster.predict(X_test, np.arange(1, horizon + 1))
    direct_rmse = _rmse(true_future, direct_preds)

    assert direct_rmse < recursive_rmse * 0.3, (
        f"direct per-bucket forecasting should avoid recursive error accumulation: "
        f"direct={direct_rmse:.4f} recursive={recursive_rmse:.4f}"
    )

    # the accumulation signature itself: recursive error in the LAST bucket should be much worse than in
    # the FIRST bucket, while direct error stays roughly flat across the horizon.
    recursive_first_bucket_rmse = _rmse(true_future[:7], recursive_preds[:7])
    recursive_last_bucket_rmse = _rmse(true_future[-7:], recursive_preds[-7:])
    assert recursive_last_bucket_rmse > recursive_first_bucket_rmse * 1.5


def test_direct_horizon_bucket_forecaster_no_buckets_raises():
    import pytest

    with pytest.raises(ValueError):
        DirectHorizonBucketForecaster([], model_factory=lambda: LinearRegression())


def test_direct_horizon_bucket_forecaster_no_group_col_pools_all_entities():
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({"x": rng.normal(0, 1, n)})
    y = 2.0 * X["x"].to_numpy() + rng.normal(0, 0.1, n)
    horizon_day = rng.integers(1, 15, n)

    forecaster = DirectHorizonBucketForecaster([(1, 7), (8, 14)], model_factory=lambda: LinearRegression())
    forecaster.fit(X, y, horizon_day)
    preds = forecaster.predict(X, horizon_day)
    assert not np.isnan(preds).any()
    assert _rmse(y, preds) < 0.5
