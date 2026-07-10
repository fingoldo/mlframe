"""biz_value test for ``training.composite.ChainedWindowForecaster``.

The win: an AR(1) latent state ``z`` drives each window's features NONLINEARLY (``sin``/``square`` of
``z``). A naive linear model fit directly on the target window's own (nonlinear) features can't decode
``z`` and so can't forecast the next window's target well. A nonlinear stage-1 model, fit on the PRECEDING
window's features predicting a proxy quantity for the current window, learns to decode the nonlinear
encoding; applying that same fitted function one step further (to the current window's own features)
extrapolates a clean estimate of the AR-driven component. Feeding that single extrapolated value into an
otherwise-linear stage-2 model recovers the true next-window target far better than the naive baseline --
mirroring the Optiver 3rd place's "300 seconds model" chaining technique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.training.composite import ChainedWindowForecaster


def _make_ar_window_dataset(n: int, seed: int, ar_coef: float = 0.9):
    rng = np.random.default_rng(seed)
    z_prev = rng.normal(size=n)
    z_curr = ar_coef * z_prev + rng.normal(scale=0.3, size=n)
    z_target = ar_coef * z_curr + rng.normal(scale=0.3, size=n)

    def make_features(z: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": np.sin(z * 2) + rng.normal(scale=0.2, size=len(z)),
                "f2": z**2 + rng.normal(scale=0.2, size=len(z)),
                "f3": rng.normal(size=len(z)),
            }
        )

    X_prev = make_features(z_prev)
    X_curr = make_features(z_curr)
    y_curr = z_curr + rng.normal(scale=0.1, size=n)
    y_target = z_target
    return X_prev, X_curr, y_curr, y_target


def test_biz_val_chained_window_forecaster_beats_naive_linear_baseline_mse():
    X_prev, X_curr, y_curr, y_target = _make_ar_window_dataset(n=3000, seed=0)
    idx = np.arange(len(y_target))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=0)

    baseline = LinearRegression()
    baseline.fit(X_curr.iloc[train_idx], y_target[train_idx])
    baseline_mse = mean_squared_error(y_target[test_idx], baseline.predict(X_curr.iloc[test_idx]))

    chained = ChainedWindowForecaster(stage1_estimator=GradientBoostingRegressor(random_state=0, n_estimators=100), stage2_estimator=LinearRegression())
    chained.fit(X_prev.iloc[train_idx], X_curr.iloc[train_idx], y_curr[train_idx], y_target[train_idx])
    chained_mse = mean_squared_error(y_target[test_idx], chained.predict(X_curr.iloc[test_idx]))

    improvement = 1.0 - chained_mse / baseline_mse
    assert improvement > 0.4, f"expected >40% MSE reduction vs. the naive linear baseline, got {improvement:.4f} (baseline={baseline_mse:.4f}, chained={chained_mse:.4f})"


def test_chained_window_forecaster_injects_chained_feature_column():
    X_prev, X_curr, y_curr, y_target = _make_ar_window_dataset(n=200, seed=1)
    chained = ChainedWindowForecaster(stage1_estimator=LinearRegression(), stage2_estimator=LinearRegression(), chained_feature_name="my_chained_col")
    chained.fit(X_prev, X_curr, y_curr, y_target)
    X2 = chained._concat_chained(X_curr, np.zeros(len(X_curr)))
    assert "my_chained_col" in X2.columns
    assert set(X2.columns) == set(X_curr.columns) | {"my_chained_col"}


def test_chained_window_forecaster_ndarray_input():
    X_prev, X_curr, y_curr, y_target = _make_ar_window_dataset(n=200, seed=2)
    chained = ChainedWindowForecaster(stage1_estimator=LinearRegression(), stage2_estimator=LinearRegression())
    chained.fit(X_prev.to_numpy(), X_curr.to_numpy(), y_curr, y_target)
    pred = chained.predict(X_curr.to_numpy())
    assert pred.shape == (200,)


def test_biz_val_chained_window_forecaster_transductive_stage1_pretraining_beats_labeled_only():
    """The source technique's "trained ... using both train and test data" detail: stage 1's proxy target
    is fully observed on unlabeled/test-like rows too (it never needs y_target), so folding those rows into
    ONLY the stage-1 fit via X_prev_extra/y_curr_extra should improve stage 1's window-to-window mapping
    when the labeled training set alone is small -- without touching stage 2, which still only sees real
    labels."""
    X_prev_train, X_curr_train, y_curr_train, y_target_train = _make_ar_window_dataset(n=80, seed=0)
    X_prev_extra, _, y_curr_extra, _ = _make_ar_window_dataset(n=1500, seed=1)
    X_prev_test, X_curr_test, _, y_target_test = _make_ar_window_dataset(n=1000, seed=2)

    labeled_only = ChainedWindowForecaster(stage1_estimator=GradientBoostingRegressor(random_state=0, n_estimators=100), stage2_estimator=LinearRegression())
    labeled_only.fit(X_prev_train, X_curr_train, y_curr_train, y_target_train)
    mse_labeled_only = mean_squared_error(y_target_test, labeled_only.predict(X_curr_test))

    with_transductive = ChainedWindowForecaster(stage1_estimator=GradientBoostingRegressor(random_state=0, n_estimators=100), stage2_estimator=LinearRegression())
    with_transductive.fit(X_prev_train, X_curr_train, y_curr_train, y_target_train, X_prev_extra=X_prev_extra, y_curr_extra=y_curr_extra)
    mse_with_transductive = mean_squared_error(y_target_test, with_transductive.predict(X_curr_test))

    improvement = 1.0 - mse_with_transductive / mse_labeled_only
    assert improvement > 0.3, f"expected >30% MSE reduction from transductive stage-1 pretraining, got {improvement:.4f} (labeled_only={mse_labeled_only:.4f}, with_transductive={mse_with_transductive:.4f})"
