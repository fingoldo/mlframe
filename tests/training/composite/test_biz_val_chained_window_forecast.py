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
    assert improvement > 0.4, (
        f"expected >40% MSE reduction vs. the naive linear baseline, got {improvement:.4f} (baseline={baseline_mse:.4f}, chained={chained_mse:.4f})"
    )


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
    _X_prev_test, X_curr_test, _, y_target_test = _make_ar_window_dataset(n=1000, seed=2)

    labeled_only = ChainedWindowForecaster(stage1_estimator=GradientBoostingRegressor(random_state=0, n_estimators=100), stage2_estimator=LinearRegression())
    labeled_only.fit(X_prev_train, X_curr_train, y_curr_train, y_target_train)
    mse_labeled_only = mean_squared_error(y_target_test, labeled_only.predict(X_curr_test))

    with_transductive = ChainedWindowForecaster(
        stage1_estimator=GradientBoostingRegressor(random_state=0, n_estimators=100), stage2_estimator=LinearRegression()
    )
    with_transductive.fit(X_prev_train, X_curr_train, y_curr_train, y_target_train, X_prev_extra=X_prev_extra, y_curr_extra=y_curr_extra)
    mse_with_transductive = mean_squared_error(y_target_test, with_transductive.predict(X_curr_test))

    improvement = 1.0 - mse_with_transductive / mse_labeled_only
    assert improvement > 0.3, (
        f"expected >30% MSE reduction from transductive stage-1 pretraining, got {improvement:.4f} (labeled_only={mse_labeled_only:.4f}, with_transductive={mse_with_transductive:.4f})"
    )


def _make_drifting_window(n: int, seed: int, position: int, drift_per_position: float, ar_coef: float = 0.9):
    """Window ``position`` steps past the one the forecaster was fit on: the nonlinear feature-encoding
    FREQUENCY shifts with ``position``, so a stage-1 model fit on position 0's encoding increasingly
    misreads ``f1``/``f2`` at later positions -- a genuine, growing extrapolation mismatch, not just fresh
    sampling noise (a stable-encoding control keeps the frequency fixed for comparison)."""
    rng = np.random.default_rng(seed * 1000 + position)
    z_curr = rng.normal(size=n)
    z_target = ar_coef * z_curr + rng.normal(scale=0.3, size=n)
    freq = 2.0 + position * drift_per_position
    X_curr = pd.DataFrame(
        {
            "f1": np.sin(z_curr * freq) + rng.normal(scale=0.2, size=n),
            "f2": z_curr**2 + rng.normal(scale=0.2, size=n),
            "f3": rng.normal(size=n),
        }
    )
    return X_curr, z_target


def test_biz_val_chained_window_forecaster_diagnose_error_accumulation_flags_drift():
    """A forecaster fit once at position 0 is reused, without refitting, across later windows whose
    nonlinear encoding drifts away from what stage 1 learned. The diagnostic should surface a materially
    worse (bigger growth_ratio, shorter trustworthy_horizon) picture for the DRIFTING chain than for a
    STABLE control chain with the identical process but a fixed encoding."""
    X_prev, X_curr0, y_curr, y_target0 = _make_ar_window_dataset(n=3000, seed=0)
    chained = ChainedWindowForecaster(stage1_estimator=GradientBoostingRegressor(random_state=0, n_estimators=100), stage2_estimator=LinearRegression())
    chained.fit(X_prev, X_curr0, y_curr, y_target0)

    n_positions = 6
    drifting_X, drifting_y = [], []
    stable_X, stable_y = [], []
    for pos in range(n_positions):
        Xd, yd = _make_drifting_window(n=1500, seed=1, position=pos, drift_per_position=0.6)
        drifting_X.append(Xd)
        drifting_y.append(yd)
        Xs, ys = _make_drifting_window(n=1500, seed=1, position=pos, drift_per_position=0.0)
        stable_X.append(Xs)
        stable_y.append(ys)

    drifting_diag = chained.diagnose_error_accumulation(drifting_X, drifting_y, accumulation_threshold=2.0)
    stable_diag = chained.diagnose_error_accumulation(stable_X, stable_y, accumulation_threshold=2.0)

    assert drifting_diag["growth_ratio"][-1] > 2.0, f"expected drifting chain's final growth_ratio > 2.0, got {drifting_diag['growth_ratio'][-1]:.4f}"
    assert drifting_diag["trustworthy_horizon"] < n_positions, (
        f"expected drift to be flagged before the end of the chain, got trustworthy_horizon={drifting_diag['trustworthy_horizon']}"
    )
    assert stable_diag["trustworthy_horizon"] == n_positions, (
        f"expected the stable control chain to stay trustworthy throughout, got trustworthy_horizon={stable_diag['trustworthy_horizon']}"
    )
    assert drifting_diag["trustworthy_horizon"] < stable_diag["trustworthy_horizon"]


def test_chained_window_forecaster_default_predict_unchanged_by_diagnostic_addition():
    """Regression test: adding the opt-in `diagnose_error_accumulation` method must not alter fit/predict's
    prior behavior at all -- predict() output must be bit-identical to calling it directly."""
    X_prev, X_curr, y_curr, y_target = _make_ar_window_dataset(n=200, seed=3)
    chained = ChainedWindowForecaster(stage1_estimator=LinearRegression(), stage2_estimator=LinearRegression())
    chained.fit(X_prev, X_curr, y_curr, y_target)

    pred_direct = chained.predict(X_curr)
    diag = chained.diagnose_error_accumulation([X_curr], [y_target])
    pred_via_diagnostic_path = chained.predict(X_curr)

    np.testing.assert_array_equal(pred_direct, pred_via_diagnostic_path)
    np.testing.assert_array_equal(diag["chain_predictions"][0], pred_direct)
