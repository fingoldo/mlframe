"""biz_value test for ``inference.recursive_forecast.recursive_multi_step_forecast``.

The win (5th_m5-forecasting-accuracy.md): when a short lag feature (e.g. ``lag_7``) is genuinely a strong
predictor but true ground truth for it is NOT available beyond the first few forecast steps (a real
multi-step-ahead forecasting scenario, not a backtest with hindsight), the only ways to fill it are (a)
freeze it at the last known value (stale, loses signal as the horizon grows) or (b) recurse using the model's
own previous predictions. This test confirms recursion recovers materially more of the lag-driven signal than
freezing, specifically because the freeze approach's staleness gets WORSE with horizon while recursive
predictions track the evolving series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from mlframe.inference.recursive_forecast import recursive_multi_step_forecast, diagnose_error_accumulation


def _make_ar_series(n_series: int, n_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    # AR(1)-style series: each step depends strongly on the previous true value.
    series = np.zeros((n_series, n_steps + 1))
    series[:, 0] = rng.normal(scale=5.0, size=n_series)
    for t in range(1, n_steps + 1):
        series[:, t] = 0.85 * series[:, t - 1] + rng.normal(scale=1.0, size=n_series)
    return series


def test_biz_val_recursive_forecast_beats_frozen_lag_on_ar_series():
    n_series, n_steps = 800, 6
    series = _make_ar_series(n_series, n_steps, seed=0)

    # Train a simple AR(1) model: predict step t from lag_1 (previous step's value).
    X_train = series[:, :-1].reshape(-1, 1)
    y_train = series[:, 1:].reshape(-1)
    model = Ridge(alpha=0.1).fit(X_train, y_train)

    true_future = series[:, 1 : n_steps + 1]  # (n_series, n_steps), ground truth for each forecast step

    # Recursive: lag_1 is fed forward from the model's own previous prediction.
    initial_features = pd.DataFrame({"lag_1": series[:, 0]})

    def _update(features, pred, step):
        return features  # lag_1 is already updated by the loop itself; no other features to update

    recursive_preds = recursive_multi_step_forecast(model, initial_features, n_steps, "lag_1", _update)
    mse_recursive = float(mean_squared_error(true_future.T, recursive_preds))

    # Frozen: lag_1 stays at its step-1 (last truly known) value for every forecast step -- the only
    # alternative when true ground truth genuinely isn't available for later steps.
    frozen_lag = np.full((n_steps, n_series), series[:, 0])
    frozen_preds = model.predict(frozen_lag.reshape(-1, 1)).reshape(n_steps, n_series)
    mse_frozen = float(mean_squared_error(true_future.T, frozen_preds))

    assert mse_recursive < mse_frozen, f"expected recursive forecasting to beat a frozen-lag baseline on an AR series where true future ground truth is unavailable, got recursive={mse_recursive:.4f} frozen={mse_frozen:.4f}"

    # Honest acknowledgment of the known risk: recursive error should still GROW with horizon (compounding),
    # even though it beats the frozen baseline overall -- confirming this is a real, imperfect necessity, not
    # a free lunch.
    per_step_mse = np.mean((true_future.T - recursive_preds) ** 2, axis=1)
    assert per_step_mse[-1] > per_step_mse[0], "expected recursive forecast error to compound (grow) across steps, matching the documented known risk of this technique"


def test_recursive_multi_step_forecast_shape_and_lag_column_required():
    import pytest

    class _DummyModel:
        def predict(self, X):
            return X["lag_1"].to_numpy() * 0.5

    features = pd.DataFrame({"lag_1": [1.0, 2.0, 3.0]})
    preds = recursive_multi_step_forecast(_DummyModel(), features, n_steps=3, lag_feature_name="lag_1", update_features_fn=lambda f, p, s: f)
    assert preds.shape == (3, 3)
    np.testing.assert_allclose(preds[0], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(preds[1], [0.25, 0.5, 0.75])

    with pytest.raises(ValueError):
        recursive_multi_step_forecast(_DummyModel(), pd.DataFrame({"other": [1.0]}), n_steps=2, lag_feature_name="lag_1", update_features_fn=lambda f, p, s: f)


def test_biz_val_diagnose_error_accumulation_surfaces_compounding_and_trustworthy_horizon():
    """The win: on a series where recursive error genuinely compounds, the diagnostic's growth_ratio must
    rise well past 1.0 and its trustworthy_horizon must correctly flag the step where recursion stops being
    materially better than the oracle (perfect-hindsight, non-recursive) floor -- giving a caller a concrete
    cutoff for switching to a direct multi-output model instead of trusting recursion indefinitely.
    """
    n_series, n_steps = 600, 12
    # A near-unit-root AR(1) with noticeable innovation noise: single-step model error is small, but because
    # each step's "lag" input IS the previous (already imperfect) prediction, the model's own noise floor
    # gets re-injected as an ever-larger input error every step, so downstream error compounds materially.
    rng = np.random.default_rng(1)
    series = np.zeros((n_series, n_steps + 1))
    series[:, 0] = rng.normal(scale=5.0, size=n_series)
    coef = 0.98
    for t in range(1, n_steps + 1):
        series[:, t] = coef * series[:, t - 1] + rng.normal(scale=1.5, size=n_series)

    X_train = series[:, :-1].reshape(-1, 1)
    y_train = series[:, 1:].reshape(-1)
    model = Ridge(alpha=0.1).fit(X_train, y_train)

    true_future = series[:, 1 : n_steps + 1]  # (n_series, n_steps)
    true_targets = true_future.T  # (n_steps, n_series), matching diagnose_error_accumulation's contract
    oracle_lag_values = series[:, :n_steps].T  # true (ground-truth) lag_1 at each step -- perfect hindsight

    initial_features = pd.DataFrame({"lag_1": series[:, 0]})
    result = diagnose_error_accumulation(
        model,
        initial_features,
        n_steps,
        "lag_1",
        lambda f, p, s: f,
        true_targets=true_targets,
        oracle_lag_values=oracle_lag_values,
        accumulation_threshold=2.0,
    )

    assert result["recursive_mse"].shape == (n_steps,)
    assert result["oracle_mse"] is not None and result["oracle_mse"].shape == (n_steps,)

    # The oracle (perfect-hindsight, non-recursive) floor stays roughly flat step-to-step since the model's
    # single-step noise doesn't compound when it's always fed the TRUE lag.
    oracle_growth = result["oracle_mse"][-1] / result["oracle_mse"][0]
    assert oracle_growth < 1.5, f"expected the oracle baseline to stay near-flat across the horizon, got growth={oracle_growth:.2f}"

    # The recursive error, in contrast, must compound materially -- this is the real, measurable effect the
    # diagnostic exists to surface.
    assert result["growth_ratio"][-1] > 3.0, f"expected recursive error to compound materially over {n_steps} steps, got growth_ratio={result['growth_ratio'][-1]:.2f}"

    # The diagnostic must correctly detect the breakdown point: recursion should still be trustworthy for at
    # least the first couple of steps (close to the oracle), but should NOT remain trustworthy for the full
    # horizon once compounding pushes it far past the oracle floor.
    assert 1 <= result["trustworthy_horizon"] < n_steps, f"expected diagnostic to flag a trustworthy_horizon strictly short of the full {n_steps}-step horizon, got {result['trustworthy_horizon']}"


def test_diagnose_error_accumulation_shape_validation():
    import pytest

    class _DummyModel:
        def predict(self, X):
            return X["lag_1"].to_numpy() * 0.5

    features = pd.DataFrame({"lag_1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        diagnose_error_accumulation(
            _DummyModel(), features, n_steps=2, lag_feature_name="lag_1", update_features_fn=lambda f, p, s: f, true_targets=np.zeros((3, 3))
        )
