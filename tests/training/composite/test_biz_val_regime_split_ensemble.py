"""biz_value test for ``training.composite.RegimeSplitEnsemble``.

The win: when different regimes (bull/bear/stable market conditions) have genuinely DIFFERENT true
feature-target relationships, a single global model is forced to compromise across all of them, fitting
none well. Training one specialist per regime and routing each row to its own regime's model recovers each
regime's own relationship -- mirroring the G-Research Crypto Forecasting 9th place's 3-regime LightGBM split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.training.composite import RegimeSplitEnsemble


def _regime_fn(X):
    trend = X["trend"].to_numpy() if hasattr(X, "columns") else np.asarray(X)[:, 0]
    return np.where(trend > 0.3, "bull", np.where(trend < -0.3, "bear", "stable"))


def _make_regime_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    trend = rng.normal(size=n)
    x2 = rng.normal(size=n)
    regime = _regime_fn(pd.DataFrame({"trend": trend}))

    weights = {"bull": np.array([3.0, 1.0]), "bear": np.array([-3.0, 1.0]), "stable": np.array([0.2, -2.0])}
    y = np.zeros(n)
    for r, w in weights.items():
        mask = regime == r
        y[mask] = trend[mask] * w[0] + x2[mask] * w[1] + rng.normal(scale=0.3, size=mask.sum())

    X = pd.DataFrame({"trend": trend, "x2": x2})
    return X, y


def test_biz_val_regime_split_ensemble_route_beats_global_model_mse():
    X, y = _make_regime_dataset(n=3000, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:2000], perm[2000:]
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    global_model = LinearRegression().fit(X_train, y_train)
    mse_global = mean_squared_error(y_test, global_model.predict(X_test))

    ensemble = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_regime_fn, combine="route")
    ensemble.fit(X_train, y_train)
    mse_route = mean_squared_error(y_test, ensemble.predict(X_test))

    improvement = 1.0 - mse_route / mse_global
    assert improvement > 0.9, f"expected >90% MSE reduction vs. a single global model, got {improvement:.4f} (global={mse_global:.4f}, route={mse_route:.4f})"


def test_regime_split_ensemble_unseen_regime_falls_back_to_global_model():
    X, y = _make_regime_dataset(n=500, seed=2)
    # Train only on bull+stable rows -- "bear" regime never seen at fit time.
    train_mask = _regime_fn(X) != "bear"
    ensemble = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_regime_fn, combine="route")
    ensemble.fit(X[train_mask].reset_index(drop=True), y[train_mask])
    assert "bear" not in ensemble.regime_models_

    bear_rows = X[~train_mask].reset_index(drop=True)
    pred = ensemble.predict(bear_rows)
    np.testing.assert_allclose(pred, ensemble.global_model_.predict(bear_rows))


def test_regime_split_ensemble_average_mode_shape():
    X, y = _make_regime_dataset(n=300, seed=3)
    ensemble = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_regime_fn, combine="average")
    ensemble.fit(X, y)
    pred = ensemble.predict(X)
    assert pred.shape == (300,)
