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


def _two_regime_fn(X):
    trend = X["trend"].to_numpy() if hasattr(X, "columns") else np.asarray(X)[:, 0]
    return np.where(trend >= 0.0, "pos", "neg")


def _two_regime_proba_fn(X, tau: float = 0.3):
    trend = X["trend"].to_numpy() if hasattr(X, "columns") else np.asarray(X)[:, 0]
    p_pos = 1.0 / (1.0 + np.exp(-trend / tau))
    return {"pos": p_pos, "neg": 1.0 - p_pos}


def _make_smooth_transition_dataset(n: int, seed: int, tau: float = 0.3):
    """True relationship's INTERCEPT interpolates SMOOTHLY (sigmoid, width `tau`) between the "neg" and
    "pos" regime intercepts across the trend=0 boundary -- unlike ``_make_regime_dataset``'s hard step.
    A per-regime linear model fit on its own hard-split half necessarily anchors its intercept to ITS side's
    data, so the two regime models disagree right at the boundary (a real prediction gap, not just a small
    trend-scaled term vanishing at 0): hard routing snaps a boundary row to one full endpoint, while a
    confidence-weighted blend interpolates between them and tracks the true smooth curve far more closely.
    """
    rng = np.random.default_rng(seed)
    trend = rng.uniform(-1.5, 1.5, size=n)
    x2 = rng.normal(size=n)
    mix = 1.0 / (1.0 + np.exp(-trend / tau))
    intercept_pos, intercept_neg = 3.0, -3.0
    intercept = mix * intercept_pos + (1 - mix) * intercept_neg
    y = intercept + trend * 1.0 + x2 * 1.0 + rng.normal(scale=0.2, size=n)
    X = pd.DataFrame({"trend": trend, "x2": x2})
    return X, y


def test_biz_val_regime_split_ensemble_blend_smooths_regime_boundary_mse():
    X, y = _make_smooth_transition_dataset(n=6000, seed=7)
    rng = np.random.default_rng(8)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:4000], perm[4000:]
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    route_ensemble = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_two_regime_fn, combine="route")
    route_ensemble.fit(X_train, y_train)

    blend_ensemble = RegimeSplitEnsemble(
        estimator_factory=lambda: LinearRegression(), regime_fn=_two_regime_fn, combine="blend", regime_proba_fn=_two_regime_proba_fn
    )
    blend_ensemble.fit(X_train, y_train)

    # Boundary band: rows close to the trend=0 crossing, where hard routing's discontinuity bites hardest.
    band_mask = X_test["trend"].abs().to_numpy() < 0.3
    assert band_mask.sum() > 50

    pred_route = route_ensemble.predict(X_test)
    pred_blend = blend_ensemble.predict(X_test)

    mse_route_band = mean_squared_error(y_test[band_mask], pred_route[band_mask])
    mse_blend_band = mean_squared_error(y_test[band_mask], pred_blend[band_mask])

    improvement = 1.0 - mse_blend_band / mse_route_band
    assert improvement > 0.4, (
        f"expected blend to cut boundary-band MSE by >20% vs. hard routing, got {improvement:.4f} (route={mse_route_band:.4f}, blend={mse_blend_band:.4f})"
    )


def test_regime_split_ensemble_blend_requires_regime_proba_fn():
    X, y = _make_regime_dataset(n=100, seed=9)
    ensemble = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_regime_fn, combine="blend")
    try:
        ensemble.fit(X, y)
        raised = False
    except ValueError:
        raised = True
    assert raised, "combine='blend' without regime_proba_fn must raise ValueError at fit time"


def test_regime_split_ensemble_route_default_unchanged_by_blend_addition():
    """Bit-identical guard: the default (route) path must be untouched by adding "blend" support."""
    X, y = _make_regime_dataset(n=500, seed=10)
    ensemble = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_regime_fn)
    ensemble.fit(X, y)
    pred_default = ensemble.predict(X)

    ensemble_explicit = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_regime_fn, combine="route")
    ensemble_explicit.fit(X, y)
    pred_explicit = ensemble_explicit.predict(X)

    np.testing.assert_array_equal(pred_default, pred_explicit)
