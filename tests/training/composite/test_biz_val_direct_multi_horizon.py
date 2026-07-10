"""biz_value test for ``training.composite.DirectMultiHorizonEnsemble``.

The win: a recursive one-step-ahead forecaster, applied H times to generate a multi-horizon forecast, feeds
each step's own PREDICTED value back in as the next step's lag feature. If the underlying lag-to-lag model
has any attenuation bias (a realistic effect of engineered lag features carrying measurement noise -- classic
regression-dilution bias), that bias compounds multiplicatively across H recursive applications. A
``DirectMultiHorizonEnsemble`` instead fits one model per horizon step directly from the SAME origin-time
features every time -- no feedback loop, so the per-horizon error is a one-shot estimation error, not a
compounded one. Pooled RMSE across all horizons should be materially lower for the direct strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import DirectMultiHorizonEnsemble


def _make_ar_horizon_dataset(n: int, horizon: int, ar_coef: float, seed: int):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=n)
    X0 = pd.DataFrame({"x": z0 + rng.normal(scale=0.3, size=n)})

    Z = np.zeros((n, horizon + 1))
    Z[:, 0] = z0
    for h in range(1, horizon + 1):
        Z[:, h] = ar_coef * Z[:, h - 1] + rng.normal(scale=0.25, size=n)
    Y = Z[:, 1:] + rng.normal(scale=0.1, size=(n, horizon))
    return X0, Y, Z, rng


def test_biz_val_direct_multi_horizon_beats_recursive_forecaster_pooled_rmse():
    n, horizon, ar_coef = 3000, 12, 0.9
    X0, Y, Z, rng = _make_ar_horizon_dataset(n, horizon, ar_coef, seed=0)
    n_train = 2000
    X_train, X_test = X0.iloc[:n_train], X0.iloc[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    Z_train = Z[:n_train]

    # Recursive baseline: one lag-to-lag model, trained on NOISY lag features (regression-dilution/
    # attenuation bias -- realistic for engineered lag features), applied recursively at predict time.
    lag_in_noisy = (Z_train[:, :-1] + rng.normal(scale=0.6, size=Z_train[:, :-1].shape)).reshape(-1, 1)
    lag_out = Z_train[:, 1:].reshape(-1)
    recursive_model = LinearRegression().fit(lag_in_noisy, lag_out)

    cur = X_test["x"].to_numpy().copy()
    pred_recursive = np.zeros((len(cur), horizon))
    for h in range(horizon):
        cur = recursive_model.predict(cur.reshape(-1, 1))
        pred_recursive[:, h] = cur

    direct = DirectMultiHorizonEnsemble(estimator_factory=lambda: LinearRegression(), horizon_blocks=[[h] for h in range(horizon)])
    direct.fit(X_train, Y_train)
    pred_direct = direct.predict(X_test)

    rmse_recursive = float(np.sqrt(np.mean((pred_recursive - Y_test) ** 2)))
    rmse_direct = float(np.sqrt(np.mean((pred_direct - Y_test) ** 2)))
    improvement = 1.0 - rmse_direct / rmse_recursive

    assert improvement > 0.15, f"expected >15% pooled RMSE reduction vs. the recursive forecaster, got {improvement:.4f} (recursive={rmse_recursive:.4f}, direct={rmse_direct:.4f})"


def test_direct_multi_horizon_grouped_blocks_multi_output():
    from sklearn.multioutput import MultiOutputRegressor

    n, horizon = 500, 8
    X0, Y, _, _ = _make_ar_horizon_dataset(n, horizon, ar_coef=0.85, seed=1)
    blocks = [[0, 1, 2, 3], [4, 5, 6, 7]]  # two 4-step blocks, matching M5's weekly grouping
    est = DirectMultiHorizonEnsemble(estimator_factory=lambda: MultiOutputRegressor(LinearRegression()), horizon_blocks=blocks)
    est.fit(X0, Y)
    pred = est.predict(X0)
    assert pred.shape == (n, horizon)
    assert len(est.block_models_) == 2


def test_direct_multi_horizon_rejects_overlapping_or_incomplete_blocks():
    est_overlap = DirectMultiHorizonEnsemble(estimator_factory=lambda: LinearRegression(), horizon_blocks=[[0, 1], [1, 2]])
    try:
        est_overlap.fit(pd.DataFrame({"x": [1.0, 2.0]}), np.zeros((2, 3)))
        assert False, "expected ValueError for overlapping blocks"
    except ValueError:
        pass

    est_incomplete = DirectMultiHorizonEnsemble(estimator_factory=lambda: LinearRegression(), horizon_blocks=[[0]])
    try:
        est_incomplete.fit(pd.DataFrame({"x": [1.0, 2.0]}), np.zeros((2, 3)))
        assert False, "expected ValueError for incomplete horizon coverage"
    except ValueError:
        pass
