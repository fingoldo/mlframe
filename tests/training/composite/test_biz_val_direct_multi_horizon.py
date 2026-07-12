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

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLasso

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


class _AutoTaskLasso(BaseEstimator, RegressorMixin):
    """``Lasso`` for a single-column block, ``MultiTaskLasso`` for a multi-column block. ``MultiTaskLasso``
    forces the SAME sparse feature-support across every column of the block (an L2,1 joint penalty) --
    a block whose columns genuinely share the same relevant features benefits from that joint shrinkage
    (less variance than fitting each column's ``Lasso`` separately); a block that blends columns driven by
    DIFFERENT feature subsets forces the joint support to compromise, incorrectly zeroing out features that
    only matter for some of the block's columns (underfit for every regime the block spans)."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> "_AutoTaskLasso":
        y_arr = np.asarray(y)
        self.model_ = (Lasso if y_arr.ndim == 1 else MultiTaskLasso)(alpha=self.alpha).fit(X, y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.model_.predict(X)


def _make_regime_horizon_dataset(n: int, n_regimes: int, steps_per_regime: int, p_per_regime: int, n_irrelevant: int, seed: int, noise: float):
    """Horizon made of ``n_regimes`` genuinely distinct linear regimes, each lasting ``steps_per_regime``
    steps and driven by its OWN disjoint subset of ``p_per_regime`` features (plus shared irrelevant
    features every regime ignores). Every column within one regime shares the exact same relevant features
    -- the true block boundary a joint (multi-task) model should discover."""
    rng = np.random.default_rng(seed)
    horizon = n_regimes * steps_per_regime
    p = n_regimes * p_per_regime + n_irrelevant
    X = rng.normal(size=(n, p))
    X0 = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Y = np.zeros((n, horizon))
    col = 0
    for r in range(n_regimes):
        idx = list(range(r * p_per_regime, r * p_per_regime + p_per_regime))
        beta = rng.normal(size=p_per_regime) * 2
        for _s in range(steps_per_regime):
            Y[:, col] = X[:, idx] @ beta + rng.normal(scale=noise, size=n)
            col += 1
    return X0, Y


def test_biz_val_direct_multi_horizon_auto_block_search_recovers_near_optimal_block_size():
    n, n_regimes, steps_per_regime = 150, 3, 3
    X0, Y = _make_regime_horizon_dataset(n, n_regimes, steps_per_regime, p_per_regime=2, n_irrelevant=4, seed=7, noise=0.6)
    n_train = 100
    X_train, X_test = X0.iloc[:n_train], X0.iloc[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    horizon = n_regimes * steps_per_regime

    def make_lasso():
        return _AutoTaskLasso(alpha=0.3)

    auto = DirectMultiHorizonEnsemble(
        estimator_factory=make_lasso,
        auto_block_search=True,
        block_size_grid=[1, 3, 9],
        cv_splits=3,
        random_state=0,
    )
    auto.fit(X_train, Y_train)
    rmse_auto = float(np.sqrt(np.mean((auto.predict(X_test) - Y_test) ** 2)))

    # the true regime boundary (block_size=3): near-optimal by construction.
    assert auto.block_search_report_ is not None
    chosen_block_size = len(auto.horizon_blocks_[0])
    assert chosen_block_size == 3, f"expected auto search to land on the true regime size 3, got {chosen_block_size}"

    # naively-bad choices: too fine (1 -- per-column Lasso forgoes the joint-shrinkage variance reduction)
    # and too coarse (9 -- one MultiTaskLasso forced to share support across all three distinct regimes).
    bad_fine = DirectMultiHorizonEnsemble(estimator_factory=make_lasso, horizon_blocks=[[h] for h in range(horizon)])
    bad_fine.fit(X_train, Y_train)
    rmse_bad_fine = float(np.sqrt(np.mean((bad_fine.predict(X_test) - Y_test) ** 2)))

    bad_coarse = DirectMultiHorizonEnsemble(estimator_factory=make_lasso, horizon_blocks=[list(range(horizon))])
    bad_coarse.fit(X_train, Y_train)
    rmse_bad_coarse = float(np.sqrt(np.mean((bad_coarse.predict(X_test) - Y_test) ** 2)))

    improvement_vs_coarse = 1.0 - rmse_auto / rmse_bad_coarse
    assert improvement_vs_coarse > 0.02, (
        f"expected auto-searched block size to beat the too-coarse (single-block, blended-regime) baseline, "
        f"got {improvement_vs_coarse:.4f} (auto={rmse_auto:.4f}, bad_coarse={rmse_bad_coarse:.4f})"
    )
    improvement_vs_fine = 1.0 - rmse_auto / rmse_bad_fine
    assert improvement_vs_fine > 0.10, (
        f"expected auto-searched block size to beat the too-fine (per-column Lasso, no joint shrinkage) baseline by >10% pooled RMSE, "
        f"got {improvement_vs_fine:.4f} (auto={rmse_auto:.4f}, bad_fine={rmse_bad_fine:.4f})"
    )


def test_direct_multi_horizon_default_behavior_unchanged_when_auto_search_omitted():
    n, horizon = 300, 8
    X0, Y, _, _ = _make_ar_horizon_dataset(n, horizon, ar_coef=0.8, seed=3)
    blocks = [[0, 1, 2, 3], [4, 5, 6, 7]]

    est = DirectMultiHorizonEnsemble(estimator_factory=lambda: LinearRegression(), horizon_blocks=blocks)
    est.fit(X0, Y)
    pred = est.predict(X0)

    assert est.horizon_blocks_ == blocks
    assert not hasattr(est, "block_search_report_")
    assert not hasattr(est, "block_feature_importances_")

    est2 = DirectMultiHorizonEnsemble(estimator_factory=lambda: LinearRegression(), horizon_blocks=blocks)
    est2.fit(X0, Y)
    pred2 = est2.predict(X0)
    np.testing.assert_array_equal(pred, pred2)


def test_direct_multi_horizon_block_diagnostics_reports_per_block_importance_similarity():
    from sklearn.ensemble import RandomForestRegressor

    n, horizon = 400, 6
    X = pd.DataFrame({"x1": np.random.default_rng(4).normal(size=n), "x2": np.random.default_rng(5).normal(size=n)})
    Y = np.random.default_rng(6).normal(size=(n, horizon))
    blocks = [[0, 1, 2], [3, 4, 5]]

    est = DirectMultiHorizonEnsemble(
        estimator_factory=lambda: RandomForestRegressor(n_estimators=10, max_depth=3, random_state=0),
        horizon_blocks=blocks,
        compute_block_diagnostics=True,
    )
    est.fit(X, Y)

    assert len(est.block_feature_importances_) == 2
    for vec in est.block_feature_importances_:
        assert vec is not None and vec.shape == (2,)
    assert len(est.block_importance_similarity_) == 1
    sim = est.block_importance_similarity_[0]
    assert sim is None or -1.0 <= sim <= 1.0


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
