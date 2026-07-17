"""biz_value tests for multi-target regression (MTR).

Two quantitative contracts on correlated-target synthetics:

1. Shared-trunk MultiRMSE (one CatBoost ensemble emitting (N, K)) must not regress on macro test RMSE
   versus K independent single-target CatBoost regressors. The shared trunk's value on synthetic data is
   parity-at-a-fraction-of-the-cost (one ensemble vs K), not a reliable RMSE lift -- a seed-stable
   RMSE-lift floor could not be established on synthetic correlated targets (the gap is within +/-1% across
   seeds), so this asserts the no-regression contract.

2. The NNLS per-column ensemble (the ``strategy='nnls'`` default when val data is present) must beat the
   best single member by a wide margin when the component models are complementary per target column.
   Measured min lift ~6.6% across 8 seeds; floor set at 4%.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from mlframe.training.core._phase_composite_post_xt_ensemble import MTRPerColumnEqualMeanEnsemble


def _macro_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_shared_latent(n: int, d: int, k: int, seed: int):
    """Correlated K-target regression: targets are noisy linear maps of a shared 2-factor latent of X."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    L = rng.normal(size=(d, 2)).astype("float32")
    Z = X @ L
    M = rng.normal(size=(2, k)).astype("float32")
    y = (Z @ M + 0.3 * rng.normal(size=(n, k))).astype("float32")
    return X, y


def _make_complementary(n: int, d: int, seed: int):
    """3 targets each favouring a different model family: linear / nonlinear-trees / mixed."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y0 = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2]
    y1 = 3.0 * (X[:, 0] * X[:, 1] > 0).astype("float32") + 2.0 * np.abs(X[:, 3]) - 1.5 * (X[:, 4] > 0.5)
    y2 = 1.5 * X[:, 2] + 2.0 * np.sin(2 * X[:, 5])
    y = np.stack([y0, y1, y2], axis=1).astype("float32")
    y += 0.15 * rng.normal(size=y.shape).astype("float32")
    return X, y


@pytest.mark.parametrize("seed", [0, 100])
def test_biz_val_mtr_shared_trunk_matches_independent(seed):
    """Shared-trunk MultiRMSE macro RMSE must not exceed K-independent CatBoost by >2% (parity contract)."""
    catboost = pytest.importorskip("catboost")
    from mlframe.training.strategies import CatBoostStrategy

    X_tr, y_tr = _make_shared_latent(1500, 8, 3, seed)
    X_te, y_te = _make_shared_latent(800, 8, 3, seed + 1000)

    kw = CatBoostStrategy().get_multi_target_objective_kwargs()
    assert kw == {"loss_function": "MultiRMSE"}
    shared = catboost.CatBoostRegressor(iterations=120, learning_rate=0.1, depth=4, verbose=False, **kw)
    shared.fit(X_tr, y_tr)
    rmse_shared = _macro_rmse(y_te, shared.predict(X_te))

    indep_preds = []
    for j in range(3):
        m = catboost.CatBoostRegressor(iterations=120, learning_rate=0.1, depth=4, verbose=False)
        m.fit(X_tr, y_tr[:, j])
        indep_preds.append(m.predict(X_te))
    rmse_indep = _macro_rmse(y_te, np.stack(indep_preds, axis=1))

    assert rmse_shared <= rmse_indep * 1.02, (
        f"shared-trunk MultiRMSE regressed vs K independent CatBoost: shared={rmse_shared:.4f} indep={rmse_indep:.4f} ratio={rmse_shared / rmse_indep:.4f}"
    )


@pytest.mark.parametrize("seed", [0, 7])
def test_biz_val_mtr_nnls_ensemble_beats_best_single_member(seed):
    """NNLS per-column ensemble test RMSE <= best single-member RMSE * 0.96 on complementary targets.

    Floor 4%; measured min lift ~6.6% across 8 seeds (LGB wins the nonlinear column, Ridge the linear ones,
    so the per-column NNLS solve picks the right model per target and beats either single member)."""
    lgb = pytest.importorskip("lightgbm")
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    X_tr, y_tr = _make_complementary(1800, 8, seed)
    X_va, y_va = _make_complementary(1200, 8, seed + 50)
    X_te, y_te = _make_complementary(1200, 8, seed + 100)

    c_lgb = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=80, verbose=-1)).fit(X_tr, y_tr)
    c_ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
    components = [c_lgb, c_ridge]

    single_rmses = [_macro_rmse(y_te, c.predict(X_te)) for c in components]
    best_single = min(single_rmses)

    ens = MTRPerColumnEqualMeanEnsemble(
        components=components,
        component_names=["lgb", "ridge"],
        n_targets=3,
        strategy="nnls",
    )
    ens.fit(X_va, y_va)  # learn per-column weights on held-out val
    rmse_ens = _macro_rmse(y_te, ens.predict(X_te))

    lift_pct = (best_single - rmse_ens) / best_single * 100.0
    assert rmse_ens <= best_single * 0.96, (
        f"NNLS ensemble failed to beat best single member by >=4%: "
        f"singles={[f'{s:.4f}' for s in single_rmses]} best={best_single:.4f} "
        f"nnls={rmse_ens:.4f} lift={lift_pct:+.2f}%"
    )
    # NNLS weights respect the non-negativity constraint per column.
    assert (ens.weights >= 0).all(), "NNLS weights must be non-negative"
