"""biz_value test for ``CompositeQRFEstimator.forest_kind`` (the leaf-residual backend).

The win: on a SMOOTH-signal + heavy-noise target the ExtraTrees backend (``forest_kind="et"``)
lowers honest-holdout point RMSE versus the RandomForest backend (``forest_kind="rf"``). ExtraTrees'
extra split-threshold randomization decorrelates members harder, cutting the variance term where the
true response is smooth and the per-tree bias is already low.

``forest_kind`` is consumed only by the pure-sklearn ``_LeafResidualForest``, so the test pins
``prefer_quantile_forest=False`` to force that backend deterministically (independent of the optional
``quantile-forest`` package being installed). A regression that ignores ``forest_kind`` (always
building a RandomForest) collapses the ET-vs-RF gap and trips this test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite.qrf import CompositeQRFEstimator


def _smooth_noisy(seed: int, n: int = 1200):
    """y = 2*base + 1.5*sin(base) + 0.5*x1 + N(0, 1); returns (X, y, noise-free truth)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-3.0, 3.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    truth = 2.0 * base + 1.5 * np.sin(base) + 0.5 * x1
    y = truth + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"base": base, "x1": x1}), y, truth


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _fit_predict(kind: str, Xtr, ytr, Xte) -> np.ndarray:
    est = CompositeQRFEstimator(
        base_column="base",
        forest_kind=kind,
        n_estimators=60,
        min_samples_leaf=5,
        prefer_quantile_forest=False,
        random_state=0,
    )
    return est.fit(Xtr, ytr).predict(Xte)


def test_biz_val_qrf_extratrees_lowers_holdout_rmse_on_smooth_noisy_target():
    """ET backend must win majority of seeds and cut mean RMSE >=8% vs RF on a smooth+noisy target.

    Measured: ET wins 6/6 seeds, ~15-25% RMSE reduction; floor set well below to absorb seed noise.
    """
    rf_rmses, et_rmses, et_wins = [], [], 0
    for seed in range(6):
        X, y, truth = _smooth_noisy(seed)
        ntr = (len(y) * 4) // 5
        Xtr, ytr, Xte, truth_te = X.iloc[:ntr], y[:ntr], X.iloc[ntr:], truth[ntr:]
        r_rf = _rmse(_fit_predict("rf", Xtr, ytr, Xte), truth_te)
        r_et = _rmse(_fit_predict("et", Xtr, ytr, Xte), truth_te)
        rf_rmses.append(r_rf)
        et_rmses.append(r_et)
        et_wins += int(r_et < r_rf)

    mean_rf = float(np.mean(rf_rmses))
    mean_et = float(np.mean(et_rmses))
    assert et_wins >= 5, f"ET backend should beat RF on >=5/6 seeds, won {et_wins}/6"
    assert mean_et <= mean_rf * 0.92, (
        f"ET mean RMSE {mean_et:.4f} should be <=92% of RF {mean_rf:.4f} on smooth+noisy target"
    )


def test_biz_val_qrf_forest_kind_is_actually_consumed():
    """ET and RF must produce DIFFERENT predictions; identical output means forest_kind is ignored."""
    X, y, _ = _smooth_noisy(0)
    ntr = (len(y) * 4) // 5
    Xtr, ytr, Xte = X.iloc[:ntr], y[:ntr], X.iloc[ntr:]
    pred_rf = _fit_predict("rf", Xtr, ytr, Xte)
    pred_et = _fit_predict("et", Xtr, ytr, Xte)
    assert not np.allclose(pred_rf, pred_et), "forest_kind='et' vs 'rf' must change predictions"
