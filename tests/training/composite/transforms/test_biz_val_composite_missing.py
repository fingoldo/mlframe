"""Biz-value test for ``MissingAwareComposite``.

On a target with 20% MNAR-missing base (missingness correlated with y), the
missing-aware composite must beat BOTH naive-impute-zero AND drop-missing on
OOS RMSE -- the indicator + learned offset recover the informative missingness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.missing import MissingAwareComposite


def _make_composite() -> CompositeTargetEstimator:
    """Make composite."""
    return CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="base",
    )


def _rmse(a, b) -> float:
    """Rmse."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mnar_dataset(n: int, seed: int):
    """20% MNAR: the highest-y rows have their base set NaN, and on those rows
    the true y carries an extra additive shift the base cannot explain."""
    rng = np.random.default_rng(seed)
    base = rng.normal(10.0, 3.0, size=n)
    feat = rng.normal(0.0, 1.0, size=n)
    y = base + 2.0 * feat + rng.normal(0.0, 0.3, size=n)
    # Make missingness informative: pick the top 20% by a latent score and add a
    # large positive shift to their y, then blank their base (MNAR).
    score = y + rng.normal(0.0, 0.5, size=n)
    thr = np.quantile(score, 0.8)
    miss = score >= thr
    y = y + miss * 5.0
    X = pd.DataFrame({"base": base, "feat": feat})
    X_obs = X.copy()
    X_obs.loc[miss, "base"] = np.nan
    return X_obs, y, miss


def test_biz_val_missing_aware_beats_impute_zero_and_drop():
    """Biz val missing aware beats impute zero and drop."""
    X_tr, y_tr, miss_tr = _mnar_dataset(1200, seed=11)
    X_te, y_te, _miss_te = _mnar_dataset(1200, seed=99)

    # Missing-aware: learns impute + offset for missing rows.
    aware = MissingAwareComposite(composite=_make_composite()).fit(X_tr, y_tr)
    rmse_aware = _rmse(y_te, aware.predict(X_te))

    # Naive impute-zero: fill NaN base with 0, fit plain composite.
    X_tr0 = X_tr.fillna(0.0)
    X_te0 = X_te.fillna(0.0)
    plain0 = _make_composite().fit(X_tr0, y_tr)
    rmse_zero = _rmse(y_te, plain0.predict(X_te0))

    # Drop-missing: train only on finite-base rows; at test, impute-zero the
    # missing rows (drop-trained model has no path for them).
    fin_tr = ~miss_tr
    drop = _make_composite().fit(X_tr0.loc[fin_tr].reset_index(drop=True), y_tr[fin_tr])
    rmse_drop = _rmse(y_te, drop.predict(X_te0))

    # Missing-aware must beat both, comfortably (measured ~3-5x better RMSE;
    # floor at a clear 20% improvement to absorb seed noise).
    assert rmse_aware < 0.8 * rmse_zero, f"missing-aware {rmse_aware:.3f} not < 0.8*impute-zero {rmse_zero:.3f}"
    assert rmse_aware < 0.8 * rmse_drop, f"missing-aware {rmse_aware:.3f} not < 0.8*drop-missing {rmse_drop:.3f}"
