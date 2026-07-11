"""biz_value test for ``core.recency_step_weight.recency_step_weight``.

Source: 9th_optiver-trading-at-the-close.md -- "xgb sample_weight 1.5 weight for latest 45 days data." When
the true relationship between features and target drifts (a regime shift), older rows encode a STALE
relationship the model shouldn't weight equally with recent, still-relevant rows. Upweighting rows within the
recent cutoff should improve held-out accuracy on a held-out period that is entirely in the new regime,
versus training with uniform weights.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from mlframe.core.recency_step_weight import recency_step_weight


def _make_regime_shift_data(n: int, shift_day: int, seed: int):
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    X = rng.normal(size=(n, 10))
    beta_per_row = np.where(days < shift_day, 1.0, -1.0)  # relationship flips sign after the shift.
    y = X[:, 0] * beta_per_row + rng.normal(scale=0.3, size=n)
    return X, y, days


def test_biz_val_recency_step_weight_improves_held_out_accuracy_after_regime_shift():
    shift_day = 2500
    X, y, days = _make_regime_shift_data(n=3000, shift_day=shift_day, seed=0)
    Xtr, ytr, days_tr = X[:2800], y[:2800], days[:2800]
    Xte, yte = X[2800:], y[2800:]  # held-out period is entirely in the new (post-shift) regime.

    model_kwargs = dict(n_estimators=100, num_leaves=15, random_state=0, verbose=-1)
    m_unweighted = LGBMRegressor(**model_kwargs).fit(Xtr, ytr)

    weights = recency_step_weight(days_tr, cutoff_date=shift_day, boost=3.0, base=1.0)
    m_weighted = LGBMRegressor(**model_kwargs).fit(Xtr, ytr, sample_weight=weights)

    rmse_unweighted = float(mean_squared_error(yte, m_unweighted.predict(Xte)) ** 0.5)
    rmse_weighted = float(mean_squared_error(yte, m_weighted.predict(Xte)) ** 0.5)

    assert rmse_weighted < rmse_unweighted * 0.95, f"expected recency step-weighting to improve post-regime-shift RMSE by >=5%, got weighted={rmse_weighted:.4f} unweighted={rmse_unweighted:.4f}"


def test_recency_step_weight_assigns_correct_levels():
    dates = np.array([1, 2, 3, 4, 5])
    w = recency_step_weight(dates, cutoff_date=3, boost=2.0, base=0.5)
    np.testing.assert_array_equal(w, [0.5, 0.5, 2.0, 2.0, 2.0])


def test_recency_step_weight_default_base_and_boost():
    dates = np.array([10, 20, 30])
    w = recency_step_weight(dates, cutoff_date=20)
    np.testing.assert_array_equal(w, [1.0, 1.5, 1.5])
