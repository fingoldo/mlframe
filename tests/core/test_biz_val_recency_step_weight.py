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

    assert rmse_weighted < rmse_unweighted * 0.95, (
        f"expected recency step-weighting to improve post-regime-shift RMSE by >=5%, got weighted={rmse_weighted:.4f} unweighted={rmse_unweighted:.4f}"
    )


def test_recency_step_weight_assigns_correct_levels():
    dates = np.array([1, 2, 3, 4, 5])
    w = recency_step_weight(dates, cutoff_date=3, boost=2.0, base=0.5)
    np.testing.assert_array_equal(w, [0.5, 0.5, 2.0, 2.0, 2.0])


def test_recency_step_weight_default_base_and_boost():
    dates = np.array([10, 20, 30])
    w = recency_step_weight(dates, cutoff_date=20)
    np.testing.assert_array_equal(w, [1.0, 1.5, 1.5])


def _make_gradual_drift_data(n: int, drift_start: int, drift_end: int, seed: int):
    """Relationship strength ramps linearly from the old regime to the new one over [drift_start, drift_end]
    instead of flipping sharply at a single day -- the scenario a single hard cutoff can't capture well."""
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    X = rng.normal(size=(n, 10))
    progress = np.clip((days - drift_start) / (drift_end - drift_start), 0.0, 1.0)
    beta_per_row = 1.0 - 2.0 * progress  # 1.0 (old regime) ramping down to -1.0 (new regime).
    y = X[:, 0] * beta_per_row + rng.normal(scale=0.2, size=n)
    return X, y, days


def test_biz_val_recency_step_weight_smooth_beats_hard_cutoff_on_gradual_drift():
    n, drift_start, drift_end = 4000, 1000, 3600
    X, y, days = _make_gradual_drift_data(n=n, drift_start=drift_start, drift_end=drift_end, seed=0)
    Xtr, ytr, days_tr = X[:3800], y[:3800], days[:3800]
    Xte, yte = X[3800:], y[3800:]  # held-out period sits at the fully-drifted end of the ramp.

    model_kwargs = dict(n_estimators=100, num_leaves=15, random_state=0, verbose=-1)
    boost = 5.0

    # a hard cutoff at the drift's start gives full boost to the WHOLE (long) transition window, including
    # early rows still closer to the OLD regime -- overweighting a still-mixed relationship.
    hard_weights = recency_step_weight(days_tr, cutoff_date=drift_start, boost=boost, base=1.0)
    m_hard = LGBMRegressor(**model_kwargs).fit(Xtr, ytr, sample_weight=hard_weights)

    # the smooth ramp instead grows the weight in step with how far the relationship has actually drifted,
    # at the same peak boost, without over-trusting the still-mixed early-transition rows.
    smooth_weights = recency_step_weight(days_tr, cutoff_date=drift_end, boost=boost, base=1.0, smooth_window=drift_end - drift_start)
    m_smooth = LGBMRegressor(**model_kwargs).fit(Xtr, ytr, sample_weight=smooth_weights)

    rmse_hard = float(mean_squared_error(yte, m_hard.predict(Xte)) ** 0.5)
    rmse_smooth = float(mean_squared_error(yte, m_smooth.predict(Xte)) ** 0.5)

    assert rmse_smooth < rmse_hard * 0.95, (
        f"expected smooth-ramp recency weighting to beat a single hard cutoff by >=5% RMSE on gradual drift, got smooth={rmse_smooth:.4f} hard={rmse_hard:.4f}"
    )


def test_recency_step_weight_tiers_assigns_ladder_levels():
    dates = np.array([1, 2, 3, 4, 5, 6])
    w = recency_step_weight(dates, cutoff_date=0, tiers=[(2, 1.5), (4, 2.0), (6, 3.0)], base=1.0)
    np.testing.assert_array_equal(w, [1.0, 1.5, 1.5, 2.0, 2.0, 3.0])


def test_recency_step_weight_smooth_window_ramps_linearly():
    dates = np.array([0.0, 5.0, 10.0])
    w = recency_step_weight(dates, cutoff_date=10.0, boost=2.0, base=1.0, smooth_window=10.0)
    np.testing.assert_allclose(w, [1.0, 1.5, 2.0])


def test_recency_step_weight_omitting_new_params_is_bit_identical_to_baseline():
    dates = np.linspace(0, 1000, 5000)
    baseline = recency_step_weight(dates, cutoff_date=500.0, boost=2.5, base=1.0)
    extended = recency_step_weight(dates, cutoff_date=500.0, boost=2.5, base=1.0, tiers=None, smooth_window=None)
    np.testing.assert_array_equal(baseline, extended)


def test_recency_step_weight_tiers_and_smooth_window_together_raises():
    dates = np.array([1, 2, 3])
    try:
        recency_step_weight(dates, cutoff_date=0, tiers=[(1, 2.0)], smooth_window=1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError when both tiers and smooth_window are set")
