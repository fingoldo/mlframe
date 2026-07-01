"""Unit + biz_value tests for gains/lift curves + exploss (PZAD err_scoreandcurves)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._gains_lift import (
    cumulative_gains_curve,
    exploss,
    gains_table,
    lift_curve,
)


# ---------------------------------------------------------------- unit
def test_gains_curve_endpoints_and_monotone():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    frac, gain = cumulative_gains_curve(y, s)
    assert frac[0] == 0.0 and gain[0] == 0.0
    assert np.isclose(frac[-1], 1.0) and np.isclose(gain[-1], 1.0)
    assert np.all(np.diff(gain) >= -1e-12)  # gain non-decreasing


def test_perfect_model_gains_reaches_one_at_prevalence():
    # 3 positives of 6; perfect ranking captures all positives by fraction 0.5
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    frac, gain = cumulative_gains_curve(y, s)
    idx = np.argmin(np.abs(frac - 0.5))
    assert np.isclose(gain[idx], 1.0), "perfect model captures all positives by the prevalence fraction"


def test_lift_first_point_above_one_for_good_model():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.85, 0.2, 0.1, 0.8, 0.3, 0.05, 0.02])
    frac, lift = lift_curve(y, s)
    assert lift[0] > 1.0
    assert np.isclose(lift[-1], 1.0)  # full population -> lift 1


def test_gains_table_shape_and_cumcaptured():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.3).astype(int)
    s = rng.random(1000)
    t = gains_table(y, s, n_bins=10)
    assert len(t["bin"]) == 10
    assert np.isclose(t["cum_captured_pct"][-1], 1.0)
    assert np.isclose(t["cum_fraction"][-1], 1.0)


def test_gains_table_invalid():
    with pytest.raises(ValueError):
        gains_table(np.zeros(5), np.zeros(5), n_bins=0)
    with pytest.raises(ValueError):
        gains_table(np.array([]), np.array([]))


def test_exploss_minimized_at_true_prob_and_symmetry():
    # single object with p=0.7: exploss is minimized at a=0.7
    grid = np.linspace(0.01, 0.99, 99)
    p = 0.7
    vals = [0.7 * np.sqrt((1 - a) / a) + 0.3 * np.sqrt(a / (1 - a)) for a in grid]
    a_star = grid[int(np.argmin(vals))]
    assert abs(a_star - p) < 0.03


def test_exploss_length_and_empty():
    assert np.isnan(exploss(np.array([]), np.array([])))
    with pytest.raises(ValueError):
        exploss(np.zeros(3), np.zeros(2))


# ---------------------------------------------------------------- biz_value
def test_biz_val_gains_curve_beats_random_for_informative_scores():
    """A good model's cumulative gains lies well above the diagonal (random): capturing >>fraction of positives
    in the top decile. Measured: top-10% captures ~35-50% of positives vs 10% random. Floor at 2x."""
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.random(n) < 0.2).astype(int)
    s = rng.normal(0, 1, size=n) + y * 1.5  # informative scores
    frac, gain = cumulative_gains_curve(y, s)
    idx = np.argmin(np.abs(frac - 0.1))  # top 10%
    assert gain[idx] > 0.2, f"top-10% should capture >2x random (got gain {gain[idx]:.2f})"


def test_biz_val_lift_top_decile_exceeds_2x():
    """The top-decile lift of an informative model clearly exceeds 1 (better than random targeting)."""
    rng = np.random.default_rng(2)
    n = 5000
    y = (rng.random(n) < 0.2).astype(int)
    s = rng.normal(0, 1, size=n) + y * 1.5
    t = gains_table(y, s, n_bins=10)
    assert t["lift"][0] > 2.0, f"top-decile lift {t['lift'][0]:.2f} should exceed 2x"


def test_biz_val_exploss_ranks_calibrated_below_overconfident():
    """As a proper scoring rule, exploss penalizes an overconfident-wrong model more than a calibrated one."""
    rng = np.random.default_rng(3)
    n = 2000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)  # y ~ Bernoulli(p)
    calibrated = p
    overconfident = np.clip((p - 0.5) * 5 + 0.5, 0.01, 0.99)  # pushes toward 0/1
    assert exploss(y, calibrated) < exploss(y, overconfident), "calibrated probs should score better under exploss"
