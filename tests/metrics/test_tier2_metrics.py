"""Tests for Tier 2 metrics:
- Tweedie / Poisson / Gamma deviance (regression GLM family)
- Hosmer-Lemeshow chi-square (binary calibration)
- Accuracy Ratio (binary, banking convention for Gini)
- CRPS from quantiles (regression probabilistic forecasting)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.metrics.core import (
    fast_poisson_deviance, fast_gamma_deviance, fast_tweedie_deviance,
    hosmer_lemeshow_test, accuracy_ratio, fast_roc_auc,
)
from mlframe.metrics.quantile import crps_from_quantiles, pinball_loss


# ----- Tweedie family -----


def test_poisson_deviance_matches_sklearn():
    from sklearn.metrics import mean_poisson_deviance
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, 500).astype(np.float64)
    p = np.maximum(0.1, y + rng.normal(0, 0.5, 500))
    assert fast_poisson_deviance(y, p) == pytest.approx(
        mean_poisson_deviance(y, p), abs=1e-12,
    )


def test_gamma_deviance_matches_sklearn():
    from sklearn.metrics import mean_gamma_deviance
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 1.0, 500)
    p = np.maximum(1e-3, y + rng.normal(0, 0.1, 500))
    assert fast_gamma_deviance(y, p) == pytest.approx(
        mean_gamma_deviance(y, p), abs=1e-12,
    )


def test_tweedie_general_matches_sklearn():
    from sklearn.metrics import mean_tweedie_deviance
    rng = np.random.default_rng(2)
    y = rng.gamma(2.0, 1.0, 500)
    p = np.maximum(1e-3, y + rng.normal(0, 0.1, 500))
    for power in (1.5, 1.7, 2.5, 3.0):
        ours = fast_tweedie_deviance(y, p, power=power)
        skl = mean_tweedie_deviance(y, p, power=power)
        assert ours == pytest.approx(skl, abs=1e-10), f"power={power}"


def test_tweedie_power_0_is_mse():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(100)
    p = y + rng.standard_normal(100)
    expected = float(np.mean((y - p) ** 2))
    assert fast_tweedie_deviance(y, p, power=0.0) == pytest.approx(expected, abs=1e-12)


def test_tweedie_rejects_invalid_power():
    y = np.array([1.0, 2.0])
    p = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        fast_tweedie_deviance(y, p, power=0.5)


def test_poisson_zero_pred_skipped_with_warning():
    """y_pred=0 is undefined for log; rows are skipped + warning fires."""
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([0.0, 2.0, 3.0])  # first row invalid
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = fast_poisson_deviance(y, p)
        assert any("poisson" in str(rec.message).lower() for rec in w)
    assert np.isfinite(val)


def test_gamma_zero_y_skipped():
    """y=0 not in gamma support."""
    y = np.array([0.0, 1.0, 2.0])
    p = np.array([0.5, 1.0, 2.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = fast_gamma_deviance(y, p)
        assert any("gamma" in str(rec.message).lower() for rec in w)
    assert np.isfinite(val)


def test_poisson_deviance_zero_on_perfect_prediction():
    rng = np.random.default_rng(4)
    y = rng.poisson(5.0, 200).astype(np.float64)
    # y_pred = y exactly -> deviance = 0
    assert fast_poisson_deviance(y, y.copy()) == pytest.approx(0.0, abs=1e-10)


def test_gamma_deviance_zero_on_perfect_prediction():
    rng = np.random.default_rng(5)
    y = rng.gamma(2.0, 1.0, 200)
    assert fast_gamma_deviance(y, y.copy()) == pytest.approx(0.0, abs=1e-10)


# ----- Hosmer-Lemeshow -----


def test_hl_well_calibrated_high_p_value():
    """Sample y from Bernoulli(p) for known p -> HL should NOT reject."""
    rng = np.random.default_rng(6)
    N = 2000
    p = rng.uniform(0.05, 0.95, N)
    y = (rng.uniform(size=N) < p).astype(np.int64)
    chi2, p_value, dof = hosmer_lemeshow_test(y, p, n_groups=10)
    assert dof == 8
    # Under the null (well-calibrated), p > 0.05 should hold typically.
    # We use a softer threshold (p > 0.01) to keep the test non-flaky.
    assert p_value > 0.01


def test_hl_miscalibrated_low_p_value():
    """Predicting 0.5 everywhere when base rate is 0.5 BUT individual rows
    are deterministic (y=0 or y=1 based on index) -> HL detects the
    miscalibration because per-decile expected diverges from observed."""
    N = 1000
    y = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.int64)
    # Score the way that DOESN'T match y - constant 0.5 everywhere.
    s = np.full(N, 0.5)
    chi2, p_value, dof = hosmer_lemeshow_test(y, s, n_groups=10)
    # All deciles have E = 0.5 * group_size; all O are either 0 or
    # group_size (the deterministic split). chi2 will be large.
    assert chi2 > 100.0
    assert p_value < 0.001


def test_hl_handles_tiny_input():
    """N < n_groups -> NaN with dof=0."""
    y = np.array([0, 1, 0])
    s = np.array([0.1, 0.5, 0.9])
    chi2, p, dof = hosmer_lemeshow_test(y, s, n_groups=10)
    assert np.isnan(chi2)
    assert dof == 0


def test_hl_n_groups_parameter():
    """5 deciles -> dof = 3."""
    rng = np.random.default_rng(7)
    N = 1000
    y = (rng.uniform(size=N) > 0.5).astype(np.int64)
    s = rng.uniform(size=N)
    _, _, dof = hosmer_lemeshow_test(y, s, n_groups=5)
    assert dof == 3


# ----- Accuracy Ratio -----


def test_accuracy_ratio_matches_gini_from_auc():
    """AR (computed via CAP trapezoid integral) must equal 2*AUC - 1
    to fp tolerance regardless of how the AUC was computed."""
    rng = np.random.default_rng(8)
    for seed in range(5):
        rng2 = np.random.default_rng(seed)
        N = 1000
        y = (rng2.uniform(size=N) > 0.6).astype(np.int64)
        s = np.clip(0.3 + 0.4 * y + rng2.normal(0, 0.15, N), 0.001, 0.999)
        ar = accuracy_ratio(y, s)
        auc = fast_roc_auc(y.astype(np.float64), s)
        # AR is computed via CAP-area trapezoid; AUC is computed via
        # rank-sum. Different algorithms, same theoretical value, so
        # the diff should be bounded by trapezoidal-discretisation error
        # (which is tiny on N>=1000 with no tied scores).
        assert ar == pytest.approx(2.0 * auc - 1.0, abs=1e-3)


def test_accuracy_ratio_single_class_returns_nan():
    y = np.zeros(100, dtype=np.int64)
    s = np.random.default_rng(0).uniform(size=100)
    assert np.isnan(accuracy_ratio(y, s))


def test_accuracy_ratio_perfect_classifier_is_one():
    """Perfect ranking -> AR = 1 (model captures all positives at the top)."""
    y = np.concatenate([np.ones(30), np.zeros(70)]).astype(np.int64)
    s = -np.arange(100, dtype=np.float64)  # first 30 highest
    assert accuracy_ratio(y, s) == pytest.approx(1.0, abs=1e-10)


# ----- CRPS from quantiles -----


def test_crps_zero_on_perfect_point_predictions():
    """If every quantile prediction = y exactly, CRPS = 0."""
    N = 50
    rng = np.random.default_rng(9)
    y = rng.standard_normal(N)
    alphas = np.array([0.1, 0.5, 0.9])
    preds_NK = np.repeat(y[:, None], 3, axis=1)
    assert crps_from_quantiles(y, preds_NK, alphas) == pytest.approx(0.0, abs=1e-12)


def test_crps_monotone_with_residual_size():
    """Wider quantile residuals -> larger CRPS."""
    rng = np.random.default_rng(10)
    N = 200
    y = rng.standard_normal(N)
    alphas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    preds_tight = np.stack([y + rng.normal(0, 0.1, N) for _ in alphas], axis=1)
    preds_wide = np.stack([y + rng.normal(0, 1.0, N) for _ in alphas], axis=1)
    preds_tight = np.sort(preds_tight, axis=1)
    preds_wide = np.sort(preds_wide, axis=1)
    crps_tight = crps_from_quantiles(y, preds_tight, alphas)
    crps_wide = crps_from_quantiles(y, preds_wide, alphas)
    assert crps_tight < crps_wide


def test_crps_rejects_unsorted_alphas():
    y = np.array([1.0, 2.0])
    preds = np.array([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError):
        crps_from_quantiles(y, preds, np.array([0.5, 0.2]))


def test_crps_rejects_mismatched_shapes():
    y = np.array([1.0, 2.0, 3.0])
    preds = np.array([[1.0, 2.0], [2.0, 3.0]])  # only 2 rows
    with pytest.raises(ValueError):
        crps_from_quantiles(y, preds, np.array([0.25, 0.75]))


def test_crps_proportional_to_pinball():
    """CRPS == 2 * integral_0^1 pinball(alpha) d alpha (Gneiting & Raftery identity).

    The same prediction is used for every alpha column, so the predicted quantile function is a
    flat constant clamped over the WHOLE [0, 1] range. ``crps_from_quantiles`` integrates the full
    [0, 1] (constant-extrapolation tail clamp below ``a[0]`` and above ``a[-1]``), NOT just the
    [0.4, 0.6] window of the supplied grid -- dropping the tails would under-estimate CRPS and break
    cross-grid comparability. With a constant predictor the per-alpha pinball is near-flat ~= pl_05,
    so CRPS ~= 2 * pl_05 over the full unit interval.
    """
    rng = np.random.default_rng(11)
    N = 500
    y = rng.standard_normal(N)
    median_pred = y + rng.normal(0, 0.5, N)
    alphas = np.array([0.4, 0.5, 0.6])
    preds_NK = np.stack([median_pred] * 3, axis=1)
    crps = crps_from_quantiles(y, preds_NK, alphas)
    pl_05 = pinball_loss(y, median_pred, 0.5)
    # Full-range integral of a near-constant pinball: 2 * 1.0 * pl_05. The tolerance absorbs the
    # tiny alpha-tilt of pinball across [0, 1] for this single-point clamped predictor.
    assert crps == pytest.approx(2.0 * pl_05, rel=0.01)
