"""biz_value + unit tests for ``calibration.ensembling.odds_ratio_combine``.

The win: under conditional independence of member sub-models given y, log-odds summation recovers the TRUE
joint posterior essentially exactly, while naive arithmetic averaging of member probabilities does not (it
systematically under-states confidence when multiple independent members agree) — measured via log-loss
against ground truth on a known conditionally-independent generative model where the true combined posterior
has a closed form.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import log_loss

from mlframe.calibration.ensembling import odds_ratio_combine
from mlframe.calibration._independence_check import member_residual_correlation


def _make_conditionally_independent_data(n: int, k: int, seed: int):
    """K conditionally-independent Gaussian features given y: x_i | y=1 ~ N(+1, 1), x_i | y=0 ~ N(-1, 1).

    Closed-form per-feature posterior: logit(P(y=1|x_i)) = 2*x_i (symmetric +-1 means, unit variance).
    Closed-form TRUE joint posterior under conditional independence: logit(P(y=1|X)) = sum_i 2*x_i.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.float64)
    x = rng.standard_normal((n, k)) + np.where(y[:, None] == 1, 1.0, -1.0)
    member_probs = 1.0 / (1.0 + np.exp(-2.0 * x))
    true_combined_logit = (2.0 * x).sum(axis=1)
    true_combined_prob = 1.0 / (1.0 + np.exp(-true_combined_logit))
    return y, member_probs, true_combined_prob


def test_odds_ratio_combine_matches_closed_form_true_posterior():
    y, member_probs, true_combined_prob = _make_conditionally_independent_data(2000, 5, seed=0)
    combined = odds_ratio_combine(member_probs)
    assert np.allclose(combined, true_combined_prob, atol=1e-9)


def test_odds_ratio_combine_shape_and_range():
    _, member_probs, _ = _make_conditionally_independent_data(50, 3, seed=1)
    combined = odds_ratio_combine(member_probs)
    assert combined.shape == (50,)
    assert np.all(combined >= 0.0) and np.all(combined <= 1.0)


def test_odds_ratio_combine_weights_shape_mismatch_raises():
    _, member_probs, _ = _make_conditionally_independent_data(10, 3, seed=2)
    with pytest.raises(ValueError):
        odds_ratio_combine(member_probs, weights=np.array([1.0, 1.0]))


def test_odds_ratio_combine_requires_2d_input():
    with pytest.raises(ValueError):
        odds_ratio_combine(np.array([0.1, 0.2, 0.3]))


def test_odds_ratio_combine_single_member_is_identity():
    _, member_probs, _ = _make_conditionally_independent_data(30, 1, seed=3)
    combined = odds_ratio_combine(member_probs)
    assert np.allclose(combined, member_probs[:, 0], atol=1e-9)


def test_biz_val_odds_ratio_combine_beats_naive_mean_on_conditionally_independent_members():
    y, member_probs, true_combined_prob = _make_conditionally_independent_data(4000, 6, seed=7)

    combined = odds_ratio_combine(member_probs)
    naive_mean = member_probs.mean(axis=1)

    ll_combined = log_loss(y, combined, labels=[0.0, 1.0])
    ll_naive = log_loss(y, naive_mean, labels=[0.0, 1.0])
    ll_true = log_loss(y, true_combined_prob, labels=[0.0, 1.0])

    # odds_ratio_combine should essentially match the true closed-form posterior's log-loss (both derive from
    # the same conditional-independence combination rule), while naive averaging discards the "multiple
    # independent members agreeing" signal and is measurably worse. Floor set with margin below the measured gap.
    assert ll_combined == pytest.approx(ll_true, abs=1e-6)
    assert ll_naive > ll_combined + 0.05, (
        f"naive mean log-loss ({ll_naive:.4f}) should be clearly worse than odds-ratio-combined log-loss ({ll_combined:.4f}) under conditional independence"
    )


def _make_correlated_duplicate_members(n: int, k: int, seed: int, noise: float = 0.05):
    """K members that are near-duplicate noisy measurements of the SAME latent z (heavily correlated, not
    conditionally independent given y): member_j = sigmoid(z + small iid noise_j) for every j.

    True calibrated posterior only needs ONE measurement's worth of evidence: logit(P(y=1|z)) = z. Naive
    log-odds summation across k near-duplicate members sums k copies of ~z -> systematically over-confident
    (~k times too extreme in logit space), while the equal-weight log opinion pool (geometric mean of member
    odds, i.e. mean of the logits) recovers ~z and stays well-calibrated.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) * 1.5
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.float64)
    member_logits = z[:, None] + noise * rng.standard_normal((n, k))
    member_probs = 1.0 / (1.0 + np.exp(-member_logits))
    return y, member_probs


def test_member_residual_correlation_low_for_conditionally_independent_members():
    _, member_probs, _ = _make_conditionally_independent_data(3000, 6, seed=11)
    diag = member_residual_correlation(member_probs)
    # conditionally-independent members still show a moderate consensus correlation purely through their
    # shared dependence on y (measured ~0.65 here) -- well clear of the near-duplicate band (~0.99+) and
    # below the default correlation_threshold=0.85.
    assert diag["mean_abs_residual_correlation"] < 0.75


def test_member_residual_correlation_high_for_correlated_duplicate_members():
    _, member_probs = _make_correlated_duplicate_members(3000, 6, seed=12)
    diag = member_residual_correlation(member_probs)
    assert diag["mean_abs_residual_correlation"] > 0.9


def test_odds_ratio_combine_check_independence_default_off_is_bit_identical():
    """``check_independence`` is opt-in: omitting it (or leaving it False) must not change output at all,
    even on data that WOULD trigger a correlation warning if the check were enabled."""
    _, member_probs = _make_correlated_duplicate_members(500, 5, seed=13)
    baseline = odds_ratio_combine(member_probs)
    with_default_flag = odds_ratio_combine(member_probs, check_independence=False)
    assert np.array_equal(baseline, with_default_flag)


def test_odds_ratio_combine_check_independence_warns_but_keeps_sum_by_default_mode():
    _, member_probs = _make_correlated_duplicate_members(500, 5, seed=14)
    plain = odds_ratio_combine(member_probs)
    with pytest.warns(UserWarning, match="residual correlation"):
        warned = odds_ratio_combine(member_probs, check_independence=True, on_correlation_violation="warn")
    assert np.array_equal(plain, warned)


def test_odds_ratio_combine_check_independence_no_warning_on_independent_members():
    y, member_probs, _ = _make_conditionally_independent_data(2000, 5, seed=15)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = odds_ratio_combine(member_probs, check_independence=True, on_correlation_violation="fallback")
    plain = odds_ratio_combine(member_probs)
    assert np.array_equal(result, plain)


def test_biz_val_odds_ratio_combine_independence_fallback_wins_on_correlated_members():
    """Scenario (b): strongly correlated (near-duplicate) submodels. Naive log-odds summation double-counts
    the shared latent signal k times over and is systematically over-confident; the auto-fallback equal-
    weight log opinion pool (triggered by ``on_correlation_violation='fallback'``) recovers a much better
    calibrated combined probability, measured via Brier score against the true labels."""
    y, member_probs = _make_correlated_duplicate_members(4000, 12, seed=21)

    with pytest.warns(UserWarning, match="residual correlation"):
        fallback_combined = odds_ratio_combine(member_probs, check_independence=True, on_correlation_violation="fallback")
    naive_sum_combined = odds_ratio_combine(member_probs)

    brier_fallback = float(np.mean((fallback_combined - y) ** 2))
    brier_naive_sum = float(np.mean((naive_sum_combined - y) ** 2))

    # naive_sum should be measurably worse than fallback; thresholds set with margin below the measured gap
    # (measured with k=12 members: fallback ~0.174, naive_sum ~0.235, a ~0.06 gap).
    assert brier_fallback < 0.20
    assert brier_naive_sum > brier_fallback + 0.04, (
        f"naive log-odds-sum Brier ({brier_naive_sum:.4f}) should be clearly worse than the "
        f"independence-fallback Brier ({brier_fallback:.4f}) on strongly correlated members"
    )


def test_biz_val_odds_ratio_combine_independence_check_scenario_a_prefers_odds_ratio():
    """Scenario (a): genuinely conditionally-independent submodels. The independence check must NOT trigger
    a fallback here -- plain odds-ratio combination stays preferred/kept, and remains the best option
    (matches the closed-form true posterior essentially exactly, unlike naive averaging)."""
    y, member_probs, true_combined_prob = _make_conditionally_independent_data(4000, 6, seed=22)

    diag = member_residual_correlation(member_probs)
    assert diag["mean_abs_residual_correlation"] < 0.85  # stays under the default correlation_threshold

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checked = odds_ratio_combine(member_probs, check_independence=True, on_correlation_violation="fallback")

    ll_checked = log_loss(y, checked, labels=[0.0, 1.0])
    ll_true = log_loss(y, true_combined_prob, labels=[0.0, 1.0])
    assert ll_checked == pytest.approx(ll_true, abs=1e-6)
