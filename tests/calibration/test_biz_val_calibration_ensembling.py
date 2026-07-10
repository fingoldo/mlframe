"""biz_value + unit tests for ``calibration.ensembling.odds_ratio_combine``.

The win: under conditional independence of member sub-models given y, log-odds summation recovers the TRUE
joint posterior essentially exactly, while naive arithmetic averaging of member probabilities does not (it
systematically under-states confidence when multiple independent members agree) — measured via log-loss
against ground truth on a known conditionally-independent generative model where the true combined posterior
has a closed form.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss

from mlframe.calibration.ensembling import odds_ratio_combine


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
        f"naive mean log-loss ({ll_naive:.4f}) should be clearly worse than "
        f"odds-ratio-combined log-loss ({ll_combined:.4f}) under conditional independence"
    )
