"""Naive-Bayes combination of per-feature posteriors must not count the base rate once per member."""

from __future__ import annotations

import numpy as np

from mlframe.calibration.ensembling import odds_ratio_combine


def test_uninformative_members_combine_back_to_the_base_rate():
    """Three members that each predict the base rate 0.1 carry no evidence, so the combination is 0.1 (it was 0.0014)."""
    p = np.full((50, 3), 0.1)
    np.testing.assert_allclose(odds_ratio_combine(p), 0.1, atol=1e-6)


def test_prior_none_keeps_the_historical_logit_sum():
    """prior=None restores the plain sum (negative control)."""
    p = np.full((5, 3), 0.1)
    assert abs(odds_ratio_combine(p, prior=None)[0] - 1.0 / (1.0 + 9.0**3)) < 1e-9


def test_explicit_prior_is_used():
    """With prior 0.5 the correction vanishes and equals the plain sum."""
    p = np.full((5, 3), 0.1)
    np.testing.assert_allclose(odds_ratio_combine(p, prior=0.5), odds_ratio_combine(p, prior=None), rtol=1e-9)
