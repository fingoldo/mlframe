"""Unit + biz_value tests for proportion CI / sample size (PZAD probweights slides 45-47)."""

from __future__ import annotations

import pytest

from mlframe.core.proportion_stats import (
    proportions_significantly_different,
    required_n_for_proportion,
    wilson_interval,
    z_for_confidence,
)


def test_z_for_confidence_known_values():
    """Z for confidence known values."""
    assert abs(z_for_confidence(0.95) - 1.959964) < 1e-3
    assert abs(z_for_confidence(0.99) - 2.575829) < 1e-3


def test_wilson_interval_contains_phat_and_within_unit():
    """Wilson interval contains phat and within unit."""
    lo, hi = wilson_interval(50, 100, confidence=0.95)
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_wilson_interval_edge_zero_successes():
    """Wilson interval edge zero successes."""
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.1


def test_wilson_invalid():
    """Wilson invalid."""
    with pytest.raises(ValueError):
        wilson_interval(10, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_biz_val_required_n_reproduces_lecture_10000():
    """The lecture claims ~10000 samples give +/-0.01 precision at 99%. Worst-case p=0.5:
    n = 2.576^2 * 0.25 / 0.01^2 ~= 16590; at 95% ~= 9604 ('n=10000'). Pin both regimes."""
    n99 = required_n_for_proportion(0.01, confidence=0.99)
    n95 = required_n_for_proportion(0.01, confidence=0.95)
    assert 16000 <= n99 <= 17000
    assert 9000 <= n95 <= 10000  # the lecture's ~10000 rule of thumb


def test_biz_val_required_n_shrinks_with_known_small_p():
    """Supplying a small expected p reduces the required n vs the worst-case p=0.5."""
    n_worst = required_n_for_proportion(0.01, confidence=0.95, p=0.5)
    n_small = required_n_for_proportion(0.01, confidence=0.95, p=0.05)
    assert n_small < n_worst / 4


def test_biz_val_zodiac_gap_not_significant_but_huge_n_is():
    """The zodiac-scoring lesson: a ~1pp gap is NOT significant at n=1000 per group but IS at n=250000."""
    # 35.3% vs 34.2% (Овен vs Рыбы from the slide)
    small = proportions_significantly_different(353, 1000, 342, 1000, confidence=0.95)
    huge = proportions_significantly_different(88250, 250000, 85500, 250000, confidence=0.95)
    assert small is False, "1pp gap at n=1000 should not be significant"
    assert huge is True, "1pp gap at n=250000 should be significant"
