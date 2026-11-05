"""Degenerate inputs raise instead of returning a number that means nothing.

The Tukey fence multiplier is undefined at the median (its denominator is zero, so the earlier code returned inf with only a RuntimeWarning, and
lru_cache kept the inf). The two-proportion test accepted counts larger than their sample size and answered as if they were valid.
"""

from __future__ import annotations

import pytest

from mlframe.core.proportion_stats import proportions_significantly_different
from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile


def test_tukey_multiplier_at_the_median_raises():
    """quantile=0.5 has no finite fence multiplier: the call raises instead of returning inf."""
    with pytest.raises(ValueError, match="median"):
        get_tukey_fences_multiplier_for_quantile(quantile=0.5, sd_sigma=2.7)


def test_tukey_multiplier_away_from_the_median_is_unchanged():
    """A quantile that is not the median still returns the finite multiplier."""
    value = get_tukey_fences_multiplier_for_quantile(quantile=0.1, sd_sigma=2.7)
    assert value == pytest.approx(0.5534105972, abs=1e-9)


def test_proportion_test_rejects_more_successes_than_trials():
    """12 successes out of 10 trials is impossible: the test raises instead of answering."""
    with pytest.raises(ValueError, match="0 <= m <= n"):
        proportions_significantly_different(12, 10, 5, 10)


def test_proportion_test_still_answers_valid_counts():
    """Valid counts keep their answer: a 5/10 versus 9/10 gap is significant at the default level."""
    assert proportions_significantly_different(5, 10, 9, 10) in (True, False)
