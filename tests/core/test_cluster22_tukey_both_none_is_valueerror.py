"""Cluster 22 regression: both sd_sigma and nonoutlying_dist_percentage unset gives a ValueError, not a TypeError (audit 2026-11-20, finding 22-F32)."""

import pytest

from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile


def test_both_unset_is_value_error():
    """sd_sigma=None with nonoutlying_dist_percentage=None names the parameter (22-F32)."""
    with pytest.raises(ValueError, match="nonoutlying_dist_percentage"):
        get_tukey_fences_multiplier_for_quantile(quantile=0.25, sd_sigma=None, nonoutlying_dist_percentage=None)


def test_explicit_sd_sigma_still_works():
    """Control: a supplied sd_sigma is used as before (22-F32)."""
    assert round(get_tukey_fences_multiplier_for_quantile(quantile=0.25, sd_sigma=2.7), 6) == 1.501513
