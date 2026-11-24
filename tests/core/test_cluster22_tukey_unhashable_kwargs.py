"""Cluster 22 regression: an unhashable distribution kwarg does not raise TypeError from lru_cache (audit 2026-11-20, finding 22-F32).

The cached helper only receives hashable arguments; extra dist_kwargs bypass the cache. A numpy-array loc gives the same multiplier as the
scalar loc, and the common cached path is unchanged.
"""

import numpy as np

from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile as g


def test_unhashable_dist_kwarg_does_not_raise_typeerror():
    """A numpy-array loc is accepted and gives the same multiplier as the scalar (22-F32)."""
    scalar = g(0.25, 2.7, loc=0.0)
    array = g(0.25, 2.7, loc=np.array([0.0]))
    assert np.isclose(scalar, array)


def test_common_cached_path_still_matches_the_documented_value():
    """Control: the cached default path returns the documented multiplier (22-F32)."""
    assert round(g(quantile=0.25, sd_sigma=2.7), 6) == 1.501513
