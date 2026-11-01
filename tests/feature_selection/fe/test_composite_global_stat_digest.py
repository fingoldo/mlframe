"""A caller-supplied content digest gives the same global statistic as the digest the function computes itself."""

import numpy as np

from mlframe.feature_selection.filters import _composite_group_agg_fe as C
from mlframe.utils.disk_cache import hash_array_content


def test_supplied_digest_equals_computed(monkeypatch):
    """Same value and same memo entry for every statistic."""
    monkeypatch.setattr(C, "_GLOBAL_STAT_MEMO", {})
    x = np.random.default_rng(0).normal(size=500)
    digest = hash_array_content(np.ascontiguousarray(x))
    for stat in ("mean", "std", "median"):
        assert C._global_value_for_stat(x, stat, digest) == C._global_value_for_stat(x, stat)
    assert len(C._GLOBAL_STAT_MEMO) == 3
