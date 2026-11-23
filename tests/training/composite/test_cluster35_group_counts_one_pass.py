"""Audit 2026-11-20 finding 35-F16: the group-holdout row counts come from one np.unique pass and match the per-group comparison."""

import numpy as np

from mlframe.training.composite.ensemble import _group_counts


def test_group_counts_match_per_group_comparison():
    """Counts equal the (group == gid).sum() values the old loop computed, for string and integer ids."""
    rng = np.random.default_rng(0)
    for groups in (rng.integers(0, 7, size=300), np.array(["a", "bb", "a", "c", "bb", "a"])):
        got = _group_counts(groups)
        assert len(got) == len(np.unique(groups)) > 0
        for gid in np.unique(groups):
            assert got[gid] == int((groups == gid).sum())
        assert sum(got.values()) == len(groups)
