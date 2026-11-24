"""Cluster 37 regressions: group-disjoint OOF subsample and NaN-safe ensemble gate (audit 2026-11-20, findings 37-F13, 37-F14)."""

import numpy as np

from mlframe.training.core._phase_composite_post_xt_ensemble import _oof_subsample_positions


def test_group_subsample_stays_group_disjoint_when_one_group_covers_most_rows():
    """A huge group that would make the kept groups cover every row must not trigger a plain random draw (37-F13)."""
    groups = np.array([0] * 10 + [1] * 990)
    for seed in range(10):  # both group orders are exercised across seeds
        pos = _oof_subsample_positions(1000, groups, cap=500, seed=seed)
        if pos is None:
            continue  # no group-disjoint subsample exists for this order; full set is group-disjoint by construction
        assert pos.size < 1000, "kept groups cover every row, so the subsample is not a subsample"
        kept = set(groups[pos].tolist())
        assert kept, "the subsample kept no group"
        for g in kept:
            assert np.all(np.isin(np.nonzero(groups == g)[0], pos)), "a group was split by the subsample"


def test_group_subsample_keeps_whole_groups_in_ordinary_case():
    """Control: with several small groups the subsample keeps whole groups only."""
    groups = np.repeat(np.arange(100), 10)
    pos = _oof_subsample_positions(1000, groups, cap=300, seed=1)
    assert pos is not None and pos.size == 300, f"ten 10-row groups under cap 300 fill it exactly; got {None if pos is None else pos.size}"
    kept = set(groups[pos].tolist())
    assert kept, "the subsample kept no group"
    for g in kept:
        assert np.all(np.isin(np.nonzero(groups == g)[0], pos))
