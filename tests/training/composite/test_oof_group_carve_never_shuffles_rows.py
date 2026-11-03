"""The single-split OOF carve keeps groups whole instead of falling back to a row permutation."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.ensemble import _oof_holdout_predic_use_time_split


def _groups_disjoint(groups, train_idx, holdout_idx):
    """True when no group appears on both sides of the split."""
    return not set(groups[train_idx].tolist()) & set(groups[holdout_idx].tolist())


def test_oversized_group_is_skipped_not_row_shuffled():
    """A giant group that would swallow the whole frame no longer triggers a group-blind shuffle."""
    groups = np.concatenate([np.zeros(900, dtype=int), np.arange(1, 101)])
    for seed in range(20):
        holdout_idx, train_idx = _oof_holdout_predic_use_time_split(False, 1000, 100, None, groups, seed)
        assert _groups_disjoint(groups, train_idx, holdout_idx)
        assert len(holdout_idx) + len(train_idx) == 1000


def test_impossible_carve_raises_instead_of_leaking():
    """When one group holds every row but a few, no group-respecting holdout exists and the error says so."""
    groups = np.concatenate([np.zeros(996, dtype=int), np.arange(1, 5)])
    with pytest.raises(ValueError, match="group-respecting holdout"):
        _oof_holdout_predic_use_time_split(False, 1000, 1000, None, groups, 0)
