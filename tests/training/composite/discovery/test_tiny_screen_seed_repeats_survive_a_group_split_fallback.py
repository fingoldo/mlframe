"""Seed repeats collapse only when the group split really is seed-invariant."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._screening_tiny import _split_is_seed_invariant


def test_enough_groups_make_the_split_seed_invariant():
    """Five groups for three folds: GroupKFold, seed-invariant, repeats collapse."""
    assert _split_is_seed_invariant({"groups": np.repeat(np.arange(5), 10), "cv_folds": 3})


def test_too_few_groups_fall_back_to_a_seed_dependent_kfold():
    """Two groups for three folds: the screen falls back to a shuffled KFold, so the repeats must run."""
    assert not _split_is_seed_invariant({"groups": np.repeat(np.arange(2), 10), "cv_folds": 3})


def test_time_aware_is_always_invariant_and_no_groups_is_not():
    """TimeSeriesSplit ignores the seed; with neither groups nor time order the shuffled split depends on it."""
    assert _split_is_seed_invariant({"time_aware": True})
    assert not _split_is_seed_invariant({})
