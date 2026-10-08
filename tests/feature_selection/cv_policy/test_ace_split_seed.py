"""The ACE held-out split draws its whole-group fold from the caller's seed, not a hardcoded constant."""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.cv_policy as cv_policy_mod
from mlframe.feature_selection.ace import _pfi_split


def test_pfi_split_forwards_split_seed_to_holdout_indices(monkeypatch):
    """Two different split seeds reach holdout_indices as two different random_state values."""
    seen: list[int] = []

    def _spy(policy, n, fraction, random_state=0, **_kw):
        """Record the seed and decline, so the caller falls through to the random split."""
        seen.append(random_state)
        return None

    monkeypatch.setattr(cv_policy_mod, "holdout_indices", _spy)
    y = (np.arange(200) % 2).astype(np.int64)
    _pfi_split(200, y, np.random.default_rng(0), None, 7)
    _pfi_split(200, y, np.random.default_rng(0), None, 11)
    assert seen == [7, 11]
