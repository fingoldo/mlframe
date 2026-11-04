"""The group-disjoint holdout carve treats missing group ids as one group instead of never holding them out."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._yscale_holdout_gate import _carve_group_disjoint


def test_nan_group_rows_are_held_out_together_or_not_at_all():
    """Every NaN-group row lands on the same side; before, they were always fit rows (``nan in set`` is False) and the holdout had no missing-id group."""
    groups = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, np.nan, np.nan, np.nan, 4.0, 4.0])
    sample_idx = np.arange(groups.size)
    seen_eval_nan = False
    for seed in range(40):
        fit_idx, eval_idx = _carve_group_disjoint(sample_idx, groups, 0.4, np.random.default_rng(seed))
        nan_rows = set(np.flatnonzero(np.isnan(groups)).tolist())
        in_eval = nan_rows & set(eval_idx.tolist())
        assert in_eval in (set(), nan_rows)
        seen_eval_nan |= bool(in_eval)
        # no group straddles the split
        for g in {1.0, 2.0, 3.0, 4.0}:
            rows = set(np.flatnonzero(groups == g).tolist())
            assert rows <= set(eval_idx.tolist()) or rows <= set(fit_idx.tolist())
    assert seen_eval_nan, "the missing-id group never reached the holdout in 40 draws"
