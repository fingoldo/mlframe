"""The FE accuracy gate's probe folds honour group ids, so a group-memorising column earns no uplift."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_accuracy_gate import measure_feature_uplift


def _panel(seed: int = 0):
    """Regression panel with a pure group effect on y, a noise base column, and a one-hot group-id block as the 'engineered' set."""
    rng = np.random.default_rng(seed)
    n_groups, per = 40, 25
    gid = np.repeat(np.arange(n_groups), per)
    y = rng.normal(0, 1.0, n_groups)[gid] + rng.normal(0, 0.3, gid.size)
    base = rng.normal(size=(gid.size, 1))
    onehot = np.eye(n_groups)[gid]
    return base, onehot, y, gid


def test_group_memorising_columns_show_uplift_only_under_iid_folds():
    """One-hot group ids look like real uplift with i.i.d. folds and none (or negative) with group-disjoint folds."""
    base, onehot, y, gid = _panel()
    iid = measure_feature_uplift(base, onehot, y, classification=False, n_splits=5, seed=0)
    grouped = measure_feature_uplift(base, onehot, y, classification=False, n_splits=5, seed=0, groups=gid)
    assert iid is not None and grouped is not None
    assert iid > 0.3
    assert grouped < 0.02
