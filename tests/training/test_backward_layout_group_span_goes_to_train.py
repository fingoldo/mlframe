"""With val older than train, a group spanning both stays in train (the newer side) instead of being pulled into val."""

from __future__ import annotations

import numpy as np

from mlframe.training.splitting import _make_train_test_sp_also_supplied


def test_group_spanning_older_val_and_newer_train_ends_in_train():
    """Rows 0-3 are the oldest (val), 4-9 train, 10-11 test; group 'g' has rows in val and train: all its rows end up in train, none in val."""
    ts = np.arange(12)
    groups = np.array(["a", "g", "b", "g", "g", "c", "d", "e", "f", "c", "t", "t"])
    test_idx, train_idx, val_idx = _make_train_test_sp_also_supplied(groups, ts, np.arange(4, 10), np.arange(0, 4), np.arange(10, 12), np.zeros(12))
    assert set(groups[val_idx]) == {"a", "b"}
    assert {1, 3, 4} <= set(train_idx.tolist())
    assert train_idx.max() < 10 and set(test_idx.tolist()) == {10, 11}
