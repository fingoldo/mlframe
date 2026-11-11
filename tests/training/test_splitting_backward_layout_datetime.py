"""Regression: the backward-layout group guard accepts datetime64 timestamps.

``_move_groups_spanning_older_val_into_train`` compared the val and train medians with ``np.median``, which averages the two middle
values and raises ``UFuncTypeError`` on datetime64 (the dtype pandas hands over for a time column). Any group-aware split with a
datetime time column and a backward layout crashed. The guard now compares order statistics, which work for every sortable dtype.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.splitting import _move_groups_spanning_older_val_into_train


def test_backward_layout_guard_accepts_datetime64_timestamps():
    """A group split across a backward val (older) / train (newer) layout keeps its rows on the newer side, with datetime64 time."""
    base = np.datetime64("2024-01-01")
    # val (idx 0..3) is older than train (idx 4..7); group "g1" appears in both.
    timestamps = np.array(
        [base + np.timedelta64(d, "D") for d in (0, 1, 2, 3, 10, 11, 12, 13)],
        dtype="datetime64[ns]",
    )
    groups = np.array(["g0", "g1", "g1", "g2", "g1", "g3", "g3", "g4"])
    train_idx = np.arange(4, 8)
    val_idx = np.arange(0, 4)

    new_train, new_val = _move_groups_spanning_older_val_into_train(groups, timestamps, train_idx, val_idx)

    # The g1 rows of val (idx 1 and 2) move into train; val keeps only the non-spanning groups.
    assert set(new_train.tolist()) == {1, 2, 4, 5, 6, 7}
    assert set(new_val.tolist()) == {0, 3}


def test_forward_layout_is_left_untouched_with_datetime64_timestamps():
    """A forward layout (val newer than train) is returned unchanged, and the datetime comparison does not raise."""
    base = np.datetime64("2024-01-01")
    timestamps = np.array(
        [base + np.timedelta64(d, "D") for d in (10, 11, 12, 13, 0, 1, 2, 3)],
        dtype="datetime64[ns]",
    )
    groups = np.array(["g1", "g2", "g3", "g4", "g1", "g2", "g3", "g4"])
    train_idx = np.arange(4, 8)
    val_idx = np.arange(0, 4)

    new_train, new_val = _move_groups_spanning_older_val_into_train(groups, timestamps, train_idx, val_idx)

    assert np.array_equal(new_train, train_idx)
    assert np.array_equal(new_val, val_idx)
