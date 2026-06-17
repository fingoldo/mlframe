"""Unit tests for the E2 embargo trim (`_apply_purge_embargo`) + cv_purge config field."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._preprocessing_configs import TrainingSplitConfig
from mlframe.training.core._phase_helpers_fit_split import _apply_purge_embargo


def test_cv_purge_field_default_and_bounds():
    assert TrainingSplitConfig().cv_purge == 0
    assert TrainingSplitConfig(cv_strategy="purged", cv_purge=10).cv_purge == 10
    with pytest.raises(ValueError):
        TrainingSplitConfig(cv_purge=-1)


def test_embargo_drops_newest_train_rows():
    train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ts = np.arange(10)  # idx == time
    out = _apply_purge_embargo(train_idx, ts, purge=3)
    # The 3 newest (times 7,8,9) are dropped; the rest preserved in order.
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3, 4, 5, 6]))


def test_embargo_respects_time_order_not_index_order():
    # train_idx not in time order; embargo must drop by TIME, not by index position.
    train_idx = np.array([5, 0, 9, 3])
    ts = np.array([50, 0, 90, 30])  # aligned to full row space; _apply_purge_embargo indexes ts[train_idx]
    full_ts = np.zeros(10)
    full_ts[[5, 0, 9, 3]] = [50, 0, 90, 30]
    out = _apply_purge_embargo(train_idx, full_ts, purge=1)
    assert 9 not in out  # idx 9 has the largest time (90) -> dropped
    assert set(out.tolist()) == {5, 0, 3}


def test_embargo_noops_on_zero_or_too_large():
    train_idx = np.arange(5)
    ts = np.arange(5)
    np.testing.assert_array_equal(_apply_purge_embargo(train_idx, ts, 0), train_idx)
    np.testing.assert_array_equal(_apply_purge_embargo(train_idx, ts, 10), train_idx)  # would empty -> no-op
    np.testing.assert_array_equal(_apply_purge_embargo(train_idx, None, 3), train_idx)
