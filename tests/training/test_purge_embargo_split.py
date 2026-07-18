"""Unit tests for the E2 embargo trim (`_apply_purge_embargo`) + cv_purge config field."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._preprocessing_configs import TrainingSplitConfig
from mlframe.training.core._phase_helpers_fit_split import (
    _apply_purge_embargo,
    _apply_val_test_embargo,
)


def test_cv_purge_field_default_and_bounds():
    """Cv purge field default and bounds."""
    assert TrainingSplitConfig().cv_purge == 0
    assert TrainingSplitConfig(cv_strategy="purged", cv_purge=10).cv_purge == 10
    with pytest.raises(ValueError):
        TrainingSplitConfig(cv_purge=-1)


def test_embargo_drops_newest_train_rows():
    """Embargo drops newest train rows."""
    train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ts = np.arange(10)  # idx == time
    out = _apply_purge_embargo(train_idx, ts, purge=3)
    # The 3 newest (times 7,8,9) are dropped; the rest preserved in order.
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3, 4, 5, 6]))


def test_embargo_respects_time_order_not_index_order():
    # train_idx not in time order; embargo must drop by TIME, not by index position.
    """Embargo respects time order not index order."""
    train_idx = np.array([5, 0, 9, 3])
    full_ts = np.zeros(10)
    full_ts[[5, 0, 9, 3]] = [50, 0, 90, 30]
    out = _apply_purge_embargo(train_idx, full_ts, purge=1)
    assert 9 not in out  # idx 9 has the largest time (90) -> dropped
    assert set(out.tolist()) == {5, 0, 3}


def test_embargo_noops_on_zero_or_too_large():
    """Embargo noops on zero or too large."""
    train_idx = np.arange(5)
    ts = np.arange(5)
    np.testing.assert_array_equal(_apply_purge_embargo(train_idx, ts, 0), train_idx)
    np.testing.assert_array_equal(_apply_purge_embargo(train_idx, ts, 10), train_idx)  # would empty -> no-op
    np.testing.assert_array_equal(_apply_purge_embargo(train_idx, None, 3), train_idx)


def test_val_test_embargo_drops_newest_val_rows():
    """Val test embargo drops newest val rows."""
    val_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ts = np.arange(10)
    out = _apply_val_test_embargo(val_idx, ts, purge=3)
    # The 3 newest val rows (times 7,8,9) are dropped so val<->test gets a gap.
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3, 4, 5, 6]))


def test_val_test_embargo_respects_time_order_not_index_order():
    """Val test embargo respects time order not index order."""
    val_idx = np.array([5, 0, 9, 3])
    full_ts = np.zeros(10)
    full_ts[[5, 0, 9, 3]] = [50, 0, 90, 30]
    out = _apply_val_test_embargo(val_idx, full_ts, purge=1)
    assert 9 not in out  # largest time -> dropped
    assert set(out.tolist()) == {5, 0, 3}


def test_val_test_embargo_noops_on_zero_or_too_large():
    """Val test embargo noops on zero or too large."""
    val_idx = np.arange(5)
    ts = np.arange(5)
    np.testing.assert_array_equal(_apply_val_test_embargo(val_idx, ts, 0), val_idx)
    np.testing.assert_array_equal(_apply_val_test_embargo(val_idx, ts, 10), val_idx)
    np.testing.assert_array_equal(_apply_val_test_embargo(val_idx, None, 3), val_idx)


def test_val_test_embargo_creates_gap_between_val_and_test():
    """With purge>0 there is a strict time gap between max(val_ts) and min(test_ts)
    in a forward-walk [train][val][test] layout: trimming the newest val rows.
    """
    ts = np.arange(30)
    # forward-walk contiguous blocks: train [0:20], val [20:25], test [25:30]
    val_idx = np.arange(20, 25)
    test_idx = np.arange(25, 30)
    purge = 2
    trimmed_val = _apply_val_test_embargo(val_idx, ts, purge)
    assert ts[trimmed_val].max() < ts[test_idx].min()
    # gap width equals the embargo (last kept val time 22, first test time 25 -> 2-row gap).
    assert ts[test_idx].min() - ts[trimmed_val].max() > purge - 1
