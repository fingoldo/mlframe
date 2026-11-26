"""Cluster 39 regression: PeriodicLearningRateFinder re-runs the sweep on epochs that are multiples of ``period`` (audit 2026-11-20, finding 39-F60).

The condition carried a redundant ``or current_epoch == 0`` clause, since 0 is a multiple of any positive period. The behaviour is kept: epoch 0,
period and 2*period run the sweep, and the others do not.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("lightning")  # the neural package subclasses Lightning classes: shards without the neural extras skip this module

from mlframe.training.neural._base_callbacks import PeriodicLearningRateFinder


def _ran_at(period, epoch):
    """Whether the sweep runs at ``epoch`` for a finder with ``period``."""
    cb = PeriodicLearningRateFinder(period)
    cb.lr_find = MagicMock()
    trainer = MagicMock(current_epoch=epoch)
    pl_module = MagicMock()
    cb.on_train_epoch_start(trainer, pl_module)
    return cb.lr_find.called


def test_sweep_runs_on_epoch_zero_and_multiples():
    """Epoch 0 and every multiple of the period run the sweep (39-F60)."""
    assert _ran_at(3, 0)
    assert _ran_at(3, 3)
    assert _ran_at(3, 6)


def test_sweep_skips_other_epochs():
    """Epochs that are not multiples of the period do not run the sweep (39-F60)."""
    assert not _ran_at(3, 1)
    assert not _ran_at(3, 4)
