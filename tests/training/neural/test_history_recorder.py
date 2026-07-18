"""Unit tests for ``TrainingHistoryRecorder``: it must accumulate per-epoch train/val history in the
booster ``evals_result_`` shape (so the training-curve chart auto-emits for neural) and track the best
epoch by the monitored metric's direction."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("lightning")

from mlframe.training.neural._history_recorder import TrainingHistoryRecorder


def _trainer(epoch, metrics, sanity=False):
    """Trainer."""
    return types.SimpleNamespace(current_epoch=epoch, callback_metrics=metrics, sanity_checking=sanity)


def test_records_train_and_val_in_booster_shape_and_aligned():
    """Records train and val in booster shape and aligned."""
    rec = TrainingHistoryRecorder(monitor="val_loss", mode="min")
    # val_loss: 0.5 -> 0.3 (best) -> 0.4 -> 0.45 ; train monotonically down.
    seq = [(0, 0.6, 0.5), (1, 0.4, 0.3), (2, 0.3, 0.4), (3, 0.25, 0.45)]
    for ep, tr, va in seq:
        rec.on_validation_epoch_end(_trainer(ep, {"train_loss": tr, "val_loss": va}), None)
    assert rec.evals_result_["train"]["loss"] == [0.6, 0.4, 0.3, 0.25]
    assert rec.evals_result_["val"]["loss"] == [0.5, 0.3, 0.4, 0.45]
    # length-aligned splits.
    assert len(rec.evals_result_["train"]["loss"]) == len(rec.evals_result_["val"]["loss"])
    # best epoch is the val-loss minimum (epoch 1).
    assert rec.best_iteration_ == 1


def test_skips_sanity_check_pass():
    """Skips sanity check pass."""
    rec = TrainingHistoryRecorder(monitor="val_loss", mode="min")
    rec.on_validation_epoch_end(_trainer(0, {"train_loss": 9.9, "val_loss": 9.9}, sanity=True), None)
    assert rec.evals_result_ == {}
    assert rec.best_iteration_ is None


def test_mode_max_tracks_metric_increasing():
    """Mode max tracks metric increasing."""
    rec = TrainingHistoryRecorder(monitor="val_roc_auc", mode="max")
    for ep, auc in [(0, 0.70), (1, 0.85), (2, 0.82)]:
        rec.on_validation_epoch_end(_trainer(ep, {"val_roc_auc": auc}), None)
    assert rec.evals_result_["val"]["roc_auc"] == [0.70, 0.85, 0.82]
    assert rec.best_iteration_ == 1  # the 0.85 peak


def test_non_float_metrics_are_skipped():
    """Non float metrics are skipped."""
    rec = TrainingHistoryRecorder(monitor="val_loss", mode="min")
    rec.on_validation_epoch_end(_trainer(0, {"val_loss": None, "train_loss": "x", "epoch": 0}), None)
    assert rec.evals_result_ == {}
