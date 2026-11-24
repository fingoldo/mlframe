"""Cluster 39 regression: an epoch with no validation batches does not crash the aggregating callback (audit 2026-11-20, finding 39-F59).

``on_validation_epoch_end`` used to call ``torch.concat`` on an empty list and raise. It now logs a WARNING and skips the metric. A populated epoch
still computes and logs the metric once.
"""

import logging
from unittest.mock import MagicMock

import torch

from mlframe.training.neural._base_callbacks import AggregatingValidationCallback


def _callback():
    """A callback computing a simple mean of the scores, for the tests."""
    return AggregatingValidationCallback(metric_name="mean_score", metric_fcn=lambda y_true, y_score: float(y_score.mean()))


def test_empty_epoch_warns_and_does_not_raise(caplog):
    """No buffered batches: a WARNING is logged and nothing is logged to the module (39-F59)."""
    cb = _callback()
    pl_module = MagicMock()
    with caplog.at_level(logging.WARNING):
        cb.on_validation_epoch_end(trainer=None, pl_module=pl_module)
    pl_module.log.assert_not_called()
    assert any("no validation batches" in r.getMessage() for r in caplog.records)


def test_populated_epoch_logs_the_metric_once():
    """Control: buffered batches are concatenated, scored and logged, then the buffers reset (39-F59)."""
    cb = _callback()
    cb.on_validation_batch_end(None, None, (torch.tensor([0.2, 0.4]), torch.tensor([1, 0])), None, 0)
    cb.on_validation_batch_end(None, None, (torch.tensor([0.6]), torch.tensor([1])), None, 1)
    pl_module = MagicMock()
    cb.on_validation_epoch_end(trainer=None, pl_module=pl_module)
    pl_module.log.assert_called_once()
    assert pl_module.log.call_args.kwargs["name"] == "val_mean_score"
    assert abs(pl_module.log.call_args.kwargs["value"] - 0.4) < 1e-6
    assert cb.batched_labels == []
