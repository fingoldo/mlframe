"""Cluster 39 regression: the all-zero-weight check is sampled, not run on every step (audit 2026-11-20, finding 39-F68).

The check costs a host-device sync. It now runs on every ``_ZERO_WEIGHT_CHECK_EVERY``-th batch, so a zero-weight batch is reported at the
next sampled batch. The helper decides which batches are checked.
"""

from mlframe.training.neural._flat_torch_module import _flat_torch_loss as fl


def test_check_runs_on_sampled_batches_only():
    """Batch 0 and every multiple of the period are checked; the others are skipped (39-F68)."""
    period = fl._ZERO_WEIGHT_CHECK_EVERY
    assert fl._zero_weight_check_due(0)
    assert fl._zero_weight_check_due(period)
    assert not fl._zero_weight_check_due(1)
    assert not fl._zero_weight_check_due(period + 1)


def test_period_is_more_than_one():
    """The sampling period is greater than one, so it actually removes per-step syncs (39-F68)."""
    assert fl._ZERO_WEIGHT_CHECK_EVERY > 1
