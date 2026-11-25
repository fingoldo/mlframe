"""Cluster 28 regression: a failed train-envelope computation is logged at WARNING, not DEBUG (audit 2026-11-20, finding 28-F56).

The reporter falls back to the per-split envelope when the train envelope cannot be computed, which changes the report, so the failure is
visible at the default level.
"""

import logging

import numpy as np

from mlframe.training import _prediction_envelope_clip as clip
from mlframe.training import _trainer_train_and_evaluate_parts as parts


def test_envelope_failure_is_logged_at_warning(monkeypatch, caplog):
    """A raising envelope computation returns None and logs a WARNING naming the fallback (28-F56)."""

    def _boom(_target):
        """Stand in for a failing envelope computation."""
        raise ValueError("bad target")

    monkeypatch.setattr(clip, "compute_train_envelope_stats", _boom)
    with caplog.at_level(logging.WARNING):
        out = parts._train_envelope_stats(np.arange(10.0), 1)
    assert out is None
    assert any(r.levelno == logging.WARNING and "falls back" in r.getMessage() for r in caplog.records)
