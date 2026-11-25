"""Cluster 32 regression: the CatBoost iteration callback captures the final round after fit (audit 2026-11-20, finding 32-F9).

With stride > 1 the final round may not land on a stride boundary. ``flush()`` captures the last round seen when the stride skipped it, and
never recaptures a round already stored.
"""

import numpy as np

from mlframe.training.callbacks.iteration_metrics import CBIterationMetricsCallback


class _Info:
    """Minimal CatBoost ``info`` object: the 1-based round and the model."""

    def __init__(self, iteration, model):
        self.iteration = iteration
        self.model = model


class _Model:
    """Regression stub whose predict returns a round-dependent vector."""

    def predict(self, pool, ntree_end):
        """Return a vector that depends on the round so each capture is distinct."""
        return np.full(4, float(ntree_end))


def _run(rounds: int, stride: int) -> CBIterationMetricsCallback:
    """Feed ``rounds`` CatBoost iterations to a regression callback with the given stride."""
    cb = CBIterationMetricsCallback(val_pool=None, y_val=np.array([1.0, 2.0, 3.0, 4.0]), target_type="regression", stride=stride)
    model = _Model()
    for it in range(1, rounds + 1):
        cb.after_iteration(_Info(it, model))
    return cb


def test_final_round_is_missing_without_flush():
    """With stride 5 over 12 rounds the last round (index 11) is not captured before flush (32-F9)."""
    cb = _run(rounds=12, stride=5)
    assert 11 not in cb.iteration_metrics_


def test_flush_captures_the_final_round():
    """After flush the final round is present (32-F9)."""
    cb = _run(rounds=12, stride=5)
    cb.flush()
    assert 11 in cb.iteration_metrics_


def test_flush_does_not_recapture_a_stored_round():
    """A round already on a stride boundary is not stored twice; flush is idempotent (32-F9)."""
    cb = _run(rounds=11, stride=5)
    assert 10 in cb.iteration_metrics_
    before = dict(cb.iteration_metrics_)
    cb.flush()
    cb.flush()
    assert cb.iteration_metrics_.keys() == before.keys()
