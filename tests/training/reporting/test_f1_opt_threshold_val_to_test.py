"""The F1-optimal title block: tuned on val, reused unchanged on test, never optimised on test labels."""

from __future__ import annotations

import numpy as np

from mlframe.training.reporting._reporting_probabilistic import _resolve_f1_opt_threshold


def _scores(seed: int, n: int = 4000):
    """A noisy but informative binary scorer."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, np.clip(0.35 * y + rng.normal(0.35, 0.2, n), 0.0, 1.0)


def test_val_tunes_and_records_the_threshold_for_the_test_split():
    """The val call computes the threshold and stores it where the test call will read it."""
    y, s = _scores(0)
    metrics: dict = {}
    thr = _resolve_f1_opt_threshold(True, y, s, None, True, metrics)
    assert thr is not None and 0.0 < thr < 1.0
    assert metrics["f1_opt_threshold"] == thr


def test_test_split_uses_the_given_val_threshold_not_its_own_optimum():
    """A threshold handed in wins even when the split's own F1 optimum is elsewhere, and nothing is recorded."""
    y, s = _scores(1)
    metrics: dict = {}
    assert _resolve_f1_opt_threshold(True, y, s, 0.123, False, metrics) == 0.123
    assert "f1_opt_threshold" not in metrics


def test_train_and_non_binary_splits_get_no_tuned_block():
    """Neither tuning requested nor a threshold given means no block; a non-positive-class report never tunes."""
    y, s = _scores(2)
    assert _resolve_f1_opt_threshold(True, y, s, None, False, {}) is None
    assert _resolve_f1_opt_threshold(False, y, s, None, True, {}) is None
