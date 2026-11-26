"""Cluster 17 regression: an optimiser failure in the ordinal cutpoint search is visible (audit 2026-11-20, finding 17-F14).

A broad ``except Exception`` turned any failure of the Powell search into a DEBUG line and a fallback result. The fallback still runs, but the failure is
now logged at WARNING with its exception type; a missing scipy stays a quiet INFO fallback.
"""

import logging

import numpy as np

from mlframe.metrics.classification._ordinal_cutpoints import optimal_ordinal_cutpoints


def _data():
    """Ordinal labels with noisy predictions around them."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=300)
    pred = y + rng.normal(scale=0.4, size=300)
    return y, pred


def test_optimiser_failure_is_a_warning_and_still_returns_cutpoints(monkeypatch, caplog):
    """A RuntimeError from scipy's minimize is logged at WARNING with its type, and the coordinate scan still returns thresholds (17-F14)."""
    import scipy.optimize

    def boom(*args, **kwargs):
        """Fail like an optimiser bug."""
        raise RuntimeError("powell exploded")

    monkeypatch.setattr(scipy.optimize, "minimize", boom)
    y, pred = _data()
    with caplog.at_level(logging.INFO, logger="mlframe.metrics.classification._ordinal_cutpoints"):
        thr, score = optimal_ordinal_cutpoints(y, pred, n_classes=4)
    assert thr.shape == (3,) and np.isfinite(score)
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and "RuntimeError" in r.getMessage()]
    assert warned, "the optimiser failure was not logged at WARNING"
