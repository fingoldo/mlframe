"""Cluster 39 regression: compute_learning_curve does not build a probe estimator when warm_start is off (audit 2026-11-20, finding 39-F13).

The audit said the warm-start probe runs for every call. The probe is only reached when ``warm_start`` is True: the conditional expression
short-circuits. This test pins that by making the probe raise; a call with ``warm_start=False`` must still succeed.
"""

import numpy as np

from mlframe.training.diagnostics import learning_curve as lc


class _Est:
    """Minimal estimator with fit and predict."""

    def fit(self, X, y):
        """Fit nothing; return self."""
        return self

    def predict(self, X):
        """Predict zeros."""
        return np.zeros(len(X))


def test_warm_start_off_never_reaches_the_probe(monkeypatch):
    """With warm_start=False the warm-start probe is not called (39-F13)."""

    def _probe_must_not_run(est):
        """Fail loudly if the probe is reached without warm start."""
        raise AssertionError("warm-start probe ran with warm_start=False")

    monkeypatch.setattr(lc, "_supports_warm_start", _probe_must_not_run)
    X = np.arange(200, dtype=float).reshape(-1, 1)
    y = np.arange(200) % 2
    result = lc.compute_learning_curve(
        _Est,
        X,
        y,
        scorer=lambda est, X, y: 0.5,
        warm_start=False,
        n_jobs=1,
    )
    assert result.warm_start_used is False
    assert len(result.train_sizes) > 0
