"""Cluster 22 regression: resamples whose metric raises are reported at WARNING, not silently dropped (audit 2026-11-20, finding 22-F41)."""

import logging

import numpy as np

from mlframe.calibration import policy as pol


def test_failed_resamples_are_counted_in_one_warning(caplog):
    """A metric that raises on some resamples produces a single WARNING with the failure count (22-F41)."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.linspace(0.05, 0.95, 8)
    idx_matrix = np.tile(np.arange(8), (12, 1))
    calls = {"n": 0}

    def _flaky(yt, yp):
        """Raise on every third call (including the point estimate's neighbours), otherwise return a plain statistic."""
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise ValueError("boom")
        return float(np.mean(yp))

    with caplog.at_level(logging.WARNING, logger=pol.logger.name):
        pol._bootstrap_ece_with_indices(y_true, y_pred, idx_matrix, _flaky, alpha=0.1)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "resamples failed" in r.getMessage()]
    assert len(warnings) == 1
