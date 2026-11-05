"""A failed bootstrap block is recorded as skipped and the reason is logged at WARNING, not only at DEBUG.

A skipped block removes a headline confidence interval from the honest report. At debug level the run looked complete, so the missing interval
had no explanation in the log.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.training import honest_diagnostics as hd


def test_failed_rmse_block_is_recorded_as_skipped_and_logged_as_a_warning(monkeypatch, caplog):
    """When the bootstrap raises, the entry is marked skipped with the reason, and a WARNING names the failure."""

    def _boom(*args, **kwargs):
        """Stand-in for a bootstrap that cannot run."""
        raise RuntimeError("bootstrap exploded")

    import mlframe.evaluation.bootstrap as _bs

    monkeypatch.setattr(_bs, "bootstrap_metric", _boom)
    out: dict = {}
    y = np.array([0.0, 1.0, 1.0, 0.0])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    with caplog.at_level(logging.WARNING, logger=hd.logger.name):
        hd._rmse_bootstrap_entry(y, p, out, 0)
    assert out["rmse"]["status"] == "skipped"
    assert "bootstrap exploded" in out["rmse"]["reason"]
    assert any(r.levelno == logging.WARNING and "rmse metric computation failed" in r.getMessage() for r in caplog.records)
