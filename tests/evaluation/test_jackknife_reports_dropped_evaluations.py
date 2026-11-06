"""The BCa jackknife reports the leave-one-out evaluations it dropped.

Dropped values were skipped at debug level, so the interval was built from fewer evaluations without any sign of it.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.evaluation import _bootstrap_jackknife as jk


def test_dropped_leave_one_out_evaluations_are_reported(caplog):
    """A metric that is undefined on some leave-one-out subsets produces a WARNING with the count dropped."""
    y_true = np.arange(20, dtype=np.float64)
    y_pred = y_true.copy()

    def flaky(yt, yp):
        """Undefined whenever the held-out row was the first one."""
        if yt[0] != 0.0:
            return float("nan")
        return float(np.mean(yt - yp) + 1.0)

    with caplog.at_level(logging.WARNING, logger=jk.logger.name):
        out = jk._jackknife_metric(y_true, y_pred, flaky)
    assert out is not None and out.size > 0
    assert any(r.levelno == logging.WARNING and "dropped" in r.getMessage() for r in caplog.records)
