"""The clustered bootstrap reports how many resamples it dropped, and warns when that is more than a quarter.

Dropped resamples used to vanish, so an interval built on a biased subset carried no sign of it.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.evaluation import _bootstrap_clusters as bc


def test_dropped_resamples_are_counted_and_a_large_share_is_warned(caplog):
    """A metric that fails on half of the resamples reports ``n_dropped`` and logs a WARNING."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.5, size=200)
    calls = {"n": 0}

    def flaky(yt, yp):
        """Succeed on the point estimate, then fail on every other resample."""
        calls["n"] += 1
        if calls["n"] > 1 and calls["n"] % 2:
            raise ValueError("undefined on this resample")
        return float(np.mean((yt - yp) ** 2))

    with caplog.at_level(logging.WARNING, logger=bc.logger.name):
        out = bc.bootstrap_metrics_clustered(y_true, y_pred, {"mse": flaky}, n_bootstrap=100, random_state=0, block_length=10)
    assert out["mse"]["n_dropped"] > 25
    assert any(r.levelno == logging.WARNING and "dropped" in r.getMessage() for r in caplog.records)
