"""Audit 2026-11-20 finding 25-F49: a learning-to-rank round reports no regression metrics.

compute_all_metrics used to send unknown and ranking target types to the MAE/MSE/RMSE/R2 block on a 1-D score, so a ranking
round logged regression numbers under ranking keys. Ranking types now return an empty dict, and regression is unchanged.
"""

import numpy as np

from mlframe.metrics.iteration_metrics import compute_all_metrics


def test_learning_to_rank_round_returns_no_regression_metrics():
    """A 1-D score under learning_to_rank yields {} rather than regression keys."""
    y = np.array([0.0, 1.0, 2.0, 0.0])
    s = np.array([0.1, 0.9, 0.5, 0.2])
    assert compute_all_metrics(y, s, "learning_to_rank") == {}


def test_regression_round_still_reports_regression_metrics():
    """A regression round keeps its block (the change does not touch regression)."""
    y = np.array([0.0, 1.0, 2.0, 3.0])
    s = np.array([0.1, 0.9, 2.2, 2.8])
    out = compute_all_metrics(y, s, "regression")
    assert out, "regression should report metrics"
