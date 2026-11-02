"""A metric that is NaN at an extreme threshold must not be chosen as the best (tracker 2026-11-02 wave 8, NAN8-2)."""

import numpy as np

from mlframe.calibration.threshold_optimizer import optimize_decision_threshold


def _metric_nan_without_positives(y_true, y_pred):
    """Accuracy, undefined (NaN) when nothing is predicted positive."""
    if y_pred.sum() == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def test_nan_score_at_threshold_one_is_not_the_best():
    """Predicting no positives at t=1.0 gives NaN; argmax used to return that index."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 400), 0.01, 0.99)
    res = optimize_decision_threshold(y, p, _metric_nan_without_positives, n_thresholds=51)
    assert np.isfinite(res["best_score"])
    assert res["best_threshold"] < 1.0
