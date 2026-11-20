"""A weighted quantile above the last midpoint returns the group's largest value, never uninitialised memory (audit 2026-11-20, finding 25-F1)."""

import numpy as np

from mlframe.inference.entity_prediction_collapse import collapse_predictions_by_group


def test_weighted_quantile_one_returns_group_maximum():
    """With quantile 1.0 each group collapses to its maximum value."""
    pred = np.array([0.1, 0.9, 0.4, 0.2, 0.7])
    group = np.array(["a", "a", "a", "b", "b"])
    weights = np.ones(5)
    out = collapse_predictions_by_group(pred, group, stat="quantile", quantile=1.0, weights=weights)
    np.testing.assert_allclose(out, [0.9, 0.9, 0.9, 0.7, 0.7])
