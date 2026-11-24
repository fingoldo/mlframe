"""Cluster 09 regression: the ordinal monotonicity count is NaN for binary targets (audit 2026-11-20, finding 09-F20).

Binary thresholds are non-nested sub-population indicators, so counting adjacent increases means nothing. The count is NaN for binary tasks and
unchanged otherwise.
"""

import numpy as np

from mlframe.feature_engineering.transformer import multi_threshold_ordinal as mto


def test_binary_violation_count_is_nan():
    """For a binary target the n_violations column is NaN (09-F20)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + rng.normal(size=80) * 0.3 > 0).astype(int)
    Xq = rng.normal(size=(10, 3))
    out = mto.compute_multi_threshold_ordinal_features(X, y, Xq, None, seed=0, task="binary")
    assert np.isnan(out["multthr_n_violations"].to_numpy()).all()
