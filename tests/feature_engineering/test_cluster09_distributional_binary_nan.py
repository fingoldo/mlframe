"""Cluster 09 regression: distributional moments are NaN for binary targets (audit 2026-11-20, finding 09-F21).

Binary "quantiles" were predictions of classifiers reweighted differently, sorted against each other, which are not quantiles of one distribution.
The binary block is NaN now and the moment features propagate it.
"""

import numpy as np

from mlframe.feature_engineering.transformer import distributional_moments as dm


def test_binary_distributional_features_are_nan():
    """With a binary target every distributional output is NaN (09-F21)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + rng.normal(size=80) * 0.3 > 0).astype(int)
    Xq = rng.normal(size=(10, 3))
    out = dm.compute_distributional_moments_features(X, y, Xq, None, seed=0, task="binary")
    assert np.isnan(out.to_numpy()).all()
