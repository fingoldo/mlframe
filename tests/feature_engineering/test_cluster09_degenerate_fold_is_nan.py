"""Cluster 09 regression: a degenerate LDA fold returns NaN, not the origin (audit 2026-11-20, finding 09-F23).

Zeros are valid projection coordinates, so a degenerate fold was indistinguishable from real rows at the origin. The degenerate fold now returns NaN.
"""

import numpy as np

from mlframe.feature_engineering.transformer import lda_projection as lda


def test_degenerate_fold_returns_nan():
    """With fewer than two positive rows the LDA features are NaN for every query row (09-F23)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = np.zeros(20, dtype=int)
    y[0] = 1  # one positive row: the positive slice is too small to fit
    Xq = rng.normal(size=(6, 3))
    out = lda.compute_lda_projection_features(X, y, Xq, None, seed=0, task="binary")
    assert np.isnan(out.to_numpy()).all()
