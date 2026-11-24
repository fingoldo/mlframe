"""Cluster 09 regression: the sign-residual baseline emits no label-copy signal for binary targets (audit 2026-11-20, finding 09-F19).

For y in {0, 1}, ``y > mu_hat`` is the label itself, so the sign model only re-learned y. For binary tasks the bias columns are now NaN; the
out-of-fold baseline column is kept.
"""

import numpy as np

from mlframe.feature_engineering.transformer import sign_residual_baseline as srb


def test_binary_bias_columns_are_nan():
    """With a binary target the bias columns are NaN and the baseline column is finite (09-F19)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + rng.normal(size=80) * 0.3 > 0).astype(int)
    Xq = rng.normal(size=(10, 3))
    out = srb.compute_sign_residual_baseline_features(X, y, Xq, None, seed=0, task="binary").to_numpy()
    assert np.isnan(out[:, 1:4]).all()
    assert np.isfinite(out[:, 0]).all()
