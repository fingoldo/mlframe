"""Cluster 09 regression: local_linear processes its anchors in blocks without changing the result (audit 2026-11-20, finding 09-F29)."""

import numpy as np

import mlframe.feature_engineering.transformer.local_linear as ll


def test_tiny_block_budget_gives_the_same_features_as_one_block():
    """Forcing one anchor per block yields exactly the output of a single block over all anchors."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=150)).astype(np.float32)
    Xq = rng.normal(size=(40, 4)).astype(np.float32)
    one_block = ll.compute_local_linear_attention(X, y, Xq, None, seed=1, k=12).to_numpy()
    original = ll._LOCAL_RIDGE_BLOCK_BYTES
    try:
        ll._LOCAL_RIDGE_BLOCK_BYTES = 1  # block = 1 anchor
        many_blocks = ll.compute_local_linear_attention(X, y, Xq, None, seed=1, k=12).to_numpy()
    finally:
        ll._LOCAL_RIDGE_BLOCK_BYTES = original
    np.testing.assert_array_equal(one_block, many_blocks)
