"""compute_extra_aggregates guards a zero softmax temperature like the fused kernel does (tracker 2026-11-02 wave 12, NJ12-8)."""

import numpy as np

from mlframe.feature_engineering.transformer._aggregation import compute_extra_aggregates


def test_zero_temperature_equals_temperature_one():
    """softmax_temp=0 divided the logits by zero (inf/NaN weights, uniform fallback); the kernel treats it as 1.0, so must this."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(30, 4)).astype(np.float32)
    k = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.normal(size=100).astype(np.float32)
    ids = rng.integers(0, 100, (30, 5))
    zero = compute_extra_aggregates(q, k, y, ids, 0.0, ("y_skew", "y_iqr"))
    one = compute_extra_aggregates(q, k, y, ids, 1.0, ("y_skew", "y_iqr"))
    for name in one:
        np.testing.assert_allclose(zero[name], one[name], rtol=1e-6, atol=1e-6)
