"""``recency_weights`` is the dense reference for the fused per-group recency kernels in ``feature_engineering.recency_aggregation``.

The production path computes the same weight families inline inside its per-group njit loops, so the dense vector form has no
caller of its own. These tests pin that the two agree, which is what keeps the reference honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.recency_weights import recency_weights
from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_mean

_CASES = [("poly", 0.0), ("poly", 1.5), ("exp", 1.0), ("exp", 0.8), ("power", 0.0), ("power", 1.2)]


@pytest.mark.parametrize("scheme,param", _CASES)
@pytest.mark.parametrize("d", [1, 7, 60])
def test_fused_per_group_mean_matches_dense_recency_weights(scheme, param, d):
    """A single-group weighted mean equals the dot product with the dense weight vector, oldest first."""
    rng = np.random.default_rng(d)
    v = rng.normal(size=d)
    dense = float(np.dot(recency_weights(d, scheme, param), v))
    fused = float(per_group_recency_weighted_mean(v, np.zeros(d, dtype=np.int64), scheme=scheme, param=param, broadcast=False)[0])
    assert fused == pytest.approx(dense, rel=1e-12, abs=1e-12)


def test_identity_parameter_gives_uniform_weights():
    """At the identity parameter the dense vector is uniform, so the fused mean is the plain mean."""
    w = recency_weights(9, "poly", 0.0)
    np.testing.assert_allclose(w, np.full(9, 1 / 9))
