"""Cluster 42 regression: the votenrank benchmark family takes sigmoid from its own shared helper module (audit 2026-11-20, finding 42-F11)."""

import numpy as np

from mlframe.votenrank._benchmarks import _bench_shared
from mlframe.votenrank._benchmarks import bench_geometric_weight_blend as bench


def test_shared_sigmoid_matches_logistic_formula():
    """The shared helper computes 1 / (1 + exp(-z)) elementwise (42-F11)."""
    z = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(_bench_shared.sigmoid(z), 1.0 / (1.0 + np.exp(-z)))


def test_bench_uses_the_shared_sigmoid():
    """The bench's sigmoid is the one defined in _bench_shared, not the package-root copy (42-F11)."""
    assert bench._sigmoid is _bench_shared.sigmoid
