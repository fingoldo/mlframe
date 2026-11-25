"""Cluster 17 regression: the adjusted-R2 bench calls the shipped functions instead of its own formulas (audit 2026-11-20, finding 17-F28).

Pinned: the bench helpers delegate to fast_r2_score and fast_adjusted_r2_score, and on a sample they agree with those functions.
"""

import numpy as np

from mlframe.metrics._benchmarks import bench_adjusted_r2 as bench
from mlframe.metrics.regression._regression_extras import fast_adjusted_r2_score
from mlframe.metrics.regression._regression_metrics import fast_r2_score


def test_bench_helpers_match_the_shipped_functions():
    """The bench's R2 and adjusted R2 equal the shipped functions on the same data (17-F28)."""
    rng = np.random.default_rng(0)
    yt = rng.normal(size=300)
    yp = yt * 0.6 + rng.normal(size=300)
    assert np.isclose(bench._r2(yt, yp), fast_r2_score(yt, yp))
    assert np.isclose(bench._adj_r2(yt, yp, 3), fast_adjusted_r2_score(yt, yp, 3))
