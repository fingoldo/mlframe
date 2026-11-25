"""Cluster 11 regression: als_rank1 has a smoke test on a small signal (audit 2026-11-20, finding 11-F20).

The rank-1 ALS fit of y on binned (x, z) is one of the stat-study kernels that had no test. The result has one value per row, is finite, and
tracks the signal it was fitted to.
"""

import numpy as np

from mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.core import als_rank1


def test_als_rank1_returns_finite_per_row_values_tracking_the_signal():
    """The fit returns one finite value per row and correlates with the target it was fitted to (11-F20)."""
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.sin(x) * np.cos(z) + 0.05 * rng.normal(size=n)
    out = als_rank1(x, z, y, np.arange(n))
    assert out.shape == (n,)
    assert np.isfinite(out).all()
    assert np.corrcoef(out, y)[0, 1] > 0.5
