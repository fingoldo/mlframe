"""DeLong's two-sided p-value keeps its magnitude for large |z| instead of cancelling to exactly 0."""

from __future__ import annotations

import numpy as np

from mlframe.evaluation.bootstrap import delong_test


def test_p_value_at_z_17_is_tiny_but_not_zero():
    """At |z| ~ 17 the p-value is ~4e-65; 1 - cdf(z) rounds to 0.0 from |z| ~ 8.3."""
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.5).astype(int)
    res = delong_test(y, y + rng.normal(scale=3.0, size=y.size), rng.normal(size=y.size))
    assert abs(res["z"]) > 10.0
    assert 0.0 < res["p_value"] < 1e-30
