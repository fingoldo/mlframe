"""Grouped-FE value-column auto-detection covers wide frames, not just the first eight columns."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._group_distance_fe import hybrid_group_distance_fe


def test_group_shifted_column_beyond_eighth_is_found():
    """A column whose per-group mean carries the target (column 12 of 14) gets a distance feature under default auto-detection."""
    n, groups = 20000, 300
    rng = np.random.default_rng(0)
    mu = rng.normal(size=groups)
    g = rng.integers(0, groups, n)
    X = pd.DataFrame(rng.standard_normal((n, 14)), columns=[f"v{i}" for i in range(14)])
    X["v12"] = X["v12"] * 0.5 + mu[g]
    X["g"] = g
    y = ((mu[g] > 0) ^ (rng.random(n) < 0.05)).astype(int)
    assert "gzdist(v12|g)" in hybrid_group_distance_fe(X, y, group_cols=["g"])[1]
