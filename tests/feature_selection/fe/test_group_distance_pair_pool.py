"""The group-distance columns built on the edge-jobs pool equal the serial build: same names in the same order, same values, same recipes."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins as adaptive
from mlframe.feature_selection.filters import _group_distance_fe as G


def _same(u, v) -> bool:
    """Deep equality of the recipe dictionaries (arrays by value)."""
    if isinstance(u, dict):
        return u.keys() == v.keys() and all(_same(u[k], v[k]) for k in u)
    if isinstance(u, (list, tuple)):
        return len(u) == len(v) and all(_same(p, q) for p, q in zip(u, v))
    if isinstance(u, np.ndarray):
        return bool(np.array_equal(u, v))
    return bool(u == v)


def test_pool_equals_serial(monkeypatch):
    """One job and four jobs, two group columns, a NaN column and a group with too few rows."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({f"v{i}": rng.normal(size=n).astype(np.float32) for i in range(8)})
    X["g1"] = rng.integers(0, 6, n).astype(np.int8)
    X["g2"] = np.where(np.arange(n) < 3, 99, rng.integers(0, 20, n)).astype(np.int16)
    X.loc[rng.random(n) < 0.05, "v3"] = np.nan
    nums = [f"v{i}" for i in range(8)]
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = G.generate_group_distance_features(X, ["g1", "g2"], nums)
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = G.generate_group_distance_features(X, ["g1", "g2"], nums)
    assert serial[0].shape[1] == 3 * 2 * 8
    assert list(serial[0].columns) == list(pooled[0].columns)
    for c in serial[0].columns:
        np.testing.assert_array_equal(serial[0][c].to_numpy(), pooled[0][c].to_numpy())
    assert _same(serial[1], pooled[1])
