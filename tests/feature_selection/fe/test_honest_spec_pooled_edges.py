"""The honest-null spec built with the per-column edges on a pool equals the one built serially."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._honest_spec import build_honest_spec


def _same(a, b) -> bool:
    """Deep equality of the spec dicts (arrays by value, NaN equal)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(p, q) for p, q in zip(a, b))
    return bool(a == b)


def test_pooled_equals_serial(monkeypatch):
    """Continuous, low-cardinality, NaN-holding, categorical and absent columns give the same spec on 1 and 4 threads."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(6)})
    X["low"] = rng.integers(0, 5, n)
    X["nan"] = np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n))
    X["cat"] = pd.Categorical(rng.integers(0, 3, n))
    y = (X["c0"] + rng.normal(size=n) > 0).astype(int).to_numpy()
    cols = [*X.columns, "missing"]
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = build_honest_spec(X, cols, y, 3)
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = build_honest_spec(X, cols, y, 3)
    assert serial is not None and _same(serial, pooled)
