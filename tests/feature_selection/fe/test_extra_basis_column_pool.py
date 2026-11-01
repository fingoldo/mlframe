"""The extra-basis columns scanned on the edge-jobs pool equal the serial scan, and an expired budget stops the workers."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins as adaptive
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import generate_extra_basis_features


def _same(u, v) -> bool:
    """Deep equality of the meta dictionaries (arrays by value)."""
    if isinstance(u, dict):
        return u.keys() == v.keys() and all(_same(u[k], v[k]) for k in u)
    if isinstance(u, (list, tuple)):
        return len(u) == len(v) and all(_same(p, q) for p, q in zip(u, v))
    if isinstance(u, np.ndarray):
        return bool(np.array_equal(u, v))
    return bool(u == v)


def _frame(n=4000, p=9):
    """A frame with a periodic signal in one column."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n).astype(np.float32) for i in range(p)})
    X["nan"] = np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)).astype(np.float32)
    y = ((np.sin(5 * X["x0"]) + 0.5 * X["x1"] ** 2 + rng.normal(size=n) * 0.5) > 0.3).astype(int).to_numpy()
    return X, y


def test_pool_equals_serial(monkeypatch):
    """One job and four jobs give the same columns in the same order with the same meta."""
    X, y = _frame()
    kw = {"y": y, "extra_bases": ("spline", "fourier", "wavelet"), "fourier_adaptive": True}
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = generate_extra_basis_features(X, **kw)
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = generate_extra_basis_features(X, **kw)
    assert serial[0].shape[1] > 0
    assert list(serial[0].columns) == list(pooled[0].columns)
    for c in serial[0].columns:
        np.testing.assert_array_equal(serial[0][c].to_numpy(), pooled[0][c].to_numpy())
    assert _same(serial[1], pooled[1])


def test_an_expired_budget_stops_the_workers(monkeypatch):
    """A deadline in the past reaches the pool threads: nothing is engineered."""
    from timeit import default_timer

    from mlframe.feature_selection.filters._fe_deadline import fe_deadline_scope

    X, y = _frame()
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 4)
    with fe_deadline_scope(default_timer() - 1.0):
        out, meta = generate_extra_basis_features(X, y=y, extra_bases=("spline", "fourier"))
    assert out.shape[1] == 0 and meta == {}
