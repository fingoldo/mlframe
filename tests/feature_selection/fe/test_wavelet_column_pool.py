"""The wavelet legs scanned on the edge-jobs pool equal the serial scan: same names in the same order, same values, same meta."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins as adaptive
from mlframe.feature_selection.filters import _wavelet_basis_fe_recipes as W


def _frame(n=6000, p=12):
    """A frame with step-shaped signal in three columns."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n).astype(np.float32) for i in range(p)})
    y = (((X["x0"] > 0.4).astype(int) + (X["x1"] < -0.5).astype(int) + (X["x2"].abs() < 0.3).astype(int) + (rng.random(n) < 0.1)) >= 1).astype(int).to_numpy()
    return X, y


def test_pool_equals_serial(monkeypatch):
    """One job and four jobs give the same legs."""
    X, y = _frame()
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = W.generate_wavelet_features(X, y=y)
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = W.generate_wavelet_features(X, y=y)
    assert len(serial[1]) > 0
    assert list(serial[0].columns) == list(pooled[0].columns)
    for c in serial[0].columns:
        np.testing.assert_array_equal(serial[0][c].to_numpy(), pooled[0][c].to_numpy())
    assert serial[1] == pooled[1]


def test_an_expired_budget_stops_the_scan_in_the_workers(monkeypatch):
    """The fit's wall-clock deadline reaches the pool threads: a deadline in the past yields no legs."""
    from timeit import default_timer

    from mlframe.feature_selection.filters._fe_deadline import fe_deadline_scope

    X, y = _frame()
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 4)
    with fe_deadline_scope(default_timer() - 1.0):
        out, meta = W.generate_wavelet_features(X, y=y)
    assert out.shape[1] == 0 and meta == {}
