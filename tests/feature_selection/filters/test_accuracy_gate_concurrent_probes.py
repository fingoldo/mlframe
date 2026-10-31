"""Probing the engineered columns of the accuracy gate on a thread pool gives the same keep / drop decisions as probing them one after another."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins as AN
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._engineered_gates import _gate_base_columns


def _frame(n=6000, seed=0):
    """Raw sources with a quadratic target; engineered columns: a useful square, a useless noise column and an exact copy of the source."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    y = ((X["x0"] ** 2 + 0.3 * rng.normal(size=n)) > 1.0).astype(np.int64).to_numpy()
    gate_cols = []
    for i in range(4):
        X[f"x{i}__sq"] = X[f"x{i}"] ** 2
        X[f"x{i}__noise"] = rng.normal(size=n)
        X[f"x{i}__copy"] = X[f"x{i}"] * 1.0
        gate_cols += [(f"x{i}__sq", f"x{i}", False), (f"x{i}__noise", f"x{i}", False), (f"x{i}__copy", f"x{i}", False)]
    return X, y, gate_cols


def test_pool_and_serial_probes_agree(monkeypatch):
    """Same surviving columns and same dropped columns with 1 and with 4 workers; the useful square of x0 survives."""
    X, y, gate_cols = _frame()
    monkeypatch.setattr(AN, "_resolve_edge_jobs", lambda n_jobs, n_miss: 1)
    serial = _gate_base_columns(X, gate_cols, y, 0)
    monkeypatch.setattr(AN, "_resolve_edge_jobs", lambda n_jobs, n_miss: 4)
    pooled = _gate_base_columns(X, gate_cols, y, 0)
    assert serial == pooled
    assert "x0__sq" in serial[0].get("x0", [])
