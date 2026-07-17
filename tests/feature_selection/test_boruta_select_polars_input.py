"""Regression test: ``boruta_select`` crashed on polars.DataFrame input.

``hasattr(X, "columns")`` also matches ``polars.DataFrame`` (which has no ``.reset_index()`` and isn't
accepted by ``pd.concat``), so the pandas-only shadow-construction branch raised
``AttributeError: 'DataFrame' object has no attribute 'reset_index'`` for any polars caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_selection.filters._boruta import boruta_select


def _importance_fn(X, y):
    """Toy per-column importance: |correlation| with the centered target, via a raw dot product."""
    X = np.asarray(X)
    return np.abs(X.T @ (y - y.mean()))


def test_boruta_select_accepts_polars_frame_input():
    """boruta_select on a polars.DataFrame must not crash and must match the pandas-input result."""
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((200, 5))
    y = (X_np[:, 0] + X_np[:, 1] > 0).astype(float)
    X_pd = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(5)])
    X_pl = pl.DataFrame(X_pd)

    result_pd = boruta_select(X_pd, y, _importance_fn, n_iterations=5, random_state=0)
    result_pl = boruta_select(X_pl, y, _importance_fn, n_iterations=5, random_state=0)

    assert result_pl["hit_counts"].tolist() == result_pd["hit_counts"].tolist()
    assert result_pl["decision"] == result_pd["decision"]
