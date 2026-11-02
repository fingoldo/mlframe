"""Pass 2 of the accuracy gate (adaptive Fourier columns) gives the same drops whatever the pool size."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._engineered_gates import _gate_fourier_columns


def _frame(n=6000, seed=0):
    """A raw column, a genuine oscillation of it, noise columns presented as Fourier columns, and a raw column with many NaN."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = (np.sin(4 * x) + 0.3 * rng.normal(size=n) > 0).astype(int)
    cols = {"x": x, "x__sin": np.sin(4 * x), "z": rng.normal(size=n)}
    for i in range(6):
        cols[f"x__noise{i}"] = rng.normal(size=n)
    xm = x.copy()
    xm[::7] = np.nan
    cols["m"] = xm
    cols["m__sin"] = np.sin(4 * np.nan_to_num(xm))
    return pd.DataFrame(cols), y


def test_drops_are_the_same_for_every_pool_size(monkeypatch):
    """The genuine oscillation stays, the noise columns and the MNAR-source column are dropped, with one worker and with four."""
    X, y = _frame()
    gate_cols = [("x__sin", "x", True)] + [(f"x__noise{i}", "x", True) for i in range(6)] + [("m__sin", "m", True)]
    results = []
    for jobs in (1, 4):
        monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, j=jobs, **k: j)
        results.append(_gate_fourier_columns(X, gate_cols, {}, y, 0, None, True))
    assert results[0] == results[1]
    assert "x__sin" not in results[0] and "m__sin" in results[0]
    assert sum(name.startswith("x__noise") for name in results[0]) >= 4
