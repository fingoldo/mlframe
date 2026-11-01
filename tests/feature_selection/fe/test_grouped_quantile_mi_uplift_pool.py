"""Binning the engineered columns on a pool leaves the uplift table equal to the serial computation."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._grouped_quantile_fe import score_grouped_quantile_by_mi_uplift


def test_pooled_binning_equals_serial(monkeypatch):
    """The table from several threads equals the table from one."""
    rng = np.random.default_rng(0)
    n = 3000
    raw = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (raw["a"] + rng.normal(size=n) > 0).astype(int).to_numpy()
    eng = pd.DataFrame({f"e{i}": raw["a" if i % 2 else "b"] * (i + 1) + rng.normal(size=n) for i in range(6)})
    eng["tiny"] = rng.integers(0, 4, n)
    src = {c: ("a" if i % 2 else "b") for i, c in enumerate(eng.columns)}
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = score_grouped_quantile_by_mi_uplift(raw, eng, y, eng_to_source=src)
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = score_grouped_quantile_by_mi_uplift(raw, eng, y, eng_to_source=src)
    pd.testing.assert_frame_equal(serial, pooled)
