"""The OOF-warp candidates scored on the edge-jobs pool equal the serial scan: same columns in the same order, same scores, same cross-fitted values."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins as adaptive
from mlframe.feature_selection.filters import _oof_warp_fe as W


@pytest.mark.parametrize("multiclass", ["rank", "ovr"])
def test_pool_equals_serial(monkeypatch, multiclass):
    """One job and four jobs; a nominal-like column is skipped in both."""
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({f"v{i}": rng.normal(size=n).astype(np.float32) for i in range(8)})
    X["nominal"] = rng.integers(0, 3, n).astype(np.float32)
    y = ((np.abs(X["v0"]) > 1) ^ (X["v1"] > 0)).astype(int).to_numpy() + (rng.random(n) < 0.1).astype(int)
    cols = list(X.columns)
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = W.warp_candidates(X, y, cols, multiclass=multiclass)
    monkeypatch.setattr(adaptive, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = W.warp_candidates(X, y, cols, multiclass=multiclass)
    assert len(serial) > 0 and "nominal" not in {c["col"] for c in serial}
    assert [(c["col"], c["cls"]) for c in serial] == [(c["col"], c["cls"]) for c in pooled]
    for a, b in zip(serial, pooled):
        for key in ("mi_raw", "mi_warp", "gain", "se", "bulk_gain", "bulk_se"):
            assert a[key] == b[key]
        np.testing.assert_array_equal(a["fit"]["oof"], b["fit"]["oof"])
