"""The candidate pool of ``score_features_by_jmim`` returns the serial scores."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._orthogonal_jmim_fe import score_features_by_jmim


def test_pool_equals_serial(monkeypatch):
    """One worker and four workers give the identical frame of scores."""
    rng = np.random.default_rng(6)
    n = 5000
    raw = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    y = (raw["x0"] * raw["x1"] + 0.3 * rng.normal(size=n) > 0).astype(int).to_numpy()
    eng = pd.DataFrame({f"x{i}__He{d}": raw[f"x{i}"] ** d for i in range(6) for d in (2, 3)})
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = score_features_by_jmim(raw, eng, y)
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = score_features_by_jmim(raw, eng, y)
    pd.testing.assert_frame_equal(serial, pooled)
    assert len(serial) == 12
