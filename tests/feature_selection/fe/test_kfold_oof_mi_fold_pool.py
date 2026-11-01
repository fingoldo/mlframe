"""The fold pool of ``score_features_by_kfold_oof_mi`` returns the serial scores."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe import score_features_by_kfold_oof_mi


def test_pool_equals_serial(monkeypatch):
    """Two workers and one worker give the identical frame of scores."""
    rng = np.random.default_rng(4)
    n = 6000
    raw = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    y = (raw["x0"] + 0.5 * rng.normal(size=n) > 0).astype(int).to_numpy()
    eng = pd.DataFrame({f"x{i}__He2": raw[f"x{i}"] ** 2 for i in range(6)})
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = score_features_by_kfold_oof_mi(raw, eng, y)
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = score_features_by_kfold_oof_mi(raw, eng, y)
    pd.testing.assert_frame_equal(serial, pooled)
    assert len(serial) == 6
