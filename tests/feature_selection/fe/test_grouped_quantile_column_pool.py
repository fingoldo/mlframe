"""The column pool of ``generate_grouped_quantile_features`` returns the serial result."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._grouped_quantile_fe import generate_grouped_quantile_features


def test_pool_equals_serial(monkeypatch):
    """One worker and four workers give the same columns, values and recipe keys in the same order."""
    rng = np.random.default_rng(8)
    n = 3000
    X = pd.DataFrame({"g1": rng.integers(0, 7, n), "g2": rng.integers(0, 4, n), **{f"x{i}": rng.normal(size=n) for i in range(6)}})
    X.loc[::11, "x2"] = np.nan
    nums = [f"x{i}" for i in range(6)]
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    enc1, rec1 = generate_grouped_quantile_features(X, ["g1", "g2"], nums)
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    enc4, rec4 = generate_grouped_quantile_features(X, ["g1", "g2"], nums)
    assert list(enc1.columns) == list(enc4.columns) and enc1.shape[1] == 36
    np.testing.assert_array_equal(enc1.to_numpy(), enc4.to_numpy())
    assert list(rec1) == list(rec4)
