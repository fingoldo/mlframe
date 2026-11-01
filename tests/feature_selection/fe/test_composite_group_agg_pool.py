"""The task pool of ``generate_composite_group_agg_features`` returns the serial result."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._composite_group_agg_fe import generate_composite_group_agg_features


def test_pool_equals_serial(monkeypatch):
    """One worker and four workers give the same columns, values and recipe keys in the same order."""
    rng = np.random.default_rng(5)
    n = 3000
    X = pd.DataFrame(
        {
            "g1": rng.integers(0, 6, n),
            "g2": rng.integers(0, 5, n),
            "g3": rng.integers(0, 4, n),
            **{f"x{i}": rng.normal(size=n) for i in range(5)},
        }
    )
    sets = [("g1", "g2"), ("g1", "g3"), ("g2", "g3")]
    nums = [f"x{i}" for i in range(5)]
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    enc1, rec1 = generate_composite_group_agg_features(X, sets, nums)
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    enc4, rec4 = generate_composite_group_agg_features(X, sets, nums)
    assert list(enc1.columns) == list(enc4.columns) and enc1.shape[1] > 0
    np.testing.assert_array_equal(enc1.to_numpy(), enc4.to_numpy())
    assert list(rec1) == list(rec4)
