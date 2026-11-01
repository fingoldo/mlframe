"""The column pool of the rounding and digit generators returns the serial result."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._numeric_decompose_fe import generate_digit_features, generate_rounding_features


def test_pool_equals_serial(monkeypatch):
    """One worker and four workers give identical columns in identical order, for both generators."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({f"x{i}": np.round(rng.normal(size=3000) * 10, 3) for i in range(8)})
    X.loc[::13, "x3"] = np.nan
    for gen in (generate_rounding_features, generate_digit_features):
        monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
        serial = gen(X)
        monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
        pooled = gen(X)
        assert list(serial.columns) == list(pooled.columns) and serial.shape[1] > 8
        np.testing.assert_array_equal(serial.to_numpy(), pooled.to_numpy())
