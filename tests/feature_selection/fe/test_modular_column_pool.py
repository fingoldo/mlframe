"""The column pool of ``generate_modular_features`` returns the serial result."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._periodic_fe import generate_modular_features


def test_pool_equals_serial(monkeypatch):
    """One worker and four workers give the same columns in the same order with the same values."""
    rng = np.random.default_rng(12)
    X = pd.DataFrame({f"x{i}": rng.normal(size=2000) * 50 for i in range(8)})
    X["t"] = np.arange(2000) * 3600.0 + 1.6e9
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = generate_modular_features(X, periods=(7.0, 24.0))
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = generate_modular_features(X, periods=(7.0, 24.0))
    assert list(serial.columns) == list(pooled.columns) and serial.shape[1] == 9 * 2 * 3
    np.testing.assert_array_equal(serial.to_numpy(), pooled.to_numpy())
