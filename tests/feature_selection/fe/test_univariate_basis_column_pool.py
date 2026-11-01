"""The column pool of ``generate_univariate_basis_features`` returns the serial result."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._orthogonal_univariate_fe import generate_univariate_basis_features


def test_pool_equals_serial(monkeypatch):
    """Four workers and one worker emit the same columns with the same values (a NaN column is skipped in both)."""
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    X.loc[::7, "f2"] = np.nan  # a skipped column
    y = (X["f0"].fillna(0) ** 2 + rng.normal(size=n) > 1).astype(int).to_numpy()
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 1)
    serial = generate_univariate_basis_features(X, y=y, degrees=(2, 3))
    monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: 4)
    pooled = generate_univariate_basis_features(X, y=y, degrees=(2, 3))
    assert list(serial.columns) == list(pooled.columns) and len(serial.columns) > 0
    np.testing.assert_array_equal(serial.to_numpy(), pooled.to_numpy())
