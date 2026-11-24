"""Cluster 09 regression: the persistence feature set has no constant mean-birth column (audit 2026-11-20, finding 09-F14).

Vietoris-Rips H0 births are all zero, so ``mean_birth`` was a constant column. It is replaced by ``std_lifespan``, which varies with the data.
"""

import numpy as np

from mlframe.feature_engineering.transformer import persistence_diagram as pdm


def test_mean_birth_column_is_replaced_by_std_lifespan():
    """The output names std_lifespan and no longer names the constant mean_birth (09-F14)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + rng.normal(size=60) * 0.1
    Xq = rng.normal(size=(12, 3))
    df = pdm.compute_persistence_diagram_features(X, y, Xq, seed=0, column_prefix="pd")
    assert "pd_std_lifespan" in df.columns
    assert "pd_mean_birth" not in df.columns


def test_std_lifespan_is_not_constant():
    """Control: the replacement varies across query rows (09-F14)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + rng.normal(size=60) * 0.1
    Xq = rng.normal(size=(12, 3))
    col = pdm.compute_persistence_diagram_features(X, y, Xq, seed=0, column_prefix="pd")["pd_std_lifespan"].to_numpy()
    assert np.unique(col).size > 1
