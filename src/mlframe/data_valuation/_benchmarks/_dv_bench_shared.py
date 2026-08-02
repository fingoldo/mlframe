"""Shared synthetic-fixture builder used by several ``_benchmarks/`` scripts in this directory:
independently duplicated across those scripts, consolidated here so a fix can't silently drift out
of sync across copies.
"""
from __future__ import annotations

import numpy as np


def make_blob_fixture(n_train: int, n_val: int, n_features: int, seed: int):
    """Synthetic 2-class blob-style fixture of the given (n_train, n_val, n_features) shape."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)
    X_val = rng.standard_normal((n_val, n_features))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)
    return X_train, y_train, X_val, y_val
