"""Shared array-coercion helpers for training.composite estimator modules: independently
duplicated across those modules, consolidated here so a fix can't silently drift out of sync
across copies.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def n_features(X: Any) -> int:
    """Robust feature count for pandas / polars / 2-D ndarray (named-column length first, then ``shape[1]``).

    A 1-D array (``shape == (n,)``) is treated as a single feature column (matching the sklearn convention of
    ``X.reshape(-1, 1)`` for a bare 1-D feature vector), not 0.
    """
    cols = getattr(X, "columns", None)
    if cols is not None:
        try:
            return len(cols)
        except TypeError:
            pass
    shape = getattr(X, "shape", None)
    if shape is not None:
        if len(shape) >= 2:
            return int(shape[1])
        if len(shape) == 1:
            return 1
    return 0


def to_1d_numpy(y: Any) -> np.ndarray:
    """Coerce any array-like (pandas/polars Series, list, 2-D column vector) to a flat float64 ndarray."""
    return np.asarray(y, dtype=np.float64).ravel()
