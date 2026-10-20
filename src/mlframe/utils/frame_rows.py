"""Row count and row subset of any frame or array-like, without materialising it."""

from __future__ import annotations

from typing import Any

import numpy as np


def n_rows(X: Any) -> int:
    """Row count across pandas / polars / numpy (``shape[0]``) and plain sequences of rows (``len``)."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(X)


def take_rows(X: Any, idx: np.ndarray) -> Any:
    """Row subset of ``X`` at integer positions ``idx``: pandas ``iloc``, polars positional gather; never a whole-frame copy."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]
