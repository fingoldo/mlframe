"""Row count of any frame or array-like, without materialising it."""

from __future__ import annotations

from typing import Any


def n_rows(X: Any) -> int:
    """Row count across pandas / polars / numpy (``shape[0]``) and plain sequences of rows (``len``)."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(X)
