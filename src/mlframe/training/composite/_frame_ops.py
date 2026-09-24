"""Appending a column to a frame without copying the frame.

``out = df.copy(); out[col] = values`` is the obvious way and the expensive one: it duplicates every existing block to add
one. ``pd.concat([df, series], axis=1)`` with ``copy=False`` builds a new frame whose existing columns are the source's
buffers, which is what the callers need - the caller's frame must not gain the column, but its data need not be duplicated.
Polars ``with_columns`` already works that way. pandas 3 is zero-copy by default and dropped the ``copy`` keyword, so the
keyword is passed only on older versions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["append_column"]

_CONCAT_NO_COPY: dict[str, Any] = {} if int(pd.__version__.split(".")[0]) >= 3 else {"copy": False}


def append_column(frame: Any, name: str, values: np.ndarray) -> Any:
    """Return a new frame of ``frame``'s type carrying ``values`` as column ``name``, sharing the existing columns.

    Parameters
    ----------
    frame
        A pandas or polars DataFrame, or a 2-D array.
    name
        The new column's name (ignored for an array input, where the values become the last column).
    values
        One value per row.

    Returns
    -------
    Any
        A new frame; ``frame`` itself is never mutated. A column of ``name`` already present is replaced.
    """
    values = np.asarray(values).reshape(-1)
    if isinstance(frame, pd.DataFrame):
        new_col = pd.Series(values, index=frame.index, name=name)
        # On pandas 2.x with copy-on-write off (the default there, and what this project runs) ``concat`` copies the
        # existing block whatever ``copy=`` says - measured on 2.3.3. That copy is the price of the contract in the
        # docstring: a frame that SHARES the caller's buffers without CoW lets a write to the result land in the
        # caller's frame (a mapping-constructor variant was tried and did exactly that). Isolation wins; pandas 3,
        # where CoW is always on, shares lazily and stays isolated.
        base = frame.drop(columns=[name]) if name in frame.columns else frame
        return pd.concat([base, new_col], axis=1, **_CONCAT_NO_COPY)
    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore[assignment]
    if pl is not None and isinstance(frame, pl.DataFrame):
        return frame.with_columns(pl.Series(name, values))
    return np.concatenate([np.asarray(frame, dtype=np.float64), values.reshape(-1, 1).astype(np.float64)], axis=1)
