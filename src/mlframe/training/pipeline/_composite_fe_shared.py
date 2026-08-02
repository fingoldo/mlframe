"""Shared helpers for the ``*_composite_fe.py`` family (categorical, cross-sectional, entity-time,
event-proximity-decay, latent-interaction-svd, ma-crossover, nearest-past-join, target-encoding):
small utilities independently duplicated across those modules, consolidated here so a fix can't
silently drift out of sync across copies.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import polars as pl


def to_pandas(df: Any) -> Optional[pd.DataFrame]:
    """Convert a polars DataFrame to pandas; pass through pandas/None unchanged."""
    if df is None:
        return None
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def attach_new_columns(df: Any, new_cols: "pd.DataFrame") -> Any:
    """Append ``new_cols`` (a pandas frame, same row count/order as ``df``) onto ``df``, preserving ``df``'s own frame type."""
    if new_cols.shape[1] == 0:
        return df
    if isinstance(df, pl.DataFrame):
        return df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)
