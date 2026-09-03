"""Shared helpers for the ``*_composite_fe.py`` family (categorical, cross-sectional, entity-time,
event-proximity-decay, latent-interaction-svd, ma-crossover, nearest-past-join, target-encoding):
small utilities independently duplicated across those modules, consolidated here so a fix can't
silently drift out of sync across copies.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import polars as pl


def row_count(df: Any) -> int:
    """Row count of ``df``, or 0 if ``df`` is None."""
    return df.shape[0] if df is not None else 0


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
    # new_cols is guaranteed to be in the SAME ROW ORDER as df (contract above), but callers commonly
    # build it with a fresh RangeIndex(0..n-1) rather than df's own (possibly non-contiguous, e.g. after
    # an upstream df.iloc[train_idx] split) index. Both df.join() and pd.concat(axis=1) align by INDEX
    # LABEL, not row position, so a mismatched index here silently produces NaN or cross-row-
    # misattributed values for most rows. Align new_cols' index to df's own index (positional, by the
    # same-row-order contract) before either alignment path runs.
    if not new_cols.index.equals(df.index):
        new_cols = new_cols.set_axis(df.index, axis=0)
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)
