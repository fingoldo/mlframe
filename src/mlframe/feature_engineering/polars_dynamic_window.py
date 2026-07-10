"""Polars ``group_by_dynamic``-based rolling-window aggregation: a fast path for large panel/time-series data.

pandas ``.rolling()`` (used internally by the composite-target ``rolling_quantile_ratio``/``ewma_residual``
transforms) streams poorly at large row counts and pays per-group Python overhead when combined with an
entity groupby. Polars' ``group_by_dynamic`` computes the same fixed-width rolling windows via its native
streaming engine. This is a standalone, general-purpose fast-path utility -- NOT a modification of the
existing composite-transform internals (a large, heavily-tested module) -- usable directly, or as a drop-in
faster source for a caller building its own rolling features on big panel data.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def polars_dynamic_window_aggregate(
    df: pd.DataFrame,
    time_col: str,
    value_cols: Sequence[str],
    every: str,
    period: Optional[str] = None,
    group_col: Optional[str] = None,
    agg_funcs: Sequence[str] = ("mean",),
) -> pd.DataFrame:
    """Fixed-width rolling-window aggregates via polars ``group_by_dynamic``, optionally per entity group.

    Parameters
    ----------
    df
        Frame with a time column and columns to aggregate; ``time_col`` must be datetime-like (parseable by
        ``pd.to_datetime`` if not already).
    time_col
        The datetime column defining window boundaries.
    value_cols
        Numeric columns to aggregate within each window.
    every
        Window step (polars duration string, e.g. ``"1d"``, ``"1h"``).
    period
        Window width; defaults to ``every`` (non-overlapping tumbling windows). A wider ``period`` than
        ``every`` gives overlapping (rolling) windows.
    group_col
        Optional entity/group column; when given, windows are computed independently per group (polars
        ``group_by_dynamic(..., group_by=group_col)``).
    agg_funcs
        Aggregation names applied to every ``value_cols`` column (``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
        ``"std"``, ``"median"``, ``"count"``).

    Returns
    -------
    pd.DataFrame
        One row per window (per group, if ``group_col`` given), with ``time_col``, ``group_col`` (if given),
        and one ``{value_col}_{agg_func}`` column per ``(value_col, agg_func)`` pair.
    """
    import polars as pl

    pl_df = pl.from_pandas(df[[time_col, *value_cols] + ([group_col] if group_col else [])])
    if pl_df.schema[time_col] not in (pl.Datetime, pl.Date):
        pl_df = pl_df.with_columns(pl.col(time_col).str.to_datetime() if pl_df.schema[time_col] == pl.Utf8 else pl.col(time_col).cast(pl.Datetime))
    pl_df = pl_df.sort(time_col)

    agg_exprs = [getattr(pl.col(col), fn)().alias(f"{col}_{fn}") for col in value_cols for fn in agg_funcs]

    if group_col is not None:
        result = pl_df.group_by_dynamic(time_col, every=every, period=period, group_by=group_col, closed="left").agg(agg_exprs)
    else:
        result = pl_df.group_by_dynamic(time_col, every=every, period=period, closed="left").agg(agg_exprs)

    return result.to_pandas()


__all__ = ["polars_dynamic_window_aggregate"]
