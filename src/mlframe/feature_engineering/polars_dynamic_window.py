"""Polars ``group_by_dynamic``-based rolling-window aggregation: a fast path for large panel/time-series data.

pandas ``.rolling()`` (used internally by the composite-target ``rolling_quantile_ratio``/``ewma_residual``
transforms) streams poorly at large row counts and pays per-group Python overhead when combined with an
entity groupby. Polars' ``group_by_dynamic`` computes the same fixed-width rolling windows via its native
streaming engine. This is a standalone, general-purpose fast-path utility -- NOT a modification of the
existing composite-transform internals (a large, heavily-tested module) -- usable directly, or as a drop-in
faster source for a caller building its own rolling features on big panel data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import pandas as pd

if TYPE_CHECKING:
    import polars as pl


def _prepare_lazy_frame(
    df: pd.DataFrame, time_col: str, value_cols: Sequence[str], group_col: Optional[str], time_unit: Optional[Literal["ns", "us", "ms", "s", "d"]] = None
) -> "pl.LazyFrame":
    """Convert the needed pandas columns to a polars LazyFrame with time_col coerced to Datetime.

    A numeric time column (an integer day index, epoch seconds) has no unit of its own: it is converted with ``time_unit`` (``"s"``, ``"ms"``, ``"us"``,
    ``"ns"``, ``"d"``) or rejected. ``cast(pl.Datetime)`` read it as microseconds, which put a 0..365 day index inside the first millisecond after the
    epoch and collapsed every row into one window.
    """
    import polars as pl

    pl_df = pl.from_pandas(df[[time_col, *value_cols] + ([group_col] if group_col else [])])
    dtype = pl_df.schema[time_col]
    if dtype not in (pl.Datetime, pl.Date):
        if dtype == pl.Utf8:
            pl_df = pl_df.with_columns(pl.col(time_col).str.to_datetime())
        elif dtype.is_numeric():
            if time_unit is None:
                raise ValueError(
                    f"polars_dynamic_window_aggregate: time column {time_col!r} is numeric ({dtype}); pass time_unit ('s', 'ms', 'us', 'ns' or 'd') so its values can be read as epoch offsets"
                )
            pl_df = pl_df.with_columns(pl.from_epoch(pl.col(time_col), time_unit=time_unit).alias(time_col))
        else:
            pl_df = pl_df.with_columns(pl.col(time_col).cast(pl.Datetime))
    return pl_df.sort(time_col).lazy()


def polars_dynamic_window_aggregate(
    df: pd.DataFrame,
    time_col: str,
    value_cols: Sequence[str],
    every: str,
    period: Optional[str] = None,
    group_col: Optional[str] = None,
    agg_funcs: Sequence[str] = ("mean",),
    periods: Optional[Sequence[str]] = None,
    label: Literal["left", "right", "datapoint"] = "right",
    time_unit: Optional[Literal["ns", "us", "ms", "s", "d"]] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
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
        ``every`` gives overlapping (rolling) windows. Ignored when ``periods`` is given.
    group_col
        Optional entity/group column; when given, windows are computed independently per group (polars
        ``group_by_dynamic(..., group_by=group_col)``).
    agg_funcs
        Aggregation names applied to every ``value_cols`` column (``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
        ``"std"``, ``"median"``, ``"count"``).
    periods
        Opt-in multi-window mode: a list of window widths (polars duration strings, e.g.
        ``["7d", "14d", "30d"]``) to compute the SAME aggregation at, in one call. The source frame is
        converted from pandas, cast, and sorted exactly once and reused (via polars lazy evaluation +
        ``pl.collect_all``) across all requested widths, rather than repeating that prep once per width as a
        naive per-window loop calling this function would. When given, ``period`` is ignored and the return
        value is a ``dict`` mapping each period string to its result frame, instead of a single frame.

    label
        Which window boundary labels the row in ``time_col``: ``"right"`` (default, the exclusive END of the window) or ``"left"`` (its start). With the
        default, joining the result back onto events by ``time_col`` only attaches features computed from rows BEFORE that time; ``"left"`` labels a
        window ``[t, t + period)`` with ``t``, so such a join would attach the following period (leakage).
    time_unit
        Unit of a numeric ``time_col`` (``"s"``, ``"ms"``, ``"us"``, ``"ns"``, ``"d"``); required for numeric time columns, ignored for datetimes.

    Returns
    -------
    pd.DataFrame
        One row per window (per group, if ``group_col`` given), with ``time_col``, ``group_col`` (if given),
        and one ``{value_col}_{agg_func}`` column per ``(value_col, agg_func)`` pair. When ``periods`` is
        given, a ``Dict[str, pd.DataFrame]`` mapping each period string to that result frame instead.
    """
    import polars as pl

    agg_exprs = [getattr(pl.col(col), fn)().alias(f"{col}_{fn}") for col in value_cols for fn in agg_funcs]

    if periods is not None:
        lazy_src = _prepare_lazy_frame(df, time_col, value_cols, group_col, time_unit)
        lazy_queries = []
        for p in periods:
            if group_col is not None:
                lazy_queries.append(lazy_src.group_by_dynamic(time_col, every=every, period=p, group_by=group_col, closed="left", label=label).agg(agg_exprs))
            else:
                lazy_queries.append(lazy_src.group_by_dynamic(time_col, every=every, period=p, closed="left", label=label).agg(agg_exprs))
        collected = pl.collect_all(lazy_queries)
        return {p: res.to_pandas() for p, res in zip(periods, collected)}

    pl_df = _prepare_lazy_frame(df, time_col, value_cols, group_col, time_unit).collect()

    if group_col is not None:
        result = pl_df.group_by_dynamic(time_col, every=every, period=period, group_by=group_col, closed="left", label=label).agg(agg_exprs)
    else:
        result = pl_df.group_by_dynamic(time_col, every=every, period=period, closed="left", label=label).agg(agg_exprs)

    return result.to_pandas()


__all__ = ["polars_dynamic_window_aggregate"]
