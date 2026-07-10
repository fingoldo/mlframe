"""Nearest-past-value ("as-of") join: match each row to its most recent historical analog by group keys.

A common feature-engineering need is "what was this entity's most recent known value of feature X as of this
row's timestamp" -- naively implemented via a per-row/per-group lookup loop, this is O(n^2) or worse. Both
polars and pandas provide a native as-of join (``join_asof``/``merge_asof``) that does this in a single
sorted-merge pass; this module is a thin, dataframe-agnostic wrapper so callers don't have to special-case
polars vs pandas call conventions, and defaults to the faster polars path when the library is available.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def nearest_past_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str,
    by: Optional[Sequence[str]] = None,
    right_value_cols: Optional[Sequence[str]] = None,
    suffix: str = "_asof",
) -> pd.DataFrame:
    """Attach each ``left_df`` row's nearest-past ``right_df`` row's value columns, matched by ``by`` keys.

    Parameters
    ----------
    left_df
        Rows to enrich; must contain ``on`` and every column in ``by``.
    right_df
        Historical rows to match against; must contain ``on`` and every column in ``by``.
    on
        The ordering column (e.g. a timestamp/period id) both frames are matched on -- for each ``left_df``
        row, the LATEST ``right_df`` row within the same ``by`` group with ``right_df[on] <= left_df[on]`` is
        selected (backward as-of match; no future leakage).
    by
        Grouping key columns (e.g. entity id) that must match exactly; ``None`` matches across the whole
        frame with no grouping.
    right_value_cols
        Columns from ``right_df`` to attach; defaults to every column in ``right_df`` other than ``on``/``by``.
    suffix
        Appended to attached column names that collide with an existing ``left_df`` column.

    Returns
    -------
    pd.DataFrame
        ``left_df`` with the matched ``right_df`` value columns appended (``NaN`` where no eligible past row
        exists for that group).
    """
    by_list = list(by) if by else []
    if right_value_cols is None:
        right_value_cols = [c for c in right_df.columns if c != on and c not in by_list]
    right_value_cols = list(right_value_cols)

    try:
        import polars as pl

        left_pl = pl.from_pandas(left_df[[on] + by_list].reset_index(drop=True)).with_row_index("_row_idx")
        right_pl = pl.from_pandas(right_df[[on] + by_list + right_value_cols].reset_index(drop=True))
        # a common source of `on` being int in one frame and float in the other (e.g. left is a computed
        # query timestamp, right is raw integer periods) -- join_asof requires identical key dtypes.
        if left_pl.schema[on] != right_pl.schema[on]:
            left_pl = left_pl.with_columns(pl.col(on).cast(pl.Float64))
            right_pl = right_pl.with_columns(pl.col(on).cast(pl.Float64))

        left_sorted = left_pl.sort(on)
        right_sorted = right_pl.sort(on)
        # both frames are sorted globally by ``on`` above, which implies per-``by``-group sortedness (a
        # subsequence of a sorted sequence is sorted) -- polars can't verify that cheaply itself when ``by``
        # groups are present and warns regardless; check_sortedness=False silences the false-positive warning.
        joined = left_sorted.join_asof(right_sorted, on=on, by=by_list or None, strategy="backward", check_sortedness=False)
        joined = joined.sort("_row_idx").drop("_row_idx")
        result_pd = joined.to_pandas()

        out = left_df.reset_index(drop=True).copy()
        for col in right_value_cols:
            new_name = f"{col}{suffix}" if col in out.columns else col
            out[new_name] = result_pd[col].to_numpy()
        return out
    except ImportError:
        left_sorted = left_df.sort_values(on).reset_index()
        right_sorted = right_df[[on] + by_list + right_value_cols].sort_values(on)
        merged = pd.merge_asof(left_sorted, right_sorted, on=on, by=by_list or None, direction="backward", suffixes=("", suffix))
        merged = merged.set_index("index").sort_index()
        merged.index.name = None
        return merged


__all__ = ["nearest_past_join"]
