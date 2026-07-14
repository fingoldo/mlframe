"""Nearest-past-value ("as-of") join: match each row to its most recent historical analog by group keys.

A common feature-engineering need is "what was this entity's most recent known value of feature X as of this
row's timestamp" -- naively implemented via a per-row/per-group lookup loop, this is O(n^2) or worse. Both
polars and pandas provide a native as-of join (``join_asof``/``merge_asof``) that does this in a single
sorted-merge pass; this module is a thin, dataframe-agnostic wrapper so callers don't have to special-case
polars vs pandas call conventions, and defaults to the faster polars path when the library is available.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _nearest_past_join_single_tier(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str,
    by_list: List[str],
    right_value_cols: List[str],
    suffix: str,
) -> pd.DataFrame:
    """One backward as-of match at a single fixed key schema; body of the original single-key implementation."""
    try:
        import polars as pl

        left_pl = pl.from_pandas(left_df[[on, *by_list]].reset_index(drop=True)).with_row_index("_row_idx")
        right_pl = pl.from_pandas(right_df[[on, *by_list, *right_value_cols]].reset_index(drop=True))
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
        right_sorted = right_df[[on, *by_list, *right_value_cols]].sort_values(on)
        merged = pd.merge_asof(left_sorted, right_sorted, on=on, by=by_list or None, direction="backward", suffixes=("", suffix))
        merged = merged.set_index("index").sort_index()
        merged.index.name = None
        return merged


def nearest_past_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str,
    by: Optional[Sequence[str]] = None,
    right_value_cols: Optional[Sequence[str]] = None,
    suffix: str = "_asof",
    fallback_by_chain: Optional[Sequence[Optional[Sequence[str]]]] = None,
    min_group_size: int = 1,
    tier_col: Optional[str] = None,
) -> pd.DataFrame:
    """Attach each ``left_df`` row's nearest-past ``right_df`` row's value columns, matched by ``by`` keys.

    Parameters
    ----------
    left_df
        Rows to enrich; must contain ``on`` and every column in ``by`` (and in every ``fallback_by_chain`` tier).
    right_df
        Historical rows to match against; must contain ``on`` and every column in ``by``.
    on
        The ordering column (e.g. a timestamp/period id) both frames are matched on -- for each ``left_df``
        row, the LATEST ``right_df`` row within the same ``by`` group with ``right_df[on] <= left_df[on]`` is
        selected (backward as-of match; no future leakage).
    by
        Grouping key columns (e.g. entity id) that must match exactly; ``None`` matches across the whole
        frame with no grouping. This is the first (finest) tier when ``fallback_by_chain`` is given.
    right_value_cols
        Columns from ``right_df`` to attach; defaults to every column in ``right_df`` other than ``on``/``by``.
    suffix
        Appended to attached column names that collide with an existing ``left_df`` column.
    fallback_by_chain
        Opt-in multi-key fallback chain (default ``None`` -- disabled, single fixed-key behavior, output
        bit-identical to omitting this param). When given, each element is a coarser ``by``-key schema
        (``None`` or ``[]`` for a global, keyless nearest-past match) tried in order for rows that remain
        unmatched -- or matched against a group with fewer than ``min_group_size`` historical rows ("too
        sparse") -- after the previous tier. A typical chain narrows a fine key (e.g. time-of-day+weekday)
        down to a coarser one (time-of-day only) and finally to no key at all, so every row gets the best
        available analog instead of ``NaN`` whenever the finest key happens to be too sparse for it.
    min_group_size
        Minimum number of historical ``right_df`` rows in a row's ``by``-key group (of the tier actually
        used) for a tier's match to be accepted as non-sparse; only consulted when ``fallback_by_chain`` is
        given. Groups smaller than this are treated as unmatched at that tier even if a nearest-past value
        was found, and the row falls through to the next, coarser tier.
    tier_col
        If given (only meaningful together with ``fallback_by_chain``), the name of an added integer column
        recording which tier matched each row: 0 for ``by`` itself, 1.. for ``fallback_by_chain`` entries in
        order, -1 if no tier produced a match at all.

    Returns
    -------
    pd.DataFrame
        ``left_df`` with the matched ``right_df`` value columns appended (``NaN`` where no eligible past row
        exists for that group, across every tier tried).
    """
    by_list = list(by) if by else []
    if right_value_cols is None:
        right_value_cols = [c for c in right_df.columns if c != on and c not in by_list]
    right_value_cols = list(right_value_cols)

    if fallback_by_chain is None:
        return _nearest_past_join_single_tier(left_df, right_df, on, by_list, right_value_cols, suffix)

    tiers: List[List[str]] = [by_list] + [list(tier) if tier else [] for tier in fallback_by_chain]

    out = left_df.reset_index(drop=True).copy()
    attached_names = [f"{col}{suffix}" if col in out.columns else col for col in right_value_cols]
    for new_name in attached_names:
        out[new_name] = pd.NA
    matched_tier = pd.Series(-1, index=out.index, dtype="int64")
    unresolved = pd.Series(True, index=out.index)

    for tier_idx, tier_by_list in enumerate(tiers):
        if not unresolved.any():
            break
        unresolved_idx = out.index[unresolved.to_numpy()]
        candidate = _nearest_past_join_single_tier(left_df.loc[unresolved_idx].reset_index(drop=True), right_df, on, tier_by_list, right_value_cols, suffix)
        candidate.index = unresolved_idx

        # a group with fewer than ``min_group_size`` historical rows counts as "too sparse" even when a
        # nearest-past value technically exists, so its rows still fall through to the next, coarser tier.
        # vectorized via merge (not a per-row dict/tuple lookup, which dominates wall time on large inputs).
        if tier_by_list:
            group_sizes = right_df.groupby(tier_by_list).size().rename("_group_size").reset_index()
            row_group_sizes = left_df.loc[unresolved_idx, tier_by_list].merge(group_sizes, on=tier_by_list, how="left")["_group_size"].fillna(0).to_numpy()
        else:
            row_group_sizes = np.full(len(candidate), len(right_df))

        sparse = row_group_sizes < min_group_size
        resolved = pd.Series(False, index=candidate.index)
        for new_name in attached_names:
            col_resolved = candidate[new_name].notna().to_numpy() & ~sparse
            resolved_idx = candidate.index[col_resolved]
            out.loc[resolved_idx, new_name] = candidate.loc[resolved_idx, new_name].to_numpy()
            resolved |= col_resolved

        matched_tier.loc[candidate.index[resolved.to_numpy()]] = tier_idx
        unresolved.loc[candidate.index[resolved.to_numpy()]] = False

    if tier_col is not None:
        out[tier_col] = matched_tier.to_numpy()
    return out


__all__ = ["nearest_past_join"]
