"""Multi-fixed-lookback-horizon aggregation: one aggregate feature set per horizon, in a single call.

Computing a single all-history aggregate per entity discards a real signal: how RECENT the driving events
were. A 9th-place Home-Credit team's fix was to separately aggregate at several fixed lookback horizons
("last 3 months", "last 6 months", "last year", ...) rather than one all-history number -- recent-vs-older
behavior divergence is itself informative (a worsening trend, a recent spike). This helper generalizes that
pattern to an arbitrary horizon list in one call, complementing the existing leakage-safe as-of aggregate.
"""
from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

from mlframe.feature_engineering.as_of_aggregate import leakage_safe_aggregate


def multi_window_aggregate(
    history_df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    as_of: pd.DataFrame,
    agg_funcs: Dict[str, Sequence[str]],
    lookback_horizons: Sequence[float],
    query_entity_col: str = "as_of",
) -> pd.DataFrame:
    """Aggregate ``history_df`` per entity at several fixed lookback horizons before each query's cutoff.

    Parameters
    ----------
    history_df, entity_col, time_col, agg_funcs, query_entity_col
        Same contract as :func:`mlframe.feature_engineering.as_of_aggregate.leakage_safe_aggregate`.
    as_of
        Query frame with the entity key and a per-row cutoff column (name given by ``query_entity_col``).
    lookback_horizons
        Window lengths (same units as ``time_col``), e.g. ``[90, 180, 270, 365]`` for "last 3/6/9/12 months"
        in day units. Each horizon ``h`` aggregates only rows with ``cutoff - h <= time_col < cutoff``
        (strictly before the cutoff, same leakage-safety contract as the full-history version).

    Returns
    -------
    pd.DataFrame
        One row per query row, entity key plus ``{agg_column}_{agg_name}_last_{horizon}`` columns for every
        ``(agg_column, agg_name, horizon)`` combination.
    """
    if not lookback_horizons:
        raise ValueError("multi_window_aggregate: lookback_horizons must be non-empty")

    query = as_of.reset_index(drop=True)
    out = pd.DataFrame({entity_col: query[entity_col].to_numpy()})

    for horizon in lookback_horizons:
        windowed_history = history_df.copy()
        # tag each history row with the horizon-specific "window start" so a per-horizon leakage_safe_aggregate
        # call only sees rows within [cutoff - horizon, cutoff) -- reuse the vetted as-of aggregation machinery
        # rather than duplicating its cumsum/searchsorted logic.
        merged_query = query[[entity_col, query_entity_col]].copy()
        merged_query["_window_start"] = merged_query[query_entity_col] - horizon

        # filter history to rows that could POSSIBLY fall in ANY query row's window for this horizon is not
        # correct per-row (different entities/cutoffs), so instead run leakage_safe_aggregate against the
        # window-shifted cutoff, then separately re-run with the window START as an exclusion floor by
        # dropping history rows before window start via a per-entity merge-asof-style filter is unnecessary:
        # leakage_safe_aggregate already computes cumulative sums up to the searchsorted cutoff position; the
        # window-start floor is enforced by subtracting the cumulative aggregate AT the window start.
        upper = leakage_safe_aggregate(
            windowed_history, entity_col=entity_col, time_col=time_col,
            as_of=query[[entity_col, query_entity_col]], agg_funcs=agg_funcs, query_entity_col=query_entity_col,
        )
        lower_query = query[[entity_col]].copy()
        lower_query[query_entity_col] = merged_query["_window_start"].to_numpy()
        lower = leakage_safe_aggregate(
            windowed_history, entity_col=entity_col, time_col=time_col,
            as_of=lower_query, agg_funcs=agg_funcs, query_entity_col=query_entity_col,
        )

        for col, fns in agg_funcs.items():
            for fn in fns:
                colname = f"{col}_{fn}"
                windowed_name = f"{colname}_last_{horizon}"
                if fn in ("sum", "count"):
                    out[windowed_name] = (upper[colname].fillna(0.0) - lower[colname].fillna(0.0)).to_numpy()
                elif fn == "mean":
                    upper_sum = upper.get(f"{col}_sum")
                    upper_count = upper.get(f"{col}_count")
                    if upper_sum is None or upper_count is None:
                        raise ValueError(f"multi_window_aggregate: computing windowed 'mean' for {col!r} requires 'sum' and 'count' also in agg_funcs[{col!r}]")
                    lower_sum = lower.get(f"{col}_sum")
                    lower_count = lower.get(f"{col}_count")
                    win_sum = upper_sum.fillna(0.0) - lower_sum.fillna(0.0)
                    win_count = upper_count.fillna(0.0) - lower_count.fillna(0.0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out[windowed_name] = np.where(win_count > 0, win_sum / win_count, np.nan)
                else:
                    # non-additive aggs (min/max/median/...) can't be derived by subtracting two cumulative
                    # snapshots; fall back to the direct windowed computation for just this (col, fn, horizon).
                    out[windowed_name] = _direct_window_agg(history_df, entity_col, time_col, query, query_entity_col, horizon, col, fn)

    return out


def _direct_window_agg(
    history_df: pd.DataFrame, entity_col: str, time_col: str, query: pd.DataFrame, query_entity_col: str, horizon: float, col: str, fn: str
) -> np.ndarray:
    history_groups = {entity: grp for entity, grp in history_df.groupby(entity_col, sort=False)}
    out = np.full(len(query), np.nan)
    for entity, entity_queries in query.groupby(entity_col, sort=False):
        entity_history = history_groups.get(entity)
        if entity_history is None or entity_history.empty:
            continue
        times = entity_history[time_col].to_numpy()
        order = np.argsort(times)
        sorted_times = times[order]
        sorted_col = entity_history[col].to_numpy()[order]
        for idx, cutoff in zip(entity_queries.index, entity_queries[query_entity_col]):
            lo = np.searchsorted(sorted_times, cutoff - horizon, side="left")
            hi = np.searchsorted(sorted_times, cutoff, side="left")
            if hi > lo:
                out[idx] = getattr(pd.Series(sorted_col[lo:hi]), fn)()
    return out


__all__ = ["multi_window_aggregate"]
