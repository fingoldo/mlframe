"""``windowed_edge_aggregate_diff``: first-N vs last-N record aggregate diff/ratio features per group.

Source: 1st_amex-default-prediction.md ("diff: last value - nth value", user/month-based rank features) and
1st_home-credit-default-risk.md (Ryan's aggregates over "last 3, 5 and first 2, 4 applications"). Complements
mlframe's existing recency/rolling-window aggregations (which pick window edges by TIME/lookback horizon)
with a RECORD-COUNT-based edge split: for each group's sequence of rows (ordered by ``time_col``), aggregate
the first ``n`` records and the last ``n`` records separately, then emit both the raw edge aggregates and
their diff/ratio -- a simple, cheap trend-direction signal ("has this entity's behavior grown or shrunk since
its earliest observed records") distinct from a fixed-lookback trailing window.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

_SUPPORTED_AGGS = {"mean", "sum", "min", "max", "median", "std"}


def windowed_edge_aggregate_diff(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    value_col: str,
    n: int,
    agg: str = "mean",
) -> pd.DataFrame:
    """Per-entity ``agg(first n records)`` vs ``agg(last n records)`` diff and ratio.

    Parameters
    ----------
    df
        Long-format panel: one row per ``(entity_col, time_col, value_col)`` observation.
    entity_col
        Grouping key.
    time_col
        Within-group ordering column.
    value_col
        Column to aggregate.
    n
        Number of records to take from each edge (groups with fewer than ``2*n`` rows use overlapping or
        smaller edge slices -- ``min(n, group_size)`` records from each end; a group with exactly one row
        has an identical first/last slice, giving diff=0/ratio=1, which is the correct "no observed trend"
        answer rather than a crash or NaN).
    agg
        One of ``{"mean", "sum", "min", "max", "median", "std"}``.

    Returns
    -------
    pd.DataFrame
        One row per unique entity (first-seen order), columns ``entity_col``,
        ``{value_col}_first{n}_{agg}``, ``{value_col}_last{n}_{agg}``,
        ``{value_col}_edge_diff_{n}_{agg}`` (last - first), ``{value_col}_edge_ratio_{n}_{agg}`` (last /
        first, ``NaN`` when the first-edge aggregate is exactly 0).
    """
    if agg not in _SUPPORTED_AGGS:
        raise ValueError(f"windowed_edge_aggregate_diff: unsupported agg {agg!r}, expected one of {sorted(_SUPPORTED_AGGS)}")
    if n < 1:
        raise ValueError("windowed_edge_aggregate_diff: n must be >= 1")

    entities = pd.unique(df[entity_col])
    first_vals = np.empty(len(entities), dtype=np.float64)
    last_vals = np.empty(len(entities), dtype=np.float64)

    # Extract the whole-frame numpy arrays ONCE and index by group positions, instead of a per-entity
    # sub-DataFrame + Series.to_numpy() pair -- the latter was the dominant cProfile hotspot (repeated
    # pandas column-access machinery per group: _ixs/_box_col_values/__finalize__), same bug class as the
    # df[df[col]==val] rescan fixed elsewhere this session, just at the column-extraction layer instead.
    time_arr = df[time_col].to_numpy()
    value_arr = df[value_col].to_numpy(dtype=np.float64)
    # A NaN entity is its own group (audit 24-F40): pd.unique lists NaN as an entity, but the default groupby drops it, so a
    # NaN-keyed lookup raised KeyError. Positions come from factorize codes, which number entities in first-seen order and keep
    # NaN as a code, so entity i is code i; a NaN dict key would only match by identity and is avoided altogether.
    codes, _uniques = pd.factorize(df[entity_col], sort=False, use_na_sentinel=False)
    by_code = np.argsort(codes, kind="stable")
    bounds = np.concatenate(([0], np.cumsum(np.bincount(codes, minlength=len(entities)))))
    reducer = getattr(np, agg)

    for i in range(len(entities)):
        positions = by_code[bounds[i] : bounds[i + 1]]
        order = np.argsort(time_arr[positions])
        sorted_vals = value_arr[positions][order]
        k = min(n, len(sorted_vals))
        first_vals[i] = reducer(sorted_vals[:k])
        last_vals[i] = reducer(sorted_vals[-k:])

    diff = last_vals - first_vals
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(first_vals != 0, last_vals / first_vals, np.nan)

    cols: Dict[str, np.ndarray] = {
        entity_col: entities,
        f"{value_col}_first{n}_{agg}": first_vals,
        f"{value_col}_last{n}_{agg}": last_vals,
        f"{value_col}_edge_diff_{n}_{agg}": diff,
        f"{value_col}_edge_ratio_{n}_{agg}": ratio,
    }
    return pd.DataFrame(cols)


__all__ = ["windowed_edge_aggregate_diff"]
