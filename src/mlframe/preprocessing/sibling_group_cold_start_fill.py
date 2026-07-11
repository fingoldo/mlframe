"""``sibling_group_cold_start_fill``: fall back to the nearest ordered sibling group's last value.

Source: dd_2nd_power-laws-forecasting.md, Future Steps -- "when a ForecastId has entirely missing values,
fall back to the last available value from the previous ForecastId rather than simple mean imputation." A
group with ZERO history offers no within-group statistic to impute from; mlframe's existing
``regime_conditioned_median_fill`` falls back to the GLOBAL median for such groups, discarding any
information about where that group sits in a meaningful sibling ordering (e.g. sequential ForecastIds sharing
a regime). Borrowing the nearest PRECEDING sibling group's last known value is a much closer cold-start proxy
whenever such an ordering exists and groups are locally similar.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def sibling_group_cold_start_fill(df: pd.DataFrame, group_col: str, order_col: str, value_col: str, fallback_value: Optional[float] = None) -> pd.Series:
    """Per-row value: the row's own group's last known value, or (if the whole group is missing) the
    nearest PRECEDING sibling group's last known value, ordered by ``order_col``.

    Parameters
    ----------
    df
        Source frame.
    group_col
        Column identifying the group (e.g. a ForecastId).
    order_col
        Column giving a sortable ordering across DISTINCT groups (e.g. a group-level sequence id or the
        group's first-seen timestamp) -- groups are visited in ascending order of this column.
    value_col
        The value column to fill.
    fallback_value
        Used for groups with no preceding non-empty sibling at all (e.g. the very first group). Defaults to
        the overall non-null mean of ``value_col``.

    Returns
    -------
    pd.Series
        One value per row: the row's group's last known value if the group has ANY history, else the
        nearest preceding sibling group's last known value.
    """
    group_order = df.groupby(group_col, sort=False)[order_col].first().sort_values()
    ordered_groups = group_order.index.to_numpy()

    # pandas' native groupby `.last()` already returns each group's last NON-NULL value (NaN only if the
    # WHOLE group is null) via a single vectorized C-level pass -- the prior `.apply(lambda s: ...)` ran one
    # Python callback PER GROUP (measured as the dominant cost: 457s cProfile / 166s wall at 50000 groups x20
    # calls), the classic per-group-Python-callback anti-pattern this codebase avoids elsewhere.
    last_known_per_group = df.groupby(group_col, sort=False)[value_col].last()
    last_known_per_group = last_known_per_group.reindex(ordered_groups)

    # forward-fill across groups in sibling order: an entirely-missing group borrows the nearest PRECEDING
    # group's last known value.
    global_fallback = float(fallback_value) if fallback_value is not None else float(df[value_col].dropna().mean())
    filled_per_group = last_known_per_group.ffill().fillna(global_fallback)

    return df[group_col].map(filled_per_group)


__all__ = ["sibling_group_cold_start_fill"]
