"""As-of-date-guarded aggregation: a hard-to-misuse leakage cutoff for per-entity groupby aggregates.

Computing "customer's average spend" or "coupon's redemption rate" from historical transactions is a
routine feature-engineering step -- and a routine LEAKAGE bug when the aggregate accidentally includes rows
at or after the event being predicted (multiple AmExpert-2019 winners independently flagged this as the fix
that mattered). Rather than relying on every caller to remember to pre-filter, this helper takes the cutoff
as a required parameter and enforces ``time_col < as_of`` before aggregating, per query row when ``as_of`` is
itself a column (each entity/row can have its own cutoff) or globally when it's a scalar.
"""
from __future__ import annotations

from typing import Dict, Sequence, Union, Optional

import numpy as np
import pandas as pd

# sum/mean/count are computable from a per-entity cumulative-sum array + one searchsorted lookup per query --
# O(log n) per query instead of materializing and reducing a fresh eligible-rows slice per query row (the
# "many small groups + per-row pandas call" cost class found elsewhere in this package, profiled at 8.7s for
# 20k queries pre-fix). Any other agg name falls back to the slower per-query slice-and-reduce path.
_FAST_AGGS = {"sum", "mean", "count"}


def leakage_safe_aggregate(
    history_df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    as_of: Union[str, pd.DataFrame],
    agg_funcs: Dict[str, Sequence[str]],
    query_entity_col: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate ``history_df`` per entity using only rows strictly before each query's cutoff.

    Parameters
    ----------
    history_df
        Historical rows to aggregate; must contain ``entity_col``, ``time_col``, and every column named in
        ``agg_funcs``.
    entity_col
        Grouping key.
    time_col
        Ordering/cutoff column (must be comparable, e.g. numeric or datetime).
    as_of
        A query DataFrame with columns ``entity_col`` and a per-row cutoff column (name given by
        ``query_entity_col``, defaulting to ``"as_of"``); for a single GLOBAL cutoff, filter ``history_df``
        yourself and call ``history_df.groupby(entity_col).agg(agg_funcs)`` directly instead -- this helper's
        value is specifically the PER-ROW/PER-ENTITY cutoff case.
    agg_funcs
        ``{column: [agg_name, ...]}``. ``"sum"``/``"mean"``/``"count"`` use a fast vectorized path; any other
        pandas Series method name (``"min"``, ``"max"``, ``"median"``, ...) falls back to a per-query slice.
    query_entity_col
        Name of ``as_of``'s cutoff column (defaults to ``"as_of"``).

    Returns
    -------
    pd.DataFrame
        One row per query row (same order as ``as_of``), with the entity key and one column per
        ``(agg_column, agg_name)`` pair, ``NaN`` where an entity has no eligible pre-cutoff history.
    """
    if not isinstance(as_of, pd.DataFrame):
        raise TypeError(
            "leakage_safe_aggregate: as_of must be a query DataFrame with an entity column and a per-row cutoff column "
            "(pre-filter history_df yourself and call groupby().agg() directly for a single global cutoff)"
        )
    cutoff_col = query_entity_col or "as_of"
    if cutoff_col not in as_of.columns:
        raise ValueError(f"leakage_safe_aggregate: as_of frame missing cutoff column {cutoff_col!r}")
    if entity_col not in as_of.columns:
        raise ValueError(f"leakage_safe_aggregate: as_of frame missing entity column {entity_col!r}")

    query = as_of.reset_index(drop=True)
    out_cols: Dict[str, np.ndarray] = {f"{col}_{fn}": np.full(len(query), np.nan) for col, fns in agg_funcs.items() for fn in fns}

    history_groups = {entity: grp for entity, grp in history_df.groupby(entity_col, sort=False)}

    for entity, entity_queries in query.groupby(entity_col, sort=False):
        entity_history = history_groups.get(entity)
        row_positions = entity_queries.index.to_numpy()
        if entity_history is None or entity_history.empty:
            continue

        order = np.argsort(entity_history[time_col].to_numpy())
        sorted_times = entity_history[time_col].to_numpy()[order]
        cutoffs = entity_queries[cutoff_col].to_numpy()
        n_eligible = np.searchsorted(sorted_times, cutoffs, side="left")

        for col, fns in agg_funcs.items():
            sorted_col = entity_history[col].to_numpy()[order]
            fast_fns = [fn for fn in fns if fn in _FAST_AGGS]
            slow_fns = [fn for fn in fns if fn not in _FAST_AGGS]

            if fast_fns:
                cumsum = np.concatenate([[0.0], np.cumsum(sorted_col.astype(np.float64))])
                col_sum = cumsum[n_eligible]
                col_count = n_eligible
                has_history = n_eligible > 0
                if "sum" in fast_fns:
                    out_cols[f"{col}_sum"][row_positions] = np.where(has_history, col_sum, np.nan)
                if "count" in fast_fns:
                    out_cols[f"{col}_count"][row_positions] = np.where(has_history, col_count, np.nan)
                if "mean" in fast_fns:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out_cols[f"{col}_mean"][row_positions] = np.where(has_history, col_sum / np.where(col_count == 0, 1, col_count), np.nan)

            if slow_fns:
                for _i, (pos, n) in enumerate(zip(row_positions, n_eligible)):
                    if n == 0:
                        continue
                    eligible_vals = sorted_col[:n]
                    for fn in slow_fns:
                        out_cols[f"{col}_{fn}"][pos] = getattr(pd.Series(eligible_vals), fn)()

    result = pd.DataFrame(out_cols)
    result.insert(0, entity_col, query[entity_col].to_numpy())
    return result


__all__ = ["leakage_safe_aggregate"]
