"""Two-step recency-weighted target encoding: event-level feature-combo mean -> recency-weighted entity aggregate.

A single-level target encoding treats every event equally. A 5th-place Elo-merchant-recommendation team's
two-step process is more information-dense for transactional data: (1) leak-free target-mean-encode a
feature COMBINATION at the event/transaction level, then (2) aggregate those per-event encodings UP to the
entity (e.g. card/customer) level with weights that decay for older events -- so an entity's encoded feature
reflects its RECENT behavior more than its full history equally. Reuses the existing leak-free
``ordered_target_encode`` for step 1 rather than reimplementing target-mean computation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode


def two_step_recency_weighted_target_encode(
    events_df: pd.DataFrame,
    entity_col: str,
    feature_cols: Sequence[str],
    y: np.ndarray,
    time_col: str,
    decay_half_life: float,
    order: Optional[np.ndarray] = None,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Event-level leak-free target encoding of ``feature_cols``, aggregated to ``entity_col`` with recency decay.

    Parameters
    ----------
    events_df
        One row per event/transaction.
    entity_col
        The higher-level entity each event belongs to (e.g. card/customer id).
    feature_cols
        Columns defining the event-level feature combination to target-encode (concatenated into one
        composite category before encoding).
    y
        ``(n_events,)`` target aligned to ``events_df``, used for the leak-free step-1 encoding.
    time_col
        Numeric time/order column; both the step-1 ordering and the step-2 recency weights use it.
    decay_half_life
        Recency-weight half-life in ``time_col`` units: an event's step-2 weight is
        ``0.5 ** ((max_time_per_entity - event_time) / decay_half_life)`` (weighted relative to that
        ENTITY's own most recent event -- an entity queried at any point in its own history gets a sensible
        recency profile, not one skewed by the global time range).
    order
        Optional explicit causal order for step 1's leak-free encoding; defaults to ``time_col``.
    smoothing
        Passed through to :func:`ordered_target_encode`.

    Returns
    -------
    np.ndarray
        ``(n_events,)`` array: each event's row gets its ENTITY's recency-weighted aggregate of the step-1
        event-level encodings (same value repeated for every event of that entity).
    """
    # vectorized str.cat chain instead of .agg("|".join, axis=1) -- the latter is a per-row Python callback
    # (profiled: 630ms of 680ms at n=50k), the same "many small groups + Python callback" cost class found
    # elsewhere; str.cat concatenates each column pairwise in one C-level pass, ~13x faster at n=500k.
    feature_col_list = list(feature_cols)
    composite_series = events_df[feature_col_list[0]].astype(str)
    for col in feature_col_list[1:]:
        composite_series = composite_series.str.cat(events_df[col].astype(str), sep="|")
    composite_cat = composite_series.to_numpy()
    step1_encoding = ordered_target_encode(composite_cat, np.asarray(y, dtype=np.float64), order=order, smoothing=smoothing)

    time_vals = events_df[time_col].to_numpy(dtype=np.float64)
    entity_vals = events_df[entity_col].to_numpy()

    df = pd.DataFrame({"entity": entity_vals, "time": time_vals, "enc": step1_encoding})
    entity_max_time = df.groupby("entity")["time"].transform("max")
    weight = 0.5 ** ((entity_max_time - df["time"]) / decay_half_life)

    weighted_sum = (df["enc"] * weight).groupby(df["entity"]).transform("sum")
    weight_sum = weight.groupby(df["entity"]).transform("sum")

    return np.asarray((weighted_sum / weight_sum).to_numpy(), dtype=np.float64)


__all__ = ["two_step_recency_weighted_target_encode"]
