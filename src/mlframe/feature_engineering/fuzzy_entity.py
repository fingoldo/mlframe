"""Group-level "outlier-ness within group" features for imperfect (heuristic) entity linking.

When true entity identity is unavailable and must be approximated via a multi-column heuristic key (e.g.
IEEE-CIS's 5th place team combining D1/D3/addr1/P_emaildomain/ProductID/C13 into a pseudo-user ``uid``), the
linking itself is imperfect — so features that treat the group as ground truth (raw group aggregates) can be
noisy. What DOES help even under imperfect linking: signals about whether THIS row's value looks like the
group's usual pattern or like an outlier — does it match the group's most common value, has this exact value
appeared in the group before, how long since it last did. These degrade gracefully when the linking is
wrong (worst case: uninformative noise) rather than silently corrupting a hard aggregate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def fuzzy_entity_group_features(
    group_ids: np.ndarray,
    values: np.ndarray,
    time_order: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Outlier-within-group features for an approximate ``(group_ids, values)`` pairing.

    Parameters
    ----------
    group_ids
        ``(n,)`` heuristically-constructed entity/group key.
    values
        ``(n,)`` the attribute whose within-group behavior to characterize (e.g. device id, email domain).
    time_order
        Optional ``(n,)`` numeric ordering key (timestamp or a monotonic event index). When given,
        ``value_occurrence_count_in_group`` and ``days_since_value_last_seen_in_group`` are computed
        strictly from PRIOR rows only (leak-safe, causal). Defaults to the input row order when omitted.

    Returns
    -------
    dict[str, np.ndarray]
        ``group_mode_match`` — bool, whether this row's value equals its group's most frequent value
        (computed over the WHOLE group; an exploratory/offline feature, NOT leak-safe for online scoring
        unless the group's mode is itself computed from training-only history).
        ``value_occurrence_count_in_group`` — how many times this exact value was seen in this group
        STRICTLY BEFORE this row (0 = first occurrence = novel value, an outlier signal).
        ``days_since_value_last_seen_in_group`` — gap (in ``time_order`` units) since this value last
        appeared in this group; NaN on first occurrence.
    """
    n = len(group_ids)
    order = np.arange(n, dtype=np.float64) if time_order is None else np.asarray(time_order, dtype=np.float64)

    df = pd.DataFrame({"group": np.asarray(group_ids), "value": np.asarray(values), "order": order})
    df["_orig_idx"] = np.arange(n)

    # Vectorised group-mode: groupby(["group","value"]).size() + idxmax() per group, entirely in pandas'
    # C-level aggregation path. The natural ``groupby("group")["value"].agg(lambda s: s.mode()...)`` calls
    # the (itself non-trivial) ``pandas.Series.mode()`` once PER GROUP via the slow python-callback
    # aggregation path -- profiled at 50k groups: 271s for 5 calls (54s/call), ~99% of wall time in
    # ``_aggregate_series_pure_python`` -> per-group ``Series.mode()``. This version has no per-group
    # Python callback at all.
    value_counts_by_group = df.groupby(["group", "value"], sort=False).size()
    mode_idx = value_counts_by_group.groupby(level="group", sort=False).idxmax()
    mode_per_group = pd.Series({g: v for g, v in mode_idx.to_numpy()})
    group_mode_match = df["value"].to_numpy() == df["group"].map(mode_per_group).to_numpy()

    df_sorted = df.sort_values("order", kind="stable")
    grp = df_sorted.groupby(["group", "value"], sort=False)
    occurrence_count_sorted = grp.cumcount().to_numpy().astype(np.float64)
    gap_sorted = grp["order"].diff().to_numpy()

    orig_idx_sorted = df_sorted["_orig_idx"].to_numpy()
    occurrence_count = np.empty(n, dtype=np.float64)
    gap = np.empty(n, dtype=np.float64)
    occurrence_count[orig_idx_sorted] = occurrence_count_sorted
    gap[orig_idx_sorted] = gap_sorted

    return {
        "group_mode_match": group_mode_match,
        "value_occurrence_count_in_group": occurrence_count,
        "days_since_value_last_seen_in_group": gap,
    }


__all__ = ["fuzzy_entity_group_features"]
