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


def _cluster_fuzzy_keys(group_ids: np.ndarray, max_distance: int, block_prefix_len: int) -> np.ndarray:
    """Union-find near-duplicate clustering of string keys, blocked by shared prefix to stay sub-quadratic.

    A heuristic key built from noisy real-world identifiers (typos, OCR errors, inconsistent formatting)
    under-groups an entity into several exact-match keys that differ by 1-2 characters. Full pairwise edit
    distance is O(n^2) and infeasible past a few thousand unique keys, so candidates are first bucketed by
    a shared prefix (``block_prefix_len`` chars) — genuinely different entities essentially never share a
    long prefix by chance, while typo'd variants of the same key usually do — and only within-block pairs
    pay the ``rapidfuzz`` Levenshtein cost. ``score_cutoff=max_distance`` lets rapidfuzz short-circuit each
    comparison instead of computing the exact distance for pairs already known to be too far apart.

    Returns a ``(n,)`` array of canonical cluster labels (one representative original key per cluster),
    which the caller uses in place of the raw ``group_ids`` for the exact-match grouping logic below.
    """
    from rapidfuzz.distance import Levenshtein  # lazy: only importable/needed when fuzzy_key_matching=True

    str_keys = np.asarray(group_ids, dtype=str)
    unique_keys = np.unique(str_keys)

    parent = {k: k for k in unique_keys}

    def find(k: str) -> str:
        """Return the canonical cluster representative for key k, path-compressing along the way."""
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    blocks: dict[str, list[str]] = {}
    for k in unique_keys:
        blocks.setdefault(k[:block_prefix_len], []).append(k)

    for block_keys in blocks.values():
        for i in range(len(block_keys)):
            a = block_keys[i]
            for j in range(i + 1, len(block_keys)):
                b = block_keys[j]
                if Levenshtein.distance(a, b, score_cutoff=max_distance) <= max_distance:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb

    cluster_of = {k: find(k) for k in unique_keys}
    return np.array([cluster_of[k] for k in str_keys], dtype=object)


def fuzzy_entity_group_features(
    group_ids: np.ndarray,
    values: np.ndarray,
    time_order: Optional[np.ndarray] = None,
    fuzzy_key_matching: bool = False,
    fuzzy_max_distance: int = 1,
    fuzzy_block_prefix_len: int = 4,
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
    fuzzy_key_matching
        Opt-in (default ``False`` = unchanged exact-key-match behavior). When ``True``, ``group_ids`` are
        first coerced to strings and near-duplicate keys (edit distance <= ``fuzzy_max_distance``) are
        merged into one cluster BEFORE the exact-match grouping logic runs, so noisy identifiers that
        differ by typos/formatting (e.g. ``"cust_00042"`` vs ``"cust_00043"`` vs ``"cust_0OO42"``) still
        link to the same group instead of silently splitting one entity into several under-populated ones.
    fuzzy_max_distance
        Maximum Levenshtein edit distance for two keys to merge. Only used when ``fuzzy_key_matching=True``.
    fuzzy_block_prefix_len
        Blocking-key length (shared-prefix chars) used to avoid full O(n^2) pairwise comparison; only keys
        sharing this prefix are ever compared. Only used when ``fuzzy_key_matching=True`` — pick a length
        that covers the part of the key least likely to contain the typo (e.g. a fixed entity-type prefix).

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

    group_arr = np.asarray(group_ids)
    if fuzzy_key_matching:
        group_arr = _cluster_fuzzy_keys(group_arr, max_distance=fuzzy_max_distance, block_prefix_len=fuzzy_block_prefix_len)

    df = pd.DataFrame({"group": group_arr, "value": np.asarray(values), "order": order})
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
