"""Ranking metrics for Learning-to-Rank target type.

Per-query NDCG@k, MAP@k, MRR -- aggregated across queries via mean (queries
with zero positive items are dropped from the mean, NaN-bound by the metric
formula). Mirrors the upfront-filter pattern from
``metrics.py::fast_aucs_per_group_optimized`` so degenerate single-item /
single-class groups don't waste the inner loop.

Public API:
    - ``ndcg_at_k(y_true, y_score, group_ids, k=10)`` -- mean NDCG@k
    - ``map_at_k(y_true, y_score, group_ids, k=10)`` -- mean MAP@k
    - ``mrr(y_true, y_score, group_ids)`` -- mean reciprocal rank
    - ``compute_ranking_summary(y_true, y_score, group_ids, eval_at)`` --
      dict with NDCG/MAP/MRR for each k in ``eval_at``.

All metrics expect:
    - ``y_true`` : (N,) graded relevance (int >= 0) or binary (0/1)
    - ``y_score`` : (N,) per-row predicted score (higher = more relevant)
    - ``group_ids`` : (N,) per-row query identifier (any hashable, not necessarily contiguous)

Gain formula choice (NDCG): this module uses the **exponential** gain
``(2^rel - 1) / log2(pos + 2)`` (Burges 2005, used internally by
LightGBM / CatBoost / XGBoost ``eval_metric='ndcg'``). sklearn's
``ndcg_score`` uses **linear** gain ``rel / log2(pos + 2)`` instead.
For binary ``y in {0, 1}`` the two formulas coincide (since
``2^1 - 1 == 1``), so binary-relevance tests against sklearn match
exactly. For graded relevance, results differ -- ours match what the
GBDT ranker internals optimise for, which is the point of mlframe.

Numba kernels follow ``mlframe.metrics.core::NUMBA_NJIT_PARAMS`` for cache
consistency: ``fastmath=False, cache=True, nogil=True``.

References:
    - Burges et al. 2005 (LambdaMART, exponential gain definition).
    - TREC standard MRR (first-relevant-rank reciprocal).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numba
from numba import prange
import numpy as np

# Reuse the project-wide numba flag tuple so prewarm + signature checks see one source.
NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


# ----------------------------------------------------------------------------------
# Per-query kernels (operate on a single query's slice of y_true / y_score)
# ----------------------------------------------------------------------------------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _dcg_at_k(rels_sorted_by_score: np.ndarray, k: int) -> float:
    """Discounted cumulative gain at k.

    ``rels_sorted_by_score`` -- per-item relevance, ordered by predicted
    score descending (so position 0 is the most-confident prediction).
    Uses the standard ``(2^rel - 1) / log2(pos + 2)`` gain formula
    (Burges et al. 2005), which subsumes binary relevance as the
    ``2^1 - 1 = 1`` case.

    Stops at ``min(k, len)`` items.
    """
    n = rels_sorted_by_score.shape[0]
    limit = k if k < n else n
    dcg = 0.0
    for i in range(limit):
        rel = rels_sorted_by_score[i]
        if rel <= 0:
            continue
        # 2^rel - 1; for binary {0,1} rel=1 -> gain=1 (matches sklearn binary NDCG).
        gain = (2.0 ** rel) - 1.0
        # Position discount: log2(i+2) so first item has discount log2(2)=1.
        dcg += gain / np.log2(i + 2.0)
    return dcg


@numba.njit(**NUMBA_NJIT_PARAMS)
def _ndcg_one_query(y_true_q: np.ndarray, y_score_q: np.ndarray, k: int) -> float:
    """NDCG@k for a single query.

    Returns NaN when the query has zero positive relevance (IDCG=0 is
    undefined; downstream caller drops NaN before mean-aggregation).
    """
    n = y_true_q.shape[0]
    if n == 0:
        return np.nan
    # Order by score descending. argsort -> ascending; flip for desc.
    order = np.argsort(-y_score_q, kind="mergesort")  # Wave 57: stable for tie-determinism
    rels_pred_order = y_true_q[order]
    # Ideal: best possible ordering is true rels sorted descending.
    # numba's @njit np.sort doesn't accept ``kind=`` (only np.argsort does)
    # - the Wave 57 stable kwarg here broke compilation. Stable is irrelevant
    # on np.sort's VALUES (all tied positions hold the same value); the
    # determinism comes from argsort above, not from this np.sort.
    rels_ideal = -np.sort(-y_true_q)
    idcg = _dcg_at_k(rels_ideal, k)
    if idcg <= 0.0:
        return np.nan
    dcg = _dcg_at_k(rels_pred_order, k)
    return dcg / idcg


@numba.njit(**NUMBA_NJIT_PARAMS)
def _map_one_query(y_true_q: np.ndarray, y_score_q: np.ndarray, k: int) -> float:
    """Mean Average Precision @ k for a single query.

    Binary-relevance MAP: any ``y_true > 0`` is counted as relevant
    (matches sklearn's ``average_precision_score`` interpretation when
    relevance is graded). Returns NaN when no relevant items exist.
    """
    n = y_true_q.shape[0]
    if n == 0:
        return np.nan
    order = np.argsort(-y_score_q, kind="mergesort")  # Wave 57: stable for tie-determinism
    rels_pred_order = y_true_q[order]
    limit = k if k < n else n

    n_relevant_total = 0
    for i in range(n):
        if y_true_q[i] > 0:
            n_relevant_total += 1
    if n_relevant_total == 0:
        return np.nan

    n_hits = 0
    sum_precisions = 0.0
    for i in range(limit):
        if rels_pred_order[i] > 0:
            n_hits += 1
            sum_precisions += n_hits / (i + 1.0)
    # Normalise by min(k, n_relevant_total) -- standard MAP@k convention
    # (Manning & Raghavan IR textbook). When n_relevant_total < k the
    # max attainable AP@k is 1.0; using min as denominator preserves that.
    denom = min(k, n_relevant_total)
    return sum_precisions / denom


@numba.njit(**NUMBA_NJIT_PARAMS)
def _mrr_one_query(y_true_q: np.ndarray, y_score_q: np.ndarray) -> float:
    """Mean Reciprocal Rank for a single query.

    Reciprocal of the rank (1-indexed) of the first relevant item in the
    score-descending ordering. NaN when no relevant items.
    """
    n = y_true_q.shape[0]
    if n == 0:
        return np.nan
    order = np.argsort(-y_score_q, kind="mergesort")  # Wave 57: stable for tie-determinism
    for i in range(n):
        if y_true_q[order[i]] > 0:
            return 1.0 / (i + 1.0)
    return np.nan


# ----------------------------------------------------------------------------------
# Group iteration helper
# ----------------------------------------------------------------------------------


def _iter_group_slices(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by group_ids once, return (sorted_y_true, sorted_y_score, group_starts).

    ``group_starts`` is an array of length ``n_unique_groups + 1`` such
    that ``slice(group_starts[i], group_starts[i+1])`` indexes the i-th
    group's rows. The last entry equals ``len(y_true)``.

    A single sort + boundary-detection is O(N log N + N) -- much cheaper
    than per-group ``y_true[group_ids == g]`` (which is O(N) per group,
    O(N * n_groups) total).
    """
    if len(group_ids) != len(y_true) or len(group_ids) != len(y_score):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} y_score={len(y_score)} "
            f"group_ids={len(group_ids)}"
        )
    if len(group_ids) == 0:
        return (
            np.asarray(y_true, dtype=np.float64),
            np.asarray(y_score, dtype=np.float64),
            np.array([0], dtype=np.intp),
        )

    sort_idx = np.argsort(group_ids, kind="mergesort")
    sorted_groups = group_ids[sort_idx]
    sorted_y_true = np.asarray(y_true, dtype=np.float64)[sort_idx]
    sorted_y_score = np.asarray(y_score, dtype=np.float64)[sort_idx]

    # Boundaries: indices where the group_id changes + start (0) + end (N).
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    group_starts = np.concatenate(([0], boundaries, [len(sorted_groups)])).astype(np.intp)

    return sorted_y_true, sorted_y_score, group_starts


# ----------------------------------------------------------------------------------
# Public mean-over-queries API
# ----------------------------------------------------------------------------------


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    k: int = 10,
) -> float:
    """Mean NDCG@k across queries.

    Returns NaN if every query is degenerate (zero positives or empty).
    Otherwise returns the mean NDCG@k over queries with IDCG > 0.
    """
    sorted_y_true, sorted_y_score, group_starts = _iter_group_slices(y_true, y_score, group_ids)
    n_groups = len(group_starts) - 1
    if n_groups == 0:
        return float("nan")
    accum = 0.0
    n_valid = 0
    for i in range(n_groups):
        s, e = group_starts[i], group_starts[i + 1]
        v = _ndcg_one_query(sorted_y_true[s:e], sorted_y_score[s:e], k)
        if not np.isnan(v):
            accum += v
            n_valid += 1
    if n_valid == 0:
        return float("nan")
    return accum / n_valid


def map_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    k: int = 10,
) -> float:
    """Mean MAP@k across queries (binary relevance: any y > 0 is relevant)."""
    sorted_y_true, sorted_y_score, group_starts = _iter_group_slices(y_true, y_score, group_ids)
    n_groups = len(group_starts) - 1
    if n_groups == 0:
        return float("nan")
    accum = 0.0
    n_valid = 0
    for i in range(n_groups):
        s, e = group_starts[i], group_starts[i + 1]
        v = _map_one_query(sorted_y_true[s:e], sorted_y_score[s:e], k)
        if not np.isnan(v):
            accum += v
            n_valid += 1
    if n_valid == 0:
        return float("nan")
    return accum / n_valid


def mrr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
) -> float:
    """Mean Reciprocal Rank across queries.

    Computed client-side because XGBoost has no native MRR -- exposing
    it here means the CB / XGB / LGB suite can report MRR uniformly.
    """
    sorted_y_true, sorted_y_score, group_starts = _iter_group_slices(y_true, y_score, group_ids)
    n_groups = len(group_starts) - 1
    if n_groups == 0:
        return float("nan")
    accum = 0.0
    n_valid = 0
    for i in range(n_groups):
        s, e = group_starts[i], group_starts[i + 1]
        v = _mrr_one_query(sorted_y_true[s:e], sorted_y_score[s:e])
        if not np.isnan(v):
            accum += v
            n_valid += 1
    if n_valid == 0:
        return float("nan")
    return accum / n_valid


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _summary_batched_kernel(
    sorted_y_true: np.ndarray,
    sorted_y_score: np.ndarray,
    group_starts: np.ndarray,
    eval_ks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """One-shot per-group pass: NDCG@k, MAP@k for every k in ``eval_ks``,
    plus k-free MRR, accumulated across all groups in a single numba
    kernel.

    Returns:
        ndcg_sums : (K,) per-k sum of NDCG@k over valid groups
        ndcg_counts : (K,) per-k count of valid groups
        map_sums : (K,) per-k sum of MAP@k over valid groups
        map_counts : (K,) per-k count of valid groups
        mrr_sum : scalar sum of MRR over valid groups
        mrr_count : scalar count of valid groups

    Caller divides sums by counts to get means.

    Rationale: the previous compute_ranking_summary called the per-query
    numba kernels 7 * n_groups times (NDCG@1/5/10 + MAP@1/5/10 + MRR).
    On 1M-row LTR with 333k groups that's 2.33M Python->numba dispatch
    transitions per call -- ~6 s wall time on c0001's profile. This
    single-kernel form does the same work in one entry-and-exit.
    Inner per-query logic is bit-exact equivalent to
    ``_ndcg_one_query`` / ``_map_one_query`` / ``_mrr_one_query``.
    """
    n_groups = len(group_starts) - 1
    K = len(eval_ks)

    # prange parallelises across groups; each iteration writes its own
    # row into per-group temp buffers, then we reduce afterwards.
    # Numba's prange supports +reduction natively on scalar accumulators,
    # but we have K-vector reductions which need explicit per-thread
    # buffering. Easiest: materialise per-group ndcg / map / mrr arrays
    # and reduce in a second (vectorised, no Python overhead) pass.
    ndcg_per_group = np.full((n_groups, K), np.nan, dtype=np.float64)
    map_per_group = np.full((n_groups, K), np.nan, dtype=np.float64)
    mrr_per_group = np.full(n_groups, np.nan, dtype=np.float64)

    for i in prange(n_groups):
        s = group_starts[i]
        e = group_starts[i + 1]
        n = e - s
        if n == 0:
            continue

        # Sort group's items by score descending for both NDCG and MAP /
        # MRR; reused across all per-k metrics for this group.
        y_t = sorted_y_true[s:e]
        y_sc = sorted_y_score[s:e]
        # Wave 57 (2026-05-20): stable sort so tied scores don't shift NDCG/MAP/MRR
        # across runs.
        order = np.argsort(-y_sc, kind="mergesort")
        rels_pred = y_t[order]
        # numba's @njit np.sort doesn't accept ``kind=`` (only np.argsort does).
        # Stable sort is irrelevant on np.sort's VALUES anyway - the per-tie
        # output positions all hold the same value, so default-quicksort and
        # stable-mergesort produce identical arrays here. Wave 57's stable
        # arg on this line was a copy-paste from line 339 (argsort) where it
        # genuinely matters for tie-breaking indices.
        rels_ideal = -np.sort(-y_t)

        # Count positives once (shared by MAP, MRR, and idcg-degeneracy).
        n_rel_total = 0
        for j in range(n):
            if y_t[j] > 0:
                n_rel_total += 1

        # MRR: first relevant rank in score-descending order.
        if n_rel_total > 0:
            for j in range(n):
                if rels_pred[j] > 0:
                    mrr_per_group[i] = 1.0 / (j + 1.0)
                    break

        # Per-k NDCG / MAP.
        for kj in range(K):
            k = eval_ks[kj]
            limit = k if k < n else n

            # IDCG@k -- if 0 the per-k NDCG is undefined (NaN sentinel).
            idcg = 0.0
            for j in range(limit):
                rel = rels_ideal[j]
                if rel > 0:
                    idcg += ((2.0 ** rel) - 1.0) / np.log2(j + 2.0)
            if idcg > 0.0:
                dcg = 0.0
                for j in range(limit):
                    rel = rels_pred[j]
                    if rel > 0:
                        dcg += ((2.0 ** rel) - 1.0) / np.log2(j + 2.0)
                ndcg_per_group[i, kj] = dcg / idcg

            # MAP@k -- needs at least one positive globally.
            if n_rel_total > 0:
                n_hits = 0
                sum_prec = 0.0
                for j in range(limit):
                    if rels_pred[j] > 0:
                        n_hits += 1
                        sum_prec += n_hits / (j + 1.0)
                denom = k if k < n_rel_total else n_rel_total
                map_per_group[i, kj] = sum_prec / denom

    # Reduce: NaN-aware mean per metric.
    ndcg_sums = np.zeros(K, dtype=np.float64)
    ndcg_counts = np.zeros(K, dtype=np.int64)
    map_sums = np.zeros(K, dtype=np.float64)
    map_counts = np.zeros(K, dtype=np.int64)
    for kj in range(K):
        for i in range(n_groups):
            v = ndcg_per_group[i, kj]
            if not np.isnan(v):
                ndcg_sums[kj] += v
                ndcg_counts[kj] += 1
            v = map_per_group[i, kj]
            if not np.isnan(v):
                map_sums[kj] += v
                map_counts[kj] += 1

    mrr_sum = 0.0
    mrr_count = 0
    for i in range(n_groups):
        v = mrr_per_group[i]
        if not np.isnan(v):
            mrr_sum += v
            mrr_count += 1

    return ndcg_sums, ndcg_counts, map_sums, map_counts, mrr_sum, mrr_count


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _per_query_ndcg_kernel(
    sorted_y_true: np.ndarray,
    sorted_y_score: np.ndarray,
    group_starts: np.ndarray,
    k: int,
) -> np.ndarray:
    """Per-group NDCG@k over the sorted-groups layout from ``_iter_group_slices``.

    Returns an (n_groups,) float64 array; NaN marks degenerate groups (empty or IDCG=0).
    One batched dispatch replaces n_groups Python->numba transitions; each prange iteration writes only its own slot so the parallel pass is race-free.
    """
    n_groups = len(group_starts) - 1
    out = np.full(n_groups, np.nan, dtype=np.float64)
    for i in prange(n_groups):
        s = group_starts[i]
        e = group_starts[i + 1]
        if e > s:
            out[i] = _ndcg_one_query(sorted_y_true[s:e], sorted_y_score[s:e], k)
    return out


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _per_query_mrr_kernel(
    sorted_y_true: np.ndarray,
    sorted_y_score: np.ndarray,
    group_starts: np.ndarray,
) -> np.ndarray:
    """Per-group reciprocal rank over the sorted-groups layout from ``_iter_group_slices``.

    Returns an (n_groups,) float64 array; NaN marks groups with no relevant item (callers choose their own no-hit convention, e.g. 0.0 for the MRR histogram).
    """
    n_groups = len(group_starts) - 1
    out = np.full(n_groups, np.nan, dtype=np.float64)
    for i in prange(n_groups):
        s = group_starts[i]
        e = group_starts[i + 1]
        if e > s:
            out[i] = _mrr_one_query(sorted_y_true[s:e], sorted_y_score[s:e])
    return out


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _lift_curve_kernel(
    sorted_y_true: np.ndarray,
    sorted_y_score: np.ndarray,
    group_starts: np.ndarray,
    max_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative-relevance lift accumulators per rank position, over the sorted-groups layout.

    For each rank position kpos in 0..max_k-1 and each group long enough to reach it:
    cum_rel@kpos / ideal_cum_rel@kpos is added into ``lift_sums[kpos]`` (skipped while the ideal cumulative is still 0) and ``counts[kpos]`` is incremented.
    Caller divides sums by counts to get the mean lift curve.

    Groups are processed in fixed chunks with per-chunk accumulators because element-wise ``+=`` on a shared array inside prange would race; the final reduce is serial and deterministic.
    """
    n_groups = len(group_starts) - 1
    n_chunks = 64 if n_groups > 64 else (n_groups if n_groups > 0 else 1)
    lift_chunks = np.zeros((n_chunks, max_k), dtype=np.float64)
    count_chunks = np.zeros((n_chunks, max_k), dtype=np.int64)
    for c in prange(n_chunks):
        g_lo = c * n_groups // n_chunks
        g_hi = (c + 1) * n_groups // n_chunks
        for i in range(g_lo, g_hi):
            s = group_starts[i]
            e = group_starts[i + 1]
            n = e - s
            if n <= 0:
                continue
            y_t = sorted_y_true[s:e]
            y_sc = sorted_y_score[s:e]
            order = np.argsort(-y_sc, kind="mergesort")
            rels_ideal = -np.sort(-y_t)
            limit = n if n < max_k else max_k
            cum = 0.0
            ideal_cum = 0.0
            for kpos in range(limit):
                cum += y_t[order[kpos]]
                ideal_cum += rels_ideal[kpos]
                if ideal_cum > 0.0:
                    lift_chunks[c, kpos] += cum / ideal_cum
                    count_chunks[c, kpos] += 1
    lift_sums = np.zeros(max_k, dtype=np.float64)
    counts = np.zeros(max_k, dtype=np.int64)
    for c in range(n_chunks):
        for kpos in range(max_k):
            lift_sums[kpos] += lift_chunks[c, kpos]
            counts[kpos] += count_chunks[c, kpos]
    return lift_sums, counts


def compute_ranking_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    eval_at: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Convenience: NDCG@k / MAP@k for each k in ``eval_at`` plus MRR.

    Returned dict keys: ``ndcg@1``, ``ndcg@5``, ``ndcg@10``, ``map@1``,
    ``map@5``, ``map@10``, ``mrr``. Single sort by group_ids reused for
    all metrics; one batched numba kernel handles all per-group work in
    a single dispatch (vs the prior form's 7 * n_groups Python->numba
    transitions, which dominated wall-time on n_groups >= 10k LTR
    inputs -- c0001's profile attributed 6 s to this on 333k groups).
    """
    sorted_y_true, sorted_y_score, group_starts = _iter_group_slices(y_true, y_score, group_ids)
    n_groups = len(group_starts) - 1
    out: Dict[str, float] = {}

    for k in eval_at:
        out[f"ndcg@{k}"] = float("nan")
        out[f"map@{k}"] = float("nan")
    out["mrr"] = float("nan")
    if n_groups == 0:
        return out

    eval_ks = np.asarray(list(eval_at), dtype=np.int64)
    ndcg_sums, ndcg_counts, map_sums, map_counts, mrr_sum, mrr_count = (
        _summary_batched_kernel(sorted_y_true, sorted_y_score, group_starts, eval_ks)
    )

    for kj, k in enumerate(eval_at):
        out[f"ndcg@{k}"] = (
            ndcg_sums[kj] / ndcg_counts[kj] if ndcg_counts[kj] > 0 else float("nan")
        )
        out[f"map@{k}"] = (
            map_sums[kj] / map_counts[kj] if map_counts[kj] > 0 else float("nan")
        )
    out["mrr"] = mrr_sum / mrr_count if mrr_count > 0 else float("nan")
    return out


__all__ = [
    "ndcg_at_k",
    "map_at_k",
    "mrr",
    "compute_ranking_summary",
]
