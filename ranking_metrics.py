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

Numba kernels follow ``mlframe.metrics::NUMBA_NJIT_PARAMS`` for cache
consistency: ``fastmath=False, cache=True, nogil=True``.

References:
    - Burges et al. 2005 (LambdaMART, exponential gain definition).
    - TREC standard MRR (first-relevant-rank reciprocal).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numba
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
    order = np.argsort(-y_score_q)
    rels_pred_order = y_true_q[order]
    # Ideal: best possible ordering is true rels sorted descending.
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
    order = np.argsort(-y_score_q)
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
    order = np.argsort(-y_score_q)
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

    sort_idx = np.argsort(group_ids, kind="stable")
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


def compute_ranking_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    eval_at: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Convenience: NDCG@k / MAP@k for each k in ``eval_at`` plus MRR.

    Returned dict keys: ``ndcg@1``, ``ndcg@5``, ``ndcg@10``, ``map@1``,
    ``map@5``, ``map@10``, ``mrr``. Single sort by group_ids reused for
    all metrics (the cost is in the sort, not the per-metric loops).
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

    # Compute all metrics in one pass (avoids three separate group-loop
    # traversals when callers want the full summary).
    for label_k, kernel in (
        ("ndcg", _ndcg_one_query),
        ("map", _map_one_query),
    ):
        for k in eval_at:
            accum = 0.0
            n_valid = 0
            for i in range(n_groups):
                s, e = group_starts[i], group_starts[i + 1]
                v = kernel(sorted_y_true[s:e], sorted_y_score[s:e], k)
                if not np.isnan(v):
                    accum += v
                    n_valid += 1
            out[f"{label_k}@{k}"] = accum / n_valid if n_valid > 0 else float("nan")

    # MRR is k-free.
    accum = 0.0
    n_valid = 0
    for i in range(n_groups):
        s, e = group_starts[i], group_starts[i + 1]
        v = _mrr_one_query(sorted_y_true[s:e], sorted_y_score[s:e])
        if not np.isnan(v):
            accum += v
            n_valid += 1
    out["mrr"] = accum / n_valid if n_valid > 0 else float("nan")
    return out


__all__ = [
    "ndcg_at_k",
    "map_at_k",
    "mrr",
    "compute_ranking_summary",
]
