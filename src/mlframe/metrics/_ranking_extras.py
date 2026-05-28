"""Additional learning-to-rank metrics.

Complements ``ranking.py`` (NDCG@k, MAP@k, MRR) with the next-most-cited
LTR scores.

Public API (re-exported from ``mlframe.metrics.core``):
    * ``dcg_at_k``                 - DCG@k (un-normalised gain)
    * ``expected_reciprocal_rank`` - ERR (Chapelle et al. 2009)
    * ``hit_at_k``                 - Hit@k (= Recall@k for single-relevant)
    * ``precision_at_k``           - Precision@k

All four take ``(y_true, y_score, group_ids)`` so they integrate with
the LTR suite's grouped-by-query evaluation. Each function aggregates
per-query (mean across queries) - this matches LightGBM / CatBoost LTR
training-loop semantics.
"""
from __future__ import annotations

from math import log2

from typing import Optional, Sequence

import numpy as np
import numba

from ._numba_params import NUMBA_NJIT_PARAMS


# ----- helpers -----


def _split_by_group(
    y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray,
):
    """Return list of (start, end) slices grouped by ``group_ids``.

    Assumes rows are SORTED by group_ids (the LTR-suite convention);
    if not we sort here and return the order so the caller can re-index.
    Implemented in pure-numpy because group counts are typically small
    (~thousands of queries x tens of docs).
    """
    n = y_true.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    gids = np.ascontiguousarray(group_ids)
    # group boundaries via diff on the sorted gids.
    order = np.argsort(gids, kind="stable")
    sorted_gids = gids[order]
    yt = y_true[order]
    ys = y_score[order]
    # boundary indices
    boundaries = np.concatenate((
        np.array([0]),
        np.nonzero(np.diff(sorted_gids))[0] + 1,
        np.array([n]),
    ))
    return boundaries, yt, ys


# ----- DCG@k -----


@numba.njit(**NUMBA_NJIT_PARAMS)
def _dcg_per_group_kernel(
    y_true: np.ndarray, y_score: np.ndarray, k: int, exp_gain: bool,
) -> float:
    """Per-group DCG@k with exponential gain.

    DCG = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)    when exp_gain=True
    DCG = sum_{i=1..k} rel_i / log2(i + 1)            when exp_gain=False
    """
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    # argsort descending by score; numba doesn't accept axis=, do it manually.
    order = np.argsort(-y_score)
    kk = k if k < n else n
    dcg = 0.0
    for i in range(kk):
        rel = y_true[order[i]]
        if exp_gain:
            gain = (2.0 ** rel) - 1.0
        else:
            gain = float(rel)
        dcg += gain / log2(i + 2.0)  # log2(rank + 1), rank starts at 1
    return dcg


def dcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    k: int = 10,
    exp_gain: bool = True,
) -> float:
    """Discounted Cumulative Gain at k, averaged across query groups.

    Unlike NDCG, NOT normalised by the ideal DCG - useful when comparing
    absolute relevance volumes across query sets. With ``exp_gain=True``
    (default) uses the (2^rel - 1) gain formula compatible with
    LightGBM / CatBoost LTR; with ``exp_gain=False`` uses linear gain
    (Burges 2005 variant 1).
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if group_ids is None:
        # Treat as a single group.
        return float(_dcg_per_group_kernel(yt, ys, int(k), bool(exp_gain)))
    boundaries, yt_s, ys_s = _split_by_group(yt, ys, group_ids)
    n_groups = boundaries.shape[0] - 1
    if n_groups == 0:
        return np.nan
    total = 0.0
    counted = 0
    for g in range(n_groups):
        s = boundaries[g]
        e = boundaries[g + 1]
        if e - s == 0:
            continue
        total += _dcg_per_group_kernel(
            yt_s[s:e], ys_s[s:e], int(k), bool(exp_gain),
        )
        counted += 1
    return total / counted if counted > 0 else np.nan


# ----- ERR (Expected Reciprocal Rank) -----


@numba.njit(**NUMBA_NJIT_PARAMS)
def _err_per_group_kernel(
    y_true: np.ndarray, y_score: np.ndarray, k: int, max_grade: float,
) -> float:
    """Per-group ERR. Models cascade browsing: user inspects rank r with
    probability P(stop|seen) = R_r and stops, where R_r = (2^rel_r - 1) /
    (2^max_grade). ERR = sum_r (1/r) * R_r * prod_{j<r}(1 - R_j).
    """
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-y_score)
    kk = k if k < n else n
    denom = (2.0 ** max_grade)
    err = 0.0
    p_remain = 1.0  # probability that the user is still browsing
    for i in range(kk):
        rel = y_true[order[i]]
        R = ((2.0 ** rel) - 1.0) / denom
        err += p_remain * R / (i + 1.0)
        p_remain *= (1.0 - R)
        if p_remain <= 0.0:
            break
    return err


def expected_reciprocal_rank(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    k: int = 10,
    max_grade: Optional[float] = None,
) -> float:
    """Expected Reciprocal Rank (Chapelle et al. 2009).

    Cascade user-model metric: higher rank => more weight, AND lower
    grades early stop probability mass earlier. ``max_grade`` defaults
    to the maximum observed relevance label.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    mg = float(max_grade) if max_grade is not None else float(yt.max() if yt.size > 0 else 1.0)
    if mg <= 0:
        mg = 1.0
    if group_ids is None:
        return float(_err_per_group_kernel(yt, ys, int(k), mg))
    boundaries, yt_s, ys_s = _split_by_group(yt, ys, group_ids)
    n_groups = boundaries.shape[0] - 1
    if n_groups == 0:
        return np.nan
    total = 0.0
    counted = 0
    for g in range(n_groups):
        s = boundaries[g]
        e = boundaries[g + 1]
        if e - s == 0:
            continue
        total += _err_per_group_kernel(yt_s[s:e], ys_s[s:e], int(k), mg)
        counted += 1
    return total / counted if counted > 0 else np.nan


# ----- Hit@k / Precision@k -----


@numba.njit(**NUMBA_NJIT_PARAMS)
def _hit_at_k_per_group_kernel(
    y_true: np.ndarray, y_score: np.ndarray, k: int,
) -> float:
    """1 if any of the top-k by score is relevant (y_true > 0), else 0."""
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-y_score)
    kk = k if k < n else n
    for i in range(kk):
        if y_true[order[i]] > 0:
            return 1.0
    return 0.0


@numba.njit(**NUMBA_NJIT_PARAMS)
def _precision_at_k_per_group_kernel(
    y_true: np.ndarray, y_score: np.ndarray, k: int,
) -> float:
    """(# relevant in top-k) / k."""
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-y_score)
    kk = k if k < n else n
    hits = 0
    for i in range(kk):
        if y_true[order[i]] > 0:
            hits += 1
    return hits / k  # NOTE: denominator is k, not kk, by LTR convention


def hit_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    k: int = 10,
) -> float:
    """Hit@k averaged across queries: fraction of queries with at least
    one relevant doc in the top-k.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if group_ids is None:
        return float(_hit_at_k_per_group_kernel(yt, ys, int(k)))
    boundaries, yt_s, ys_s = _split_by_group(yt, ys, group_ids)
    n_groups = boundaries.shape[0] - 1
    if n_groups == 0:
        return np.nan
    total = 0.0
    counted = 0
    for g in range(n_groups):
        s = boundaries[g]
        e = boundaries[g + 1]
        if e - s == 0:
            continue
        total += _hit_at_k_per_group_kernel(yt_s[s:e], ys_s[s:e], int(k))
        counted += 1
    return total / counted if counted > 0 else np.nan


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    k: int = 10,
) -> float:
    """Precision@k averaged across queries: (# relevant in top-k) / k.

    Differs from Hit@k: Hit@k saturates at 1.0 as soon as ONE relevant
    doc appears, Precision@k keeps counting.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if group_ids is None:
        return float(_precision_at_k_per_group_kernel(yt, ys, int(k)))
    boundaries, yt_s, ys_s = _split_by_group(yt, ys, group_ids)
    n_groups = boundaries.shape[0] - 1
    if n_groups == 0:
        return np.nan
    total = 0.0
    counted = 0
    for g in range(n_groups):
        s = boundaries[g]
        e = boundaries[g + 1]
        if e - s == 0:
            continue
        total += _precision_at_k_per_group_kernel(yt_s[s:e], ys_s[s:e], int(k))
        counted += 1
    return total / counted if counted > 0 else np.nan
