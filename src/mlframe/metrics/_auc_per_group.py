"""Per-group AUC helpers for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import fast_aucs_per_group`` (and the other
moved names) imports continue to work.

What lives here:
  - ``fast_aucs_per_group`` (naive per-group split-and-call)
  - ``fast_aucs_per_group_optimized`` (upfront filter + JIT inner loop)
  - ``compute_grouped_group_aucs`` (@njit kernel over pre-sorted data)
  - ``fast_numba_aucs_simple`` (@njit per-group ROC/PR AUC scan)
  - ``compute_mean_aucs_per_group`` (NaN-safe mean over the per-group dict)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import numba

from ._numba_params import NUMBA_NJIT_PARAMS

# ``fast_numba_aucs`` lives in core.py (tied-score AUC scan used by the
# overall-AUC pre-pass). Lazy import inside the function bodies avoids a
# circular import: ``core`` re-exports the symbols below at the bottom of
# its module, so we must not import core eagerly.

from ._core_auc_brier import _argsort_desc_for_metrics  # iter338 dispatcher


def fast_aucs_per_group(y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    Compute overall AUCs and per-group AUCs efficiently.

    Returns:
        - Overall ROC AUC
        - Overall PR AUC
        - Dictionary mapping group_id -> (roc_auc, pr_auc)
    """
    from .core import fast_numba_aucs as _fast_numba_aucs
    if y_score.ndim == 2:
        y_score = y_score[:, -1]

    # Overall AUCs
    desc_score_indices = _argsort_desc_for_metrics(y_score)  # iter338 dispatcher
    overall_roc_auc, overall_pr_auc = _fast_numba_aucs(y_true, y_score, desc_score_indices)

    # Per-group AUCs
    unique_groups = np.unique(group_ids)
    group_aucs = {}

    for group_id in unique_groups:
        group_mask = group_ids == group_id
        group_y_true = y_true[group_mask]
        group_y_score = y_score[group_mask]

        if len(group_y_true) > 1:  # Need at least 2 samples
            group_desc_indices = _argsort_desc_for_metrics(group_y_score)  # iter338 dispatcher
            roc_auc, pr_auc = _fast_numba_aucs(group_y_true, group_y_score, group_desc_indices)
            group_aucs[int(group_id)] = (roc_auc, pr_auc)
        else:
            group_aucs[int(group_id)] = (0.0, 0.0)

    return overall_roc_auc, overall_pr_auc, group_aucs


def fast_aucs_per_group_optimized(y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray = None) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    More memory-efficient version that groups data by group first.
    Better for cases with many groups and reasonable group sizes.

    Upfront filter: groups with <2 samples OR single-class y_true are NaN-
    bound by the underlying formula. We precompute per-group (count,
    pos_count) once, drop sample rows belonging to doomed groups before
    calling the numba inner loop, and emit the NaNs directly. On production
    workloads where 95 %+ of groups are single-sample (fine-grained
    group_ids), this slashes the per-group sort + iteration to only the
    valid-group subset.
    """
    from .core import fast_numba_aucs as _fast_numba_aucs
    if y_score.ndim == 2:
        y_score = y_score[:, -1]

    # Overall AUCs
    desc_score_indices = _argsort_desc_for_metrics(y_score)  # iter338 dispatcher
    overall_roc_auc, overall_pr_auc = _fast_numba_aucs(y_true, y_score, desc_score_indices)

    # By group very efficiently
    if group_ids is not None:
        # One pass over (group_id -> sample count, pos_count). np.bincount is
        # ~2-3x faster than np.add.at for the pos-count accumulation.
        unique_groups, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
        pos_counts = np.bincount(inverse, weights=y_true, minlength=len(unique_groups))
        valid_mask = (counts >= 2) & (pos_counts > 0) & (pos_counts < counts)

        group_aucs: Dict[int, Tuple[float, float]] = {}
        # Emit NaN entries for all doomed groups up front -- preserves the
        # full output contract so downstream (compute_mean_aucs_per_group +
        # the >=50 % NaN warning below) sees the same dict keys as before.
        invalid_group_ids = unique_groups[~valid_mask]
        for gid in invalid_group_ids:
            group_aucs[int(gid)] = (np.nan, np.nan)

        if valid_mask.any():
            # Mask samples belonging to valid groups only (typically 5 % of
            # input when single-sample granularity dominates) and pass the
            # subset to the JIT loop.
            sample_valid = valid_mask[inverse]
            sub_y_true = y_true[sample_valid]
            sub_y_score = y_score[sample_valid]
            sub_group_ids = group_ids[sample_valid]

            sort_indices = np.argsort(sub_group_ids)
            sorted_group_ids = sub_group_ids[sort_indices]
            sorted_y_true = sub_y_true[sort_indices]
            sorted_y_score = sub_y_score[sort_indices]

            valid_group_aucs = compute_grouped_group_aucs(sorted_group_ids, sorted_y_true, sorted_y_score)
            group_aucs.update(valid_group_aucs)

        # Observability preserved: log once per call when >=50 % of groups
        # collapsed to NaN, so operators still see "most of my group AUCs
        # are single-sample" without reading every entry.
        if group_aucs:
            n_total = len(group_aucs)
            n_nan_roc = sum(1 for (r, _p) in group_aucs.values() if np.isnan(r))
            if n_nan_roc and n_nan_roc * 2 >= n_total:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "fast_aucs_per_group_optimized: %d / %d groups returned NaN ROC AUC "
                    "(single-class or single-sample). Per-group mean is built on %d "
                    "valid groups; likely causes: target imbalance concentrated in few "
                    "groups, or group_ids granularity too fine (many 1-sample groups).",
                    n_nan_roc, n_total, n_total - n_nan_roc,
                )
    else:
        group_aucs = {}

    return overall_roc_auc, overall_pr_auc, group_aucs


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_grouped_group_aucs(sorted_group_ids: np.ndarray, sorted_y_true: np.ndarray, sorted_y_score: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """
    Compute AUCs for each group from pre-sorted data.

    NOT parallelised. Benched against a ``parallel=True`` variant that pre-
    computes group boundaries + writes to per-thread arrays (avoiding Dict-
    write contention). Result: ``par/seq`` ratio 1.77-4.80x SLOWER than seq
    across (n_groups, avg_size) shapes from (100, 10) up to (100_000, 10).
    Reason: per-group work (argsort + AUC scan on 5-50 elements) is
    microseconds; numba prange thread-spawn overhead per iteration
    dominates. Bench preserved at
    ``profiling/bench_grouped_aucs_parallel.py``.
    """
    group_aucs = {}
    n = len(sorted_group_ids)

    if n == 0:
        return group_aucs

    start_idx = 0
    current_group = sorted_group_ids[0]

    for i in range(1, n + 1):
        # Check if we've reached end or found a new group
        if i == n or sorted_group_ids[i] != current_group:
            end_idx = i
            group_size = end_idx - start_idx

            if group_size > 1:
                # Extract group data
                group_y_true = sorted_y_true[start_idx:end_idx]
                group_y_score = sorted_y_score[start_idx:end_idx]

                # Sort by score for this group.
                # numba @njit np.argsort accepts kind="mergesort" (stable
                # algorithm) but rejects kind="stable" (synonym alias).
                group_desc_indices = np.argsort(group_y_score, kind="mergesort")[::-1]  # stable sort

                # Compute AUCs for this group
                roc_auc, pr_auc = fast_numba_aucs_simple(group_y_true, group_y_score, group_desc_indices)
                group_aucs[int(current_group)] = (roc_auc, pr_auc)
            else:
                # Single-sample group: AUC is mathematically undefined.
                # Return NaN (not 0.0) so compute_mean_aucs_per_group's
                # NaN filter drops it from the mean. Previously (0.0, 0.0)
                # was treated as legitimate data and silently depressed
                # the mean AUC when a fold had many single-sample groups.
                group_aucs[int(current_group)] = (np.nan, np.nan)

            # Move to next group
            if i < n:
                start_idx = i
                current_group = sorted_group_ids[i]

    return group_aucs


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_aucs_simple(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> Tuple[float, float]:
    """
    Simplified version of the per-group AUC kernel: ROC + PR in one pass over
    score-sorted data.
    """
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    total_pos = np.sum(y_true_sorted)
    total_neg = len(y_true_sorted) - total_pos

    if total_pos == 0 or total_neg == 0:
        # Single-class data: both ROC AUC and PR AUC are undefined
        return np.nan, np.nan

    # Variables for ROC AUC
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0

    # Variables for PR AUC
    prev_recall = 0.0
    pr_auc = 0.0
    n = len(y_true_sorted)

    for i in range(n):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            # Update ROC AUC
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            # Update PR AUC
            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision
            prev_recall = current_recall

    # Normalize ROC AUC
    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        # Should not reach here due to early return, but handle defensively
        roc_auc = np.nan

    return roc_auc, pr_auc


def compute_mean_aucs_per_group(group_aucs: dict) -> tuple:
    """NaN-safe mean of per-group (roc_auc, pr_auc) entries."""
    # Compute mean per-group AUCs, ignoring NaN values
    group_roc_aucs = np.array([aucs[0] for aucs in group_aucs.values()])
    group_pr_aucs = np.array([aucs[1] for aucs in group_aucs.values()])

    # Filter out NaN values for mean calculation
    valid_roc = ~np.isnan(group_roc_aucs)
    valid_pr = ~np.isnan(group_pr_aucs)
    mean_roc_auc = np.mean(group_roc_aucs[valid_roc]) if np.any(valid_roc) else np.nan
    mean_pr_auc = np.mean(group_pr_aucs[valid_pr]) if np.any(valid_pr) else np.nan

    return mean_roc_auc, mean_pr_auc
