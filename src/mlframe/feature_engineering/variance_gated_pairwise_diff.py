"""``variance_gated_pairwise_diff``: combinatorial pairwise differences, pruned by variance before materializing.

Source: 4th_mechanisms-of-action-moa-prediction.md -- combinatorial ``c[0]-c[1]`` diff features across all
pairs, kept only if ``np.var(diff) > threshold`` to control combinatorial explosion (872 choose 2 pruned by
variance). At hundreds of columns, C(n,2) grows quadratically (872 choose 2 = ~380k pairs) -- materializing
the full combinatorial set before filtering would waste memory proportional to the UNPRUNED count, so
survival is decided from a single (n_cols, n_cols) covariance matrix BEFORE any diff array is built (see
the implementation note below); only SURVIVING pairs' diff columns are ever materialized. Peak memory is
therefore ``O(n_rows * n_surviving_pairs)`` -- inherent to the output itself, not the unpruned candidate
count -- plus the ``O(n_rows * n_cols)`` input and ``O(n_cols^2)`` covariance matrix. If ``min_variance`` is
set low enough that most pairs survive, that output can still be large; an explicit ceiling check raises a
clear ``MemoryError`` (with the projected size) before allocating, rather than silently exhausting RAM.
"""
from __future__ import annotations

import os
from itertools import combinations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc

# RAM ceiling (bytes) for the surviving-pairs diff-column output (float64, n_rows x n_surviving_pairs).
# Default ~1 GiB; override at runtime via MLFRAME_VARIANCE_GATED_DIFF_MAX_BYTES (read per-call). Mirrors
# calibration.policy._build_resample_indices's projected-size-before-alloc guard.
_DEFAULT_MAX_OUTPUT_BYTES: int = 1 << 30


def _max_output_bytes() -> int:
    """RAM ceiling (bytes) for the surviving-pairs diff-column output; env-overridable per call."""
    raw = os.environ.get("MLFRAME_VARIANCE_GATED_DIFF_MAX_BYTES")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_OUTPUT_BYTES
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_MAX_OUTPUT_BYTES


def variance_gated_pairwise_diff(
    df: pd.DataFrame,
    columns: Sequence[str],
    min_variance: float = 1e-6,
    chunk_size: int = 2000,
    prune_against_target: Optional[Tuple[np.ndarray, float]] = None,
) -> pd.DataFrame:
    """Generate ``col_a - col_b`` for every pair in ``columns``, keeping only pairs with variance above threshold.

    Parameters
    ----------
    df
        Source frame.
    columns
        Numeric columns to combine pairwise.
    min_variance
        A pair's diff column is kept only if its variance exceeds this threshold (drops near-constant diffs
        -- e.g. two near-duplicate columns whose difference carries almost no signal). Computed as ``var(a) +
        var(b) - 2*cov(a, b)`` via ``np.cov`` (see the implementation note below), whose default ``ddof=1``
        (sample variance) differs from ``np.var(diff)``'s default ``ddof=0`` (population variance) by a
        factor of ``n/(n-1)`` -- immaterial at any real-world ``n``, but stated explicitly here since the two
        aren't literally the same formula.
    chunk_size
        Number of surviving pairs' diff columns materialized per batch. The variance gate itself is decided
        up front from one covariance-matrix pass (not chunked), so this no longer bounds peak memory --
        peak memory is ``O(n_rows * n_surviving_pairs)``, inherent to the output DataFrame -- but still
        batches the materialization loop to keep any single iteration's working set small.
    prune_against_target
        Optional ``(y, tolerance)``. The variance gate alone only rules out near-constant diffs -- a diff can
        have high variance yet carry zero relationship to the target (e.g. the difference of two independent
        noise columns). When supplied, every variance-surviving diff is additionally screened by
        :func:`mlframe.feature_selection.drop_near_noise_univariate_auc.drop_near_noise_univariate_auc`
        against ``y``, and dropped when its own univariate AUC sits within ``tolerance`` of chance (0.5).
        ``None`` (default) keeps every variance-surviving diff, matching the original unconditional behaviour.

    Returns
    -------
    pd.DataFrame
        One column per surviving pair, named ``"{col_a}__diff__{col_b}"``.
    """
    col_index = {col: i for i, col in enumerate(columns)}
    values = {col: df[col].to_numpy(dtype=np.float64) for col in columns}

    # var(a - b) = var(a) + var(b) - 2*cov(a, b) -- computing the (n_cols, n_cols) covariance matrix ONCE
    # (a single BLAS-backed pass) and deriving every pair's variance via vectorized arithmetic replaces
    # C(n_cols, 2) separate O(n_rows) np.var() reductions (each paying its own numpy-dispatch overhead,
    # measured as the dominant cProfile cost at n_cols=150) with one O(n_rows * n_cols^2) matrix computation
    # plus O(n_cols^2) elementwise math. Crucially, PRUNED pairs' diff arrays are never materialized at all
    # (the survival decision is made from the covariance matrix alone, before any subtraction happens).
    X = df[columns].to_numpy(dtype=np.float64)
    cov = np.cov(X, rowvar=False)
    var_diag = np.diag(cov)

    all_pairs = list(combinations(columns, 2))
    surviving_pairs = []
    for col_a, col_b in all_pairs:
        i, j = col_index[col_a], col_index[col_b]
        pair_var = var_diag[i] + var_diag[j] - 2.0 * cov[i, j]
        if pair_var > min_variance:
            surviving_pairs.append((col_a, col_b))

    n_rows = len(df)
    projected_bytes = 8 * n_rows * len(surviving_pairs)
    ceiling = _max_output_bytes()
    if projected_bytes > ceiling:
        raise MemoryError(
            f"variance_gated_pairwise_diff: {len(surviving_pairs)} pairs survived the variance gate "
            f"(min_variance={min_variance!r}) over {n_rows} rows, projecting a "
            f"{projected_bytes / (1 << 30):.2f} GiB output DataFrame, exceeding the {ceiling / (1 << 30):.2f} GiB "
            f"ceiling. Raise min_variance, reduce the candidate column count, or raise "
            f"MLFRAME_VARIANCE_GATED_DIFF_MAX_BYTES if the RAM is available."
        )

    out: Dict[str, np.ndarray] = {}
    for chunk_start in range(0, len(surviving_pairs), chunk_size):
        for col_a, col_b in surviving_pairs[chunk_start : chunk_start + chunk_size]:
            out[f"{col_a}__diff__{col_b}"] = values[col_a] - values[col_b]

    result = pd.DataFrame(out, index=df.index)

    if prune_against_target is not None and result.shape[1] > 0:
        y, tolerance = prune_against_target
        y_arr = np.asarray(y)
        dropped = drop_near_noise_univariate_auc(result, y_arr, columns=list(result.columns), tolerance=tolerance)
        if dropped:
            result = result.drop(columns=dropped)

    return result


__all__ = ["variance_gated_pairwise_diff"]
