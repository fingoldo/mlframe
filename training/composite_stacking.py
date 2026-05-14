"""Stacking-aware transform selection (measure-first): residual_correlation_matrix + max_off_diagonal_correlation diagnostic; stacking_aware_gate (NNLS-weight gate that keeps transforms contributing orthogonal signal). Pure numpy + scipy.optimize.nnls; no composite-internal deps."""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# stacking-aware transform selection (R10c brainstorm round-2 extension C; measure-first gate).
#
# Two complementary helpers:
# 1. ``residual_correlation_matrix(transform_residuals)``: measurement-first diagnostic. Computes the pairwise Pearson correlation between transform residuals to answer "are these transforms decorrelated enough that stacking will help?" When the off-diagonal max correlation exceeds ~0.8, the agent's brainstorm recommends NOT using the stacking-aware gate (transforms are too redundant -- the ensemble can't extract orthogonal signal).
# 2. ``stacking_aware_gate(transform_predictions, y_train, min_weight=0.05)``: the actual gate. Fits NNLS over the transform predictions to recover non-negative weights summing to ~1; transforms with weight below ``min_weight`` are dropped. Survivors carry orthogonal signal; rejected ones don't add value over the rest.
#
# Measure-first protocol:
#   corr_matrix = residual_correlation_matrix({name: T_for_name, ...})
#   if max_off_diag(corr_matrix) > 0.8:  # too redundant
#       skip_stacking_gate = True
#   else:
#       survivors = stacking_aware_gate(predictions, y_train)
# ----------------------------------------------------------------------


def residual_correlation_matrix(
    transform_residuals: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Pairwise Pearson correlation between transform residuals.

    Parameters
    ----------
    transform_residuals
        Ordered mapping ``transform_name -> 1-D residual ndarray``. All arrays must have the same length (caller is responsible).

    Returns
    -------
    (corr_matrix, names):
    - ``corr_matrix``: ``(K, K)`` ndarray of Pearson correlations. Diagonal is 1.0. NaN entries indicate degenerate inputs (constant residual).
    - ``names``: list of transform names in matrix-order (preserves the input dict's insertion order).

    A high max-off-diagonal correlation (>= 0.8 by convention from the brainstorm) means the transforms produce nearly the same residual; stacking them adds compute without information. Low max-off-diagonal means the transforms are orthogonal and stacking can extract more signal than any single transform.
    """
    if not transform_residuals:
        return np.zeros((0, 0), dtype=np.float64), []
    names = list(transform_residuals.keys())
    K = len(names)
    arrs = [
        np.asarray(transform_residuals[n], dtype=np.float64).reshape(-1)
        for n in names
    ]
    n_rows = arrs[0].size
    if any(a.size != n_rows for a in arrs):
        raise ValueError(
            "residual_correlation_matrix: all residual arrays must have the same length; "
            f"got lengths {[a.size for a in arrs]}"
        )
    M = np.column_stack(arrs)
    # Masked correlation: drop rows with any non-finite entry to keep np.corrcoef well-defined.
    finite_rows = np.all(np.isfinite(M), axis=1)
    if finite_rows.sum() < 3:
        return np.full((K, K), np.nan, dtype=np.float64), names
    if K == 1:
        # ``np.corrcoef`` on a single column returns a 0-D scalar; return a proper 1x1 matrix for downstream consumers expecting ``.shape[0]``.
        return np.array([[1.0]], dtype=np.float64), names
    return np.corrcoef(M[finite_rows], rowvar=False), names


def max_off_diagonal_correlation(corr_matrix: np.ndarray) -> float:
    """Convenience: max of abs(off-diagonal) entries. Returns 0.0 for 0-D / 0x0 / 1x1 matrices, NaN-tolerant via ``np.nanmax``."""
    if corr_matrix.ndim < 2:
        return 0.0
    K = corr_matrix.shape[0]
    if K < 2:
        return 0.0
    # Mask out the diagonal.
    off = np.abs(corr_matrix).copy()
    np.fill_diagonal(off, np.nan)
    val = float(np.nanmax(off))
    return 0.0 if not np.isfinite(val) else val


def stacking_aware_gate(
    transform_predictions: dict[str, np.ndarray],
    y_train: np.ndarray,
    *,
    min_weight: float = 0.05,
) -> tuple[list[str], dict[str, float]]:
    """NNLS-weight-based gate: keep only transforms whose stacking weight clears ``min_weight``.

    Parameters
    ----------
    transform_predictions
        Ordered mapping ``transform_name -> 1-D y-scale prediction ndarray``. All arrays must have the same length as ``y_train``.
    y_train
        True training targets on y-scale (1-D ndarray of the same length).
    min_weight
        Threshold below which a transform is rejected. Default 0.05 (= 5% of the unit weight budget).

    Returns
    -------
    (survivors, weights):
    - ``survivors``: list of transform names whose NNLS weight >= ``min_weight``. Empty when no transform clears the threshold (caller should fall back to the single-best transform by RMSE).
    - ``weights``: dict[name -> weight] for ALL inputs (including rejected ones for diagnostic), normalised so the survivors' weights sum to 1.0.
    """
    if not transform_predictions:
        return [], {}
    names = list(transform_predictions.keys())
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    arrs = [
        np.asarray(transform_predictions[n], dtype=np.float64).reshape(-1)
        for n in names
    ]
    if any(a.size != y.size for a in arrs):
        raise ValueError(
            "stacking_aware_gate: all prediction arrays must match y_train length; "
            f"got y={y.size}, preds={[a.size for a in arrs]}"
        )
    X = np.column_stack(arrs)
    finite_rows = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if finite_rows.sum() < max(3, X.shape[1] + 1):
        # Degenerate: too few finite rows for NNLS. Return everything with uniform weights.
        uniform = {n: 1.0 / len(names) for n in names}
        return list(names), uniform
    from scipy.optimize import nnls  # lazy
    try:
        w, _ = nnls(X[finite_rows], y[finite_rows])
    except Exception:
        uniform = {n: 1.0 / len(names) for n in names}
        return list(names), uniform
    raw_weights = {n: float(w[i]) for i, n in enumerate(names)}
    survivors = [n for n, wv in raw_weights.items() if wv >= min_weight]
    if survivors:
        s = sum(raw_weights[n] for n in survivors)
        if s > 0:
            normalised = {
                n: (raw_weights[n] / s if n in survivors else raw_weights[n])
                for n in names
            }
        else:
            normalised = raw_weights
    else:
        normalised = raw_weights
    return survivors, normalised
