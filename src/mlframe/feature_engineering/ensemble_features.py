"""Per-row disagreement features across N predictors of the same target.

Given a set of columns ``pred_1, pred_2, ..., pred_N`` that all
predict the SAME target (typically from a level-0 stack of base
estimators), generate features capturing how much they disagree per
row. These features feed into meta-learners (level-1 stacking) and
into uncertainty quantification.

Catalogue:

* ``predictor_disagreement_iqr`` -- inter-quartile range across the N
  predictions per row. Robust to a single outlier predictor.
* ``predictor_disagreement_var`` -- unbiased sample variance per row.
* ``predictor_pairwise_abs_diffs`` -- one column per (i, j) pair,
  ``|pred_i - pred_j|``. Useful for tree models that can exploit
  specific-pair disagreement.
* ``predictor_consensus_entropy`` -- multi-modal entropy of the
  predictions (histogrammed per row into a few bins).
* ``predictor_top2_mode_gap`` -- fraction of predictors in the top
  bin minus second bin (a "decisiveness" indicator).
* ``predictor_consensus_mean`` -- simple mean across predictors
  (useful as a baseline feature alongside disagreement signals).

All functions take a 2-D ``preds`` array of shape ``(n_rows, n_preds)``
and return either a 1-D ``(n_rows,)`` or a 2-D ``(n_rows, n_pairs)``
array of per-row features. NaN-safe: non-finite predictions are
coerced to the row's median before computation so a single failing
predictor doesn't NaN-propagate to all features.
"""

from __future__ import annotations

__all__ = [
    "predictor_disagreement_iqr",
    "predictor_disagreement_var",
    "predictor_pairwise_abs_diffs",
    "predictor_consensus_entropy",
    "predictor_top2_mode_gap",
    "predictor_consensus_mean",
    "predictor_disagreement_features",
]

from typing import Tuple

import numpy as np


def _coerce_preds(preds: np.ndarray) -> np.ndarray:
    """Validate + NaN-impute the predictor matrix.

    Coerces non-finite cells to the row's median across the remaining
    finite predictors; if a row has ALL non-finite predictions, fills
    with 0.0 (caller has no information for that row anyway).
    """
    arr = np.ascontiguousarray(preds, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"preds must be 2-D (n_rows, n_preds); got shape {arr.shape}"
        )
    if arr.shape[1] < 2:
        raise ValueError(
            f"need >= 2 predictors for disagreement features; got {arr.shape[1]}"
        )
    finite = np.isfinite(arr)
    if not finite.all():
        # Row medians excluding non-finite cells.
        row_med = np.where(
            finite.any(axis=1, keepdims=True),
            np.nanmedian(np.where(finite, arr, np.nan), axis=1, keepdims=True),
            0.0,
        )
        arr = np.where(finite, arr, row_med)
    return arr


def predictor_consensus_mean(preds: np.ndarray) -> np.ndarray:
    """Simple mean across predictors per row. NaN-safe."""
    arr = _coerce_preds(preds)
    return arr.mean(axis=1)


def predictor_disagreement_iqr(preds: np.ndarray) -> np.ndarray:
    """Per-row inter-quartile range across the N predictors.

    Uses ``np.sort`` + linear-interp at fractional positions instead
    of ``np.percentile`` for speed (the latter goes through
    ``apply_along_axis`` which is ~10x slower at this shape).
    """
    arr = _coerce_preds(preds)
    sorted_arr = np.sort(arr, axis=1)
    nc = sorted_arr.shape[1]
    p25 = 0.25 * (nc - 1)
    p75 = 0.75 * (nc - 1)
    fi25, ff25 = int(p25), p25 - int(p25)
    fi75, ff75 = int(p75), p75 - int(p75)
    q25 = (
        sorted_arr[:, fi25] * (1 - ff25)
        + sorted_arr[:, min(fi25 + 1, nc - 1)] * ff25
    )
    q75 = (
        sorted_arr[:, fi75] * (1 - ff75)
        + sorted_arr[:, min(fi75 + 1, nc - 1)] * ff75
    )
    return q75 - q25


def predictor_disagreement_var(preds: np.ndarray) -> np.ndarray:
    """Per-row unbiased sample variance across the N predictors."""
    arr = _coerce_preds(preds)
    return arr.var(axis=1, ddof=1) if arr.shape[1] > 1 else np.zeros(arr.shape[0])


def predictor_pairwise_abs_diffs(preds: np.ndarray) -> np.ndarray:
    """Absolute differences for every (i, j) predictor pair.

    Output shape: ``(n_rows, n_pairs)`` where ``n_pairs = N*(N-1)//2``,
    column order = pair index in lexicographic ``(i, j)`` with ``i < j``.
    Tree-based meta-learners exploit specific-pair disagreements;
    aggregating into IQR or variance throws this away.
    """
    arr = _coerce_preds(preds)
    n, k = arr.shape
    iu, ju = np.triu_indices(k, k=1)
    n_pairs = iu.size
    # Vectorised gather: shape (n, n_pairs)
    return np.abs(arr[:, iu] - arr[:, ju])


def predictor_consensus_entropy(
    preds: np.ndarray, n_bins: int = 5,
) -> np.ndarray:
    """Shannon entropy of the per-row predictor histogram.

    For each row, bin the N predictor values into ``n_bins`` equal-
    width bins on ``[row_min - eps, row_max + eps]`` and emit
    ``H = -sum(p_i * log(p_i + eps))``. High = disagreement; low =
    consensus on a narrow range.
    """
    arr = _coerce_preds(preds)
    n, k = arr.shape
    lo = arr.min(axis=1, keepdims=True) - 1e-9
    hi = arr.max(axis=1, keepdims=True) + 1e-9
    span = (hi - lo) + 1e-12
    binned = np.clip(
        ((arr - lo) / span * n_bins).astype(np.int32),
        0, n_bins - 1,
    )
    counts = np.zeros((n, n_bins), dtype=np.float64)
    # scatter-add: counts[r, binned[r, j]] += 1 for each j
    for j in range(k):
        np.add.at(counts, (np.arange(n), binned[:, j]), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def predictor_top2_mode_gap(
    preds: np.ndarray, n_bins: int = 5,
) -> np.ndarray:
    """Gap between top-1 and top-2 bin counts, normalised by N.

    Same binning as ``predictor_consensus_entropy``. Returns
    ``(c_top1 - c_top2) / N`` per row. Large = one bin dominates
    (high consensus); small = two bins tied (multimodal disagreement).
    """
    arr = _coerce_preds(preds)
    n, k = arr.shape
    lo = arr.min(axis=1, keepdims=True) - 1e-9
    hi = arr.max(axis=1, keepdims=True) + 1e-9
    span = (hi - lo) + 1e-12
    binned = np.clip(
        ((arr - lo) / span * n_bins).astype(np.int32),
        0, n_bins - 1,
    )
    counts = np.zeros((n, n_bins), dtype=np.float64)
    for j in range(k):
        np.add.at(counts, (np.arange(n), binned[:, j]), 1.0)
    sorted_counts = -np.sort(-counts, axis=1)
    return (sorted_counts[:, 0] - sorted_counts[:, 1]) / float(k)


def predictor_disagreement_features(
    preds: np.ndarray,
    *,
    emit_pairs: bool = True,
    n_bins: int = 5,
) -> dict:
    """All-in-one builder: returns a dict of feature arrays.

    Keys returned (caller decides which to inject into the DataFrame):

    * ``mean``
    * ``iqr``
    * ``var``
    * ``entropy``
    * ``top2_gap``
    * ``pairs`` (shape ``(n_rows, N*(N-1)/2)``) -- only when
      ``emit_pairs=True``; suppress for large N where pair count
      explodes (N=20 -> 190 pair columns).

    Caller usage::

        out = predictor_disagreement_features(df[pred_cols].to_numpy())
        df = df.with_columns([
            pl.Series("preds_mean", out["mean"]),
            pl.Series("preds_iqr", out["iqr"]),
            pl.Series("preds_entropy", out["entropy"]),
            ...
        ])
    """
    arr = _coerce_preds(preds)
    out = {
        "mean": predictor_consensus_mean(arr),
        "iqr": predictor_disagreement_iqr(arr),
        "var": predictor_disagreement_var(arr),
        "entropy": predictor_consensus_entropy(arr, n_bins=n_bins),
        "top2_gap": predictor_top2_mode_gap(arr, n_bins=n_bins),
    }
    if emit_pairs:
        out["pairs"] = predictor_pairwise_abs_diffs(arr)
    return out
