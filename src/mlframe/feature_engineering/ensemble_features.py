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
    "predictor_weighted_consensus",
    "predictor_consensus_trimmed_stats",
    "predictor_outlier_signature",
    "predictor_max_pairwise_distance",
    "predictor_quantile_spread",
]


import numpy as np
import numba


@numba.njit(parallel=True, cache=True)
def _row_bin_histogram_njit(binned: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-row histogram of small integer bin labels: ``counts[r, binned[r, j]] += 1``.

    Single fused prange pass over rows replaces the per-predictor ``np.add.at`` scatter loop (the slowest possible
    unbuffered scatter, ~27x slower than this kernel at 10M rows). Bit-identical: integer counts in float64.
    """
    n, k = binned.shape
    counts = np.zeros((n, n_bins), dtype=np.float64)
    for r in numba.prange(n):
        for j in range(k):
            counts[r, binned[r, j]] += 1.0
    return counts


def _coerce_preds(preds: np.ndarray) -> np.ndarray:
    """Validate + NaN-impute the predictor matrix.

    Coerces non-finite cells to the row's median across the remaining
    finite predictors; if a row has ALL non-finite predictions, fills
    with 0.0 (caller has no information for that row anyway).
    """
    arr = np.ascontiguousarray(preds, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"preds must be 2-D (n_rows, n_preds); got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"need >= 2 predictors for disagreement features; got {arr.shape[1]}")
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
    q25 = sorted_arr[:, fi25] * (1 - ff25) + sorted_arr[:, min(fi25 + 1, nc - 1)] * ff25
    q75 = sorted_arr[:, fi75] * (1 - ff75) + sorted_arr[:, min(fi75 + 1, nc - 1)] * ff75
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
    # Vectorised gather: shape (n, n_pairs)
    return np.abs(arr[:, iu] - arr[:, ju])


def _bin_counts(arr: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-row equal-width histogram counts over ``[row_min - eps, row_max + eps]``.

    Shared by ``predictor_consensus_entropy`` and ``predictor_top2_mode_gap`` so the
    builder computes the (identical) binning + scatter exactly once instead of twice.
    """
    lo = arr.min(axis=1, keepdims=True) - 1e-9
    hi = arr.max(axis=1, keepdims=True) + 1e-9
    span = (hi - lo) + 1e-12
    binned = np.clip(
        ((arr - lo) / span * n_bins).astype(np.int32),
        0, n_bins - 1,
    )
    return _row_bin_histogram_njit(np.ascontiguousarray(binned), n_bins)


def _entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    probs = counts / counts.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def _top2_gap_from_counts(counts: np.ndarray, k: int) -> np.ndarray:
    sorted_counts = -np.sort(-counts, axis=1)
    return (sorted_counts[:, 0] - sorted_counts[:, 1]) / float(k)


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
    return _entropy_from_counts(_bin_counts(arr, n_bins))


def predictor_top2_mode_gap(
    preds: np.ndarray, n_bins: int = 5,
) -> np.ndarray:
    """Gap between top-1 and top-2 bin counts, normalised by N.

    Same binning as ``predictor_consensus_entropy``. Returns
    ``(c_top1 - c_top2) / N`` per row. Large = one bin dominates
    (high consensus); small = two bins tied (multimodal disagreement).
    """
    arr = _coerce_preds(preds)
    return _top2_gap_from_counts(_bin_counts(arr, n_bins), arr.shape[1])


def predictor_weighted_consensus(
    preds: np.ndarray, weights: np.ndarray,
) -> tuple:
    """Weighted mean + weighted variance across predictors.

    ``weights`` shape ``(n_preds,)``; auto-normalised to sum to 1.
    Use case: when base estimators have unequal OOF-RMSE, weighting
    by inverse-OOF-RMSE produces a strictly better baseline than the
    plain ``predictor_consensus_mean``. Pair with the weighted variance
    for uncertainty estimates that respect predictor quality.

    Returns ``(weighted_mean, weighted_var)`` per row.
    """
    arr = _coerce_preds(preds)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != arr.shape[1]:
        raise ValueError(f"weights len {w.size} != n_preds {arr.shape[1]}")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("weights must sum to > 0")
    w_norm = w / w_sum
    mean = (arr * w_norm[None, :]).sum(axis=1)
    var = ((arr - mean[:, None]) ** 2 * w_norm[None, :]).sum(axis=1)
    return mean, var


def predictor_consensus_trimmed_stats(
    preds: np.ndarray, trim_frac: float = 0.2,
) -> tuple:
    """Trimmed mean + robust scale (MAD * 1.4826) per row.

    Drops ``floor(trim_frac * N)`` from each tail before averaging.
    The MAD-based scale uses the median absolute deviation around the
    full-data median (not the trimmed mean), so it's the textbook
    robust scale estimator complementing the trimmed mean.

    Use case: stacking + uncertainty for ensembles where ONE bad
    predictor can swing the plain mean. Meta-learner picks up when
    robust vs plain mean diverge (= "one base went rogue here").

    Returns ``(trimmed_mean, mad_scale)`` per row.
    """
    arr = _coerce_preds(preds)
    n, k = arr.shape
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError(f"trim_frac must be in [0, 0.5), got {trim_frac}")
    n_trim = int(np.floor(trim_frac * k))
    sorted_arr = np.sort(arr, axis=1)
    if n_trim > 0 and k - 2 * n_trim > 0:
        trimmed = sorted_arr[:, n_trim : k - n_trim]
    else:
        trimmed = sorted_arr
    trimmed_mean = trimmed.mean(axis=1)
    # MAD around the full-row median (not trimmed mean).
    median = np.median(arr, axis=1)
    mad = np.median(np.abs(arr - median[:, None]), axis=1)
    mad_scale = mad * 1.4826
    return trimmed_mean, mad_scale


def predictor_outlier_signature(
    preds: np.ndarray, k_mad: float = 2.5,
) -> tuple:
    """Per-row outlier-count + index of the most-deviating predictor.

    Outlier = predictor whose value deviates from the row's median by
    more than ``k_mad * MAD``. Returns
    ``(n_outliers, argmax_dev_idx)``: int count of outlier predictors +
    the int index ``[0 .. n_preds-1]`` of the SINGLE most-deviating
    predictor (the latter is a categorical feature the meta-learner
    can exploit as an interaction signal).

    Use case: tree meta-learners benefit from knowing WHICH base
    estimator is the dissenter (e.g. "when base_3 disagrees most,
    trust the LGBM ensemble" is a learnable pattern).
    """
    arr = _coerce_preds(preds)
    median = np.median(arr, axis=1, keepdims=True)
    dev = np.abs(arr - median)
    mad = np.median(dev, axis=1, keepdims=True) + 1e-12
    outlier_mask = dev > (k_mad * mad * 1.4826)
    n_outliers = outlier_mask.sum(axis=1).astype(np.float64)
    argmax_dev_idx = np.argmax(dev, axis=1).astype(np.float64)
    return n_outliers, argmax_dev_idx


def predictor_max_pairwise_distance(preds: np.ndarray) -> np.ndarray:
    """Max ``|pred_i - pred_j|`` across all pairs, per row.

    Worst-case disagreement signal. Cheaper than emitting all pairs
    (and survives for large N where pair-count explodes). Risk-sensitive
    applications (medical / credit) need this more than the average
    disagreement: a unanimous-but-one ensemble with one extreme outlier
    is structurally different from a uniformly disagreeing ensemble
    with the same IQR.
    """
    arr = _coerce_preds(preds)
    return arr.max(axis=1) - arr.min(axis=1)


def predictor_quantile_spread(
    preds: np.ndarray, q_low: float = 0.1, q_high: float = 0.9,
) -> tuple:
    """Per-row ``(p_low, p_high, spread)`` across predictors.

    Generalises IQR with configurable quantiles. p_low / p_high
    themselves are useful features (conformal-prediction-like band
    bounds for the meta-learner), not just the spread.

    Use case: stacking + uncertainty where the level of the prediction
    band matters (e.g. asymmetric risk: a wide upper-tail band is more
    concerning than a wide lower-tail band).
    """
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"need 0 <= q_low < q_high <= 1; got {q_low}, {q_high}")
    arr = _coerce_preds(preds)
    sorted_arr = np.sort(arr, axis=1)
    nc = sorted_arr.shape[1]
    def _q(q):
        p = q * (nc - 1)
        fi, ff = int(p), p - int(p)
        return sorted_arr[:, fi] * (1 - ff) + sorted_arr[:, min(fi + 1, nc - 1)] * ff
    p_lo = _q(q_low)
    p_hi = _q(q_high)
    return p_lo, p_hi, p_hi - p_lo


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
    counts = _bin_counts(arr, n_bins)  # shared by entropy + top2_gap; identical binning, computed once
    out = {
        "mean": predictor_consensus_mean(arr),
        "iqr": predictor_disagreement_iqr(arr),
        "var": predictor_disagreement_var(arr),
        "entropy": _entropy_from_counts(counts),
        "top2_gap": _top2_gap_from_counts(counts, arr.shape[1]),
    }
    if emit_pairs:
        out["pairs"] = predictor_pairwise_abs_diffs(arr)
    return out
