"""``collapse_predictions_by_group``: broadcast a group-level prediction statistic back to every row.

Source: 6th_ieee-cis-fraud-detection.md -- "Take all predictions from a customer based on our UID6, and
combine the predictions to a single value so that all transactions of a customer have the same value... take
90% quantile of the predictions." Individual per-row predictions for the same entity can disagree even when
within-entity consistency is known to matter more than per-row independence (e.g. fraud risk genuinely is a
property of the CUSTOMER, not the individual transaction) -- collapsing to a group statistic (mean, or a
high quantile to bias toward the entity's worst-observed risk) and broadcasting it back removes that
inconsistency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _weighted_quantile_by_group_numba(
    sorted_val: np.ndarray, sorted_w: np.ndarray, sorted_codes: np.ndarray, total_w: np.ndarray, n_groups: int, q: float
) -> np.ndarray:
    """Single O(n) pass over rows pre-sorted by (group, value) computing a weighted quantile per group.

    Uses the midpoint weighted-quantile definition: point ``i``'s cumulative position is
    ``(cumw_i - w_i / 2) / total_w``, and the result linearly interpolates between the two points straddling
    ``q`` (matches ``np.quantile``'s unweighted linear interpolation when all weights are equal).
    """
    result = np.empty(n_groups, dtype=np.float64)
    found = np.zeros(n_groups, dtype=np.bool_)
    cumw = np.zeros(n_groups, dtype=np.float64)
    prev_val = np.zeros(n_groups, dtype=np.float64)
    prev_pos = np.zeros(n_groups, dtype=np.float64)

    for i in range(sorted_val.shape[0]):
        g = sorted_codes[i]
        if found[g]:
            continue
        w = sorted_w[i]
        cumw[g] += w
        pos = (cumw[g] - 0.5 * w) / total_w[g]
        if pos >= q:
            if cumw[g] == w:
                # first (and possibly only) row seen for this group already meets/exceeds q.
                result[g] = sorted_val[i]
            else:
                denom = pos - prev_pos[g]
                if denom <= 0.0:
                    result[g] = sorted_val[i]
                else:
                    frac = (q - prev_pos[g]) / denom
                    result[g] = prev_val[g] + frac * (sorted_val[i] - prev_val[g])
            found[g] = True
        else:
            prev_val[g] = sorted_val[i]
            prev_pos[g] = pos

    return result


def collapse_predictions_by_group(
    predictions: np.ndarray, group: np.ndarray, stat: str = "mean", quantile: float = 0.9, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Collapse ``predictions`` within each ``group`` to a single statistic, broadcast back to every row.

    Parameters
    ----------
    predictions
        ``(n,)`` per-row predictions.
    group
        ``(n,)`` grouping key (e.g. a customer/entity id).
    stat
        ``"mean"`` or ``"quantile"``.
    quantile
        Used only when ``stat="quantile"`` (e.g. ``0.9`` matches the source's own choice -- biases toward
        the entity's higher-risk transactions rather than averaging them away).
    weights
        Optional ``(n,)`` non-negative per-row reliability/recency weight. When given, the group statistic is
        computed as a weighted mean/quantile instead of an unweighted one -- e.g. weight a recent prediction
        higher than a stale one for the same entity so it dominates the group aggregate. ``None`` (the
        default) reproduces the original unweighted behavior bit-for-bit; this parameter is strictly opt-in.

    Returns
    -------
    np.ndarray
        ``(n,)`` -- every row of the same group replaced by that group's collapsed statistic.
    """
    if stat not in ("mean", "quantile"):
        raise ValueError(f"collapse_predictions_by_group: unsupported stat {stat!r}, expected 'mean' or 'quantile'")

    predictions_arr = np.asarray(predictions, dtype=np.float64)

    if weights is None:
        df = pd.DataFrame({"prediction": predictions_arr, "group": group})
        grouped = df.groupby("group", sort=False)["prediction"]
        if stat == "mean":
            group_stat = grouped.transform("mean")
        else:
            # transform(lambda s: s.quantile(q)) invokes one Python-level callback PER GROUP (each paying its
            # own DataFrame/Index construction overhead) -- measured at 27s/50000 groups vs 81ms for the "mean"
            # path. pandas' GroupBy.quantile() is a single vectorized, C-level pass computing every group's
            # quantile at once; map the (small) per-group result back onto every row instead.
            per_group_quantile = grouped.quantile(quantile)
            group_stat = df["group"].map(per_group_quantile)
        return np.asarray(group_stat.to_numpy())

    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.shape != predictions_arr.shape:
        raise ValueError(f"collapse_predictions_by_group: weights shape {weights_arr.shape} must match predictions shape {predictions_arr.shape}")
    if np.any(weights_arr < 0):
        raise ValueError("collapse_predictions_by_group: weights must be non-negative")

    codes, uniques = pd.factorize(pd.Series(group), sort=False)
    n_groups = uniques.shape[0]
    total_w_by_code = np.bincount(codes, weights=weights_arr, minlength=n_groups)
    if np.any(total_w_by_code[np.unique(codes)] <= 0):
        raise ValueError("collapse_predictions_by_group: every group needs at least one positive weight")

    if stat == "mean":
        weighted_pred_sum = np.bincount(codes, weights=weights_arr * predictions_arr, minlength=n_groups)
        result_per_group = weighted_pred_sum / total_w_by_code
    else:
        sort_idx = np.lexsort((predictions_arr, codes))
        sorted_val = predictions_arr[sort_idx]
        sorted_codes = codes[sort_idx].astype(np.int64)
        sorted_w = weights_arr[sort_idx]
        result_per_group = _weighted_quantile_by_group_numba(sorted_val, sorted_w, sorted_codes, total_w_by_code, n_groups, float(quantile))

    return np.asarray(result_per_group[codes])


__all__ = ["collapse_predictions_by_group"]
