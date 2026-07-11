"""``apply_group_zero_sum_constraint``: enforce a known per-group weighted-sum conservation law on predictions.

Source: 9th_optiver-trading-at-the-close.md -- code subtracting the ``groupby(date_id, seconds_in_bucket)``
weighted mean of predictions so the weighted sum is exactly zero within each group, matching a known property
of the target (Optiver's target is itself a cross-sectional relative price move, which sums to ~0 per
auction-second by construction). Unlike :func:`mlframe.calibration.group_bias_correction`, this needs no
validation-slice fitting: the constraint is a fact about the target's DEFINITION, not something learned from
data, so it applies directly to any prediction at inference time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def apply_group_zero_sum_constraint(predictions: np.ndarray, group: np.ndarray, weights: Optional[np.ndarray] = None, target_sum: float = 0.0) -> np.ndarray:
    """Shift each row's prediction by a constant per-group offset so the group's weighted sum equals ``target_sum``.

    ``corrected = pred - weighted_mean(pred | group) + target_sum / sum(weights | group)`` -- equivalently,
    for the default ``target_sum=0.0``, each group's weighted predictions are re-centered to sum to exactly
    zero, without changing the RELATIVE ordering of predictions within a group (a constant shift per group).

    Parameters
    ----------
    predictions
        ``(n,)`` raw model predictions.
    group
        ``(n,)`` group label per row (e.g. a combined ``date_id``/``seconds_in_bucket`` key).
    weights
        ``(n,)`` per-row weight, or ``None`` for an unweighted (equal-weight) sum/mean within each group.
    target_sum
        The known constant the group's WEIGHTED sum should equal. Default ``0.0`` (zero-sum conservation law).

    Returns
    -------
    np.ndarray
        ``(n,)`` corrected predictions.
    """
    pred_arr = np.asarray(predictions, dtype=np.float64)
    w_arr = np.ones_like(pred_arr) if weights is None else np.asarray(weights, dtype=np.float64)

    df = pd.DataFrame({"pred": pred_arr, "w": w_arr, "wpred": pred_arr * w_arr, "group": group})
    agg = df.groupby("group", sort=False).agg(w_sum=("w", "sum"), wpred_sum=("wpred", "sum"))
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(agg["w_sum"].to_numpy() != 0, (agg["wpred_sum"].to_numpy() - target_sum) / agg["w_sum"].to_numpy(), 0.0)
    offset_by_group = pd.Series(offset, index=agg.index)

    group_offset = pd.Series(group).map(offset_by_group).fillna(0.0).to_numpy(dtype=np.float64)
    return np.asarray(pred_arr - group_offset)


__all__ = ["apply_group_zero_sum_constraint"]
