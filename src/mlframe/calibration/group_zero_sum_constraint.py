"""``apply_group_zero_sum_constraint``: enforce a known per-group weighted-sum conservation law on predictions.

Source: 9th_optiver-trading-at-the-close.md -- code subtracting the ``groupby(date_id, seconds_in_bucket)``
weighted mean of predictions so the weighted sum is exactly zero within each group, matching a known property
of the target (Optiver's target is itself a cross-sectional relative price move, which sums to ~0 per
auction-second by construction). Unlike :func:`mlframe.calibration.group_bias_correction`, this needs no
validation-slice fitting: the constraint is a fact about the target's DEFINITION, not something learned from
data, so it applies directly to any prediction at inference time.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
        ``(n,)`` group label per row (e.g. a combined ``date_id``/``seconds_in_bucket`` key). A NaN group
        label is excluded from the fitted offset table (pandas groupby's own ``dropna=True``) and receives
        offset ``0.0`` (a logged warning, no correction) -- not an error.
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

    n_nan_groups = int(pd.isna(pd.Series(group)).sum())
    if n_nan_groups:
        logger.warning(
            "apply_group_zero_sum_constraint: %d row(s) have a NaN group label; pandas groupby excludes them "
            "from the fitted offset table and they receive offset=0.0 (no correction) unchanged.",
            n_nan_groups,
        )

    df = pd.DataFrame({"pred": pred_arr, "w": w_arr, "wpred": pred_arr * w_arr, "group": group})
    agg = df.groupby("group", sort=False).agg(w_sum=("w", "sum"), wpred_sum=("wpred", "sum"))
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(agg["w_sum"].to_numpy() != 0, (agg["wpred_sum"].to_numpy() - target_sum) / agg["w_sum"].to_numpy(), 0.0)
    offset_by_group = pd.Series(offset, index=agg.index)

    group_offset = pd.Series(group).map(offset_by_group).fillna(0.0).to_numpy(dtype=np.float64)
    return np.asarray(pred_arr - group_offset)


def apply_group_zero_sum_constraint_multi(
    predictions: np.ndarray,
    groups: list,
    weights: Optional[np.ndarray] = None,
    target_sums: Optional[list] = None,
    max_iterations: int = 10,
    tol: float = 1e-8,
) -> np.ndarray:
    """Satisfy MULTIPLE simultaneous per-group zero-sum constraints via Dykstra-style alternating projection.

    Real conservation-law targets can carry more than one known grouping that each must independently sum
    (weighted) to a target -- e.g. Optiver's target sums to ~0 both within ``(date_id, seconds_in_bucket)``
    AND, if a second orthogonal grouping is also known to be constrained, within that grouping too. Applying
    :func:`apply_group_zero_sum_constraint` for one grouping alone re-satisfies THAT constraint but generally
    re-breaks any other grouping's constraint (the two group-by partitions overlap, so a per-row shift that
    fixes one recentres rows into different partial sums for the other). Alternating the single-constraint
    projection across all groupings and iterating converges both (all) constraints simultaneously, exactly
    like Dykstra's alternating projection algorithm onto multiple convex sets.

    Parameters
    ----------
    predictions
        ``(n,)`` raw model predictions.
    groups
        List of ``(n,)`` group-label arrays, one per independent zero-sum constraint (e.g.
        ``[group_by_time, group_by_other_key]``).
    weights
        ``(n,)`` per-row weight, or ``None`` for an unweighted (equal-weight) sum/mean within each group.
    target_sums
        Per-constraint known weighted-sum target, one value per entry in ``groups``. Defaults to all-zero.
    max_iterations
        Maximum number of full sweeps over all constraints.
    tol
        Sweep stops early once every constraint's max absolute weighted-sum residual is below ``tol``.

    Returns
    -------
    np.ndarray
        ``(n,)`` corrected predictions satisfying all constraints within ``tol`` (or after ``max_iterations``).
    """
    if not groups:
        return np.asarray(predictions, dtype=np.float64)

    n_constraints = len(groups)
    targets = [0.0] * n_constraints if target_sums is None else list(target_sums)

    corrected = np.asarray(predictions, dtype=np.float64).copy()
    w_arr = None if weights is None else np.asarray(weights, dtype=np.float64)

    for _ in range(max_iterations):
        for group, target_sum in zip(groups, targets):
            corrected = apply_group_zero_sum_constraint(corrected, group, weights=w_arr, target_sum=target_sum)
        # the just-applied (last) constraint is satisfied to float precision by construction -- no need to
        # re-scan it; only the earlier constraints in this sweep may have been disturbed by later applies.
        max_residual = max(
            (_max_abs_group_residual(corrected, group, w_arr, target_sum) for group, target_sum in zip(groups[:-1], targets[:-1])),
            default=0.0,
        )
        if max_residual < tol:
            break

    return np.asarray(corrected)


def _max_abs_group_residual(predictions: np.ndarray, group: np.ndarray, weights: Optional[np.ndarray], target_sum: float) -> float:
    """Return the largest absolute deviation of any group's weighted prediction sum from ``target_sum``."""
    w_arr = np.ones_like(predictions) if weights is None else weights
    df = pd.DataFrame({"wpred": predictions * w_arr, "group": group})
    sums = df.groupby("group", sort=False)["wpred"].sum().to_numpy()
    if sums.size == 0:
        return 0.0
    return float(np.max(np.abs(sums - target_sum)))


__all__ = ["apply_group_zero_sum_constraint", "apply_group_zero_sum_constraint_multi"]
