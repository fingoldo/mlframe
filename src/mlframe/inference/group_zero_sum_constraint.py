"""Group weighted-sum post-processing constraint: correct predictions to satisfy a known conservation identity.

Some targets have a known linear identity by construction -- market-neutral/index-relative returns sum to
zero across all constituents at a given instant (a 9th-place Optiver-trading writeup's exact case:
weighted-sum-of-target-across-stocks == 0 per ``(date_id, seconds_in_bucket)``), or compositional shares sum
to a known constant. A model's raw predictions rarely satisfy this exactly; this post-processor corrects each
group's predictions minimally (in a weighted-least-squares sense) to enforce the constraint, by subtracting
the group's weighted-mean prediction (the unique minimal correction that zeroes the weighted sum).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def apply_group_zero_sum_constraint(
    preds: np.ndarray,
    group_ids: np.ndarray,
    weights: Optional[np.ndarray] = None,
    target_sum: float = 0.0,
) -> np.ndarray:
    """Correct ``preds`` so each group's weighted sum equals ``target_sum``.

    Parameters
    ----------
    preds
        ``(n,)`` raw model predictions.
    group_ids
        ``(n,)`` grouping key (e.g. a combined date/time-bucket id) -- the constraint is enforced
        independently within each group.
    weights
        Optional ``(n,)`` per-row weights (e.g. market cap); defaults to equal weights (``1.0`` each), in
        which case the correction reduces to subtracting each group's plain mean.
    target_sum
        The known weighted-sum identity's target value (``0.0`` for a market-neutral/zero-sum target).

    Returns
    -------
    np.ndarray
        ``(n,)`` corrected predictions: ``preds[i] - (group_weighted_sum - target_sum) / group_weight_sum``
        for row ``i``'s group -- the minimal (weighted-least-squares-optimal) per-row correction that exactly
        satisfies the constraint in every group.
    """
    preds = np.asarray(preds, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    weights = np.ones_like(preds) if weights is None else np.asarray(weights, dtype=np.float64)
    if not (preds.shape == group_ids.shape == weights.shape):
        raise ValueError("apply_group_zero_sum_constraint: preds, group_ids, weights must share the same shape")

    # np.unique + bincount instead of pandas groupby().transform("sum") (profiled 2x, one call per sum):
    # pandas groupby machinery (code-factorization, cython dispatch) is rebuilt from scratch on EACH
    # transform call despite both calls sharing the same grouping -- bincount on the shared integer group
    # codes computes both weighted sums in one factorization pass, ~5x faster at 1M rows (measured in
    # bench_group_zero_sum_constraint.py).
    _uniq, group_codes = np.unique(group_ids, return_inverse=True)
    n_groups = _uniq.shape[0]
    group_weighted_sum_by_code = np.bincount(group_codes, weights=preds * weights, minlength=n_groups)
    group_weight_sum_by_code = np.bincount(group_codes, weights=weights, minlength=n_groups)

    with np.errstate(invalid="ignore", divide="ignore"):
        correction_by_code = np.where(
            group_weight_sum_by_code > 0, (group_weighted_sum_by_code - target_sum) / group_weight_sum_by_code, 0.0
        )
    correction = correction_by_code[group_codes]

    return np.asarray(preds - correction, dtype=np.float64)


__all__ = ["apply_group_zero_sum_constraint"]
