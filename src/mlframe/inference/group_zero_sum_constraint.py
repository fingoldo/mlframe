"""Group weighted-sum post-processing constraint: correct predictions to satisfy a known conservation identity.

Some targets have a known linear identity by construction -- market-neutral/index-relative returns sum to
zero across all constituents at a given instant (a 9th-place Optiver-trading writeup's exact case:
weighted-sum-of-target-across-stocks == 0 per ``(date_id, seconds_in_bucket)``), or compositional shares sum
to a known constant. A model's raw predictions rarely satisfy this exactly; this post-processor corrects each
group's predictions minimally (in a weighted-least-squares sense) to enforce the constraint, by subtracting
the group's weighted-mean prediction (the unique minimal correction that zeroes the weighted sum).

Some targets carry TWO (or more) simultaneous known linear identities per group -- e.g. a zero-sum identity
plus a known ratio/bound between a subset of members and an external control total. ``extra_constraint_coefs``
/ ``extra_constraint_targets`` generalize the single constant-shift correction to a constrained weighted-least-
squares projection satisfying all constraints simultaneously (Lagrangian stationarity under the same weighted
metric as the base constraint), while staying bit-identical to the original single-constraint formula when
omitted.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _weighted_pava(values: np.ndarray, weights: np.ndarray, ascending: bool) -> np.ndarray:
    """Pool-adjacent-violators weighted isotonic regression, in the given order (not group-aware -- caller slices per group)."""
    v = values if ascending else values[::-1]
    w = weights if ascending else weights[::-1]
    # blocks: running (weighted_mean, weight_sum, count) stack; merge back-to-front whenever monotonicity is violated.
    block_means = []
    block_weights = []
    block_counts = []
    for vi, wi in zip(v, w):
        block_means.append(float(vi))
        block_weights.append(float(wi))
        block_counts.append(1)
        while len(block_means) > 1 and block_means[-2] > block_means[-1]:
            w_merged = block_weights[-2] + block_weights[-1]
            mean_merged = (block_means[-2] * block_weights[-2] + block_means[-1] * block_weights[-1]) / w_merged if w_merged > 0 else 0.0
            n_merged = block_counts[-2] + block_counts[-1]
            block_means.pop()
            block_weights.pop()
            block_counts.pop()
            block_means[-1] = mean_merged
            block_weights[-1] = w_merged
            block_counts[-1] = n_merged
    out = np.empty(v.shape[0], dtype=np.float64)
    pos = 0
    for mean, count in zip(block_means, block_counts):
        out[pos : pos + count] = mean
        pos += count
    return out if ascending else out[::-1]


def apply_group_zero_sum_constraint(
    preds: np.ndarray,
    group_ids: np.ndarray,
    weights: Optional[np.ndarray] = None,
    target_sum: float = 0.0,
    extra_constraint_coefs: Optional[Sequence[np.ndarray]] = None,
    extra_constraint_targets: Optional[Sequence[float]] = None,
    preserve_rank_order: bool = False,
) -> np.ndarray:
    """Correct ``preds`` so each group's weighted sum equals ``target_sum`` (plus any extra linear constraints).

    Parameters
    ----------
    preds
        ``(n,)`` raw model predictions.
    group_ids
        ``(n,)`` grouping key (e.g. a combined date/time-bucket id) -- the constraint is enforced
        independently within each group.
    weights
        Optional ``(n,)`` per-row weights (e.g. market cap); defaults to equal weights (``1.0`` each), in
        which case the correction reduces to subtracting each group's plain mean. Also doubles as the metric
        of the weighted-least-squares projection when ``extra_constraint_coefs`` is given.
    target_sum
        The known weighted-sum identity's target value (``0.0`` for a market-neutral/zero-sum target).
    extra_constraint_coefs
        Opt-in: additional per-row coefficient vectors (each ``(n,)``), one per extra linear constraint, e.g. a
        second grouping/known-ratio weighting whose group sum must simultaneously match a control number. When
        given, the correction generalizes from a single constant per-group shift to a constrained weighted-
        least-squares projection satisfying every constraint at once (Lagrangian stationarity under the
        ``weights`` metric) -- the base constraint (``weights``, ``target_sum``) is always included as the
        first row of the system, so passing ``None`` here is exactly the original single-constraint behavior
        (same code path, bit-identical output).
    extra_constraint_targets
        Target values matching ``extra_constraint_coefs`` one-to-one; required (same length) when
        ``extra_constraint_coefs`` is given.
    preserve_rank_order
        Opt-in, only meaningful with ``extra_constraint_coefs`` (a single constraint's correction is already a
        per-group constant, so relative order never changes). When multiple constraints are active, the
        per-row correction is no longer constant within a group and CAN reorder members. If ``True``, the
        projected result is passed through a weighted isotonic regression (matching the original ``preds``
        rank order) and then re-projected onto the exact same constraint system -- this is a heuristic that
        keeps both constraints exactly satisfied while pulling the result back towards the original order; it
        is not a global guarantee against reordering in pathological cases (e.g. two constraints that are
        fundamentally incompatible with the original order).

    Returns
    -------
    np.ndarray
        ``(n,)`` corrected predictions satisfying every active constraint exactly (to float precision) in
        every group.
    """
    preds = np.asarray(preds, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    weights = np.ones_like(preds) if weights is None else np.asarray(weights, dtype=np.float64)
    if not (preds.shape == group_ids.shape == weights.shape):
        raise ValueError("apply_group_zero_sum_constraint: preds, group_ids, weights must share the same shape")

    _uniq, group_codes = np.unique(group_ids, return_inverse=True)
    n_groups = _uniq.shape[0]

    if extra_constraint_coefs is None:
        # np.unique + bincount instead of pandas groupby().transform("sum") (profiled 2x, one call per sum):
        # pandas groupby machinery (code-factorization, cython dispatch) is rebuilt from scratch on EACH
        # transform call despite both calls sharing the same grouping -- bincount on the shared integer group
        # codes computes both weighted sums in one factorization pass, ~5x faster at 1M rows (measured in
        # bench_group_zero_sum_constraint.py).
        group_weighted_sum_by_code = np.bincount(group_codes, weights=preds * weights, minlength=n_groups)
        group_weight_sum_by_code = np.bincount(group_codes, weights=weights, minlength=n_groups)

        with np.errstate(invalid="ignore", divide="ignore"):
            correction_by_code = np.where(group_weight_sum_by_code > 0, (group_weighted_sum_by_code - target_sum) / group_weight_sum_by_code, 0.0)
        correction = correction_by_code[group_codes]

        return np.asarray(preds - correction, dtype=np.float64)

    if extra_constraint_targets is None or len(extra_constraint_targets) != len(extra_constraint_coefs):
        raise ValueError("apply_group_zero_sum_constraint: extra_constraint_targets must match extra_constraint_coefs in length")

    coefs = [weights] + [np.asarray(c, dtype=np.float64) for c in extra_constraint_coefs]
    for c in coefs:
        if c.shape != preds.shape:
            raise ValueError("apply_group_zero_sum_constraint: every constraint coefficient array must share preds' shape")
    targets = [target_sum] + list(extra_constraint_targets)
    n_constraints = len(coefs)

    with np.errstate(invalid="ignore", divide="ignore"):
        inv_weights = np.where(weights != 0, 1.0 / weights, 0.0)

    # Lagrangian stationarity of "minimize sum(weights * correction^2) s.t. C @ correction = r" gives
    # correction_i = inv_weights_i * sum_k lambda_k * coefs_k_i, with per-group M @ lambda = r,
    # M[j, k] = sum_i coefs_j_i * coefs_k_i * inv_weights_i, r[j] = sum_i coefs_j_i * preds_i - target_j.
    # K is small (a handful of known constraints, not a per-row degree of freedom), so the K x K Gram matrix
    # is built via one bincount per (j, k) pair and solved batched over all groups at once via np.linalg.pinv
    # (never a Python loop over groups) -- pinv tolerates a singular/degenerate group (e.g. all-zero weights)
    # the same way the single-constraint path's np.where guard does, returning a zero correction there.
    M = np.zeros((n_groups, n_constraints, n_constraints), dtype=np.float64)
    r = np.zeros((n_groups, n_constraints), dtype=np.float64)
    for j in range(n_constraints):
        r[:, j] = np.bincount(group_codes, weights=coefs[j] * preds, minlength=n_groups) - targets[j]
        for k in range(j, n_constraints):
            m_jk = np.bincount(group_codes, weights=coefs[j] * coefs[k] * inv_weights, minlength=n_groups)
            M[:, j, k] = m_jk
            M[:, k, j] = m_jk

    lambda_by_group = np.einsum("gjk,gk->gj", np.linalg.pinv(M), r)
    lambda_per_row = lambda_by_group[group_codes]  # (n, n_constraints)
    coefs_per_row = np.stack(coefs, axis=1)  # (n, n_constraints)
    correction = inv_weights * np.einsum("nk,nk->n", lambda_per_row, coefs_per_row)
    corrected = np.asarray(preds - correction, dtype=np.float64)

    if not preserve_rank_order:
        return corrected

    # Isotonic re-fit towards the ORIGINAL preds order, then re-project onto the same constraint system so the
    # re-fit result (which can drift off the constraints) is snapped back to exactly satisfying them.
    iso_target = corrected.copy()
    order = np.argsort(preds, kind="stable")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.shape[0])
    for code in range(n_groups):
        member_mask = group_codes == code
        if member_mask.sum() < 2:
            continue
        member_idx = np.flatnonzero(member_mask)
        local_order = member_idx[np.argsort(preds[member_idx], kind="stable")]
        iso_vals = _weighted_pava(corrected[local_order], weights[local_order], ascending=True)
        iso_target[local_order] = iso_vals

    r_iso = np.zeros((n_groups, n_constraints), dtype=np.float64)
    for j in range(n_constraints):
        r_iso[:, j] = np.bincount(group_codes, weights=coefs[j] * iso_target, minlength=n_groups) - targets[j]
    lambda_iso_by_group = np.einsum("gjk,gk->gj", np.linalg.pinv(M), r_iso)
    lambda_iso_per_row = lambda_iso_by_group[group_codes]
    correction_iso = inv_weights * np.einsum("nk,nk->n", lambda_iso_per_row, coefs_per_row)
    return np.asarray(iso_target - correction_iso, dtype=np.float64)


__all__ = ["apply_group_zero_sum_constraint"]
