"""Empirical conditional-independence check for ``odds_ratio_combine`` members.

Log-odds summation (``odds_ratio_combine``) is only the correct combination rule when member
sub-models are conditionally independent given y. When members instead carry substantially
redundant signal (e.g. two sub-models both largely derived from the same upstream feature, or
noisy near-duplicate re-estimates of the same latent effect), summing their logits double-counts
that shared evidence -> systematically over-confident (too extreme) combined probabilities.

This module estimates that redundancy empirically from the member probabilities alone (no ground
truth needed): for each member, correlate its logit against the leave-one-out mean logit of the
OTHER members (their "consensus"). A genuinely conditionally-independent member's logit is only
correlated with the other members' consensus through their shared dependence on y, which is a
real but bounded effect; a redundant/near-duplicate member's logit tracks the consensus far more
tightly because both are measuring (up to noise) the same underlying quantity.

NOTE: naively regressing a member's logit on the consensus and looking at the RESIDUAL correlation
(an earlier version of this check) does not work -- for a clean near-duplicate case the consensus
already explains nearly all of the shared factor, so the residual is left with only the (by
construction independent) per-member noise and reports near-zero correlation despite the members
being almost entirely redundant. The member-vs-consensus correlation itself (not its residual) is
the quantity that actually separates redundant members from conditionally-independent ones.
"""
from __future__ import annotations

import numpy as np


def _member_consensus_correlations(logits: np.ndarray) -> np.ndarray:
    """Per-member Pearson correlation between the member's logit and the leave-one-out mean logit of the rest.

    Closed-form via sufficient statistics rather than a per-member O(n) Python loop: the leave-one-out mean
    for member j, ``(row_sum - logits[:, j]) / (k - 1)``, is an affine function of ``logits[:, j]`` and the
    row sum, so its correlation with ``logits[:, j]`` reduces to column sums/sums-of-squares plus a single
    ``logits.T @ row_sum`` matrix-vector product -- one BLAS-backed O(n*k) pass instead of k separate O(n)
    reductions.
    """
    n, k = logits.shape
    row_sum = logits.sum(axis=1)
    col_sum = logits.sum(axis=0)
    col_sumsq = np.sum(logits * logits, axis=0)
    sum_aS = logits.T @ row_sum
    sum_S = row_sum.sum()
    sum_S2 = float(np.dot(row_sum, row_sum))

    denom_loo = k - 1
    sum_b = (sum_S - col_sum) / denom_loo
    sum_b2 = (sum_S2 - 2.0 * sum_aS + col_sumsq) / (denom_loo * denom_loo)
    sum_ab = (sum_aS - col_sumsq) / denom_loo

    mean_a = col_sum / n
    mean_b = sum_b / n
    cov_ab = sum_ab / n - mean_a * mean_b
    var_a = col_sumsq / n - mean_a * mean_a
    var_b = sum_b2 / n - mean_b * mean_b

    denom = np.sqrt(np.clip(var_a * var_b, 0.0, None))
    return np.asarray(np.divide(cov_ab, denom, out=np.zeros(k, dtype=np.float64), where=denom > 1e-300))


def member_residual_correlation(member_probs: np.ndarray, clip: float = 1e-7) -> dict:
    """Estimate member-vs-consensus correlation of member log-odds, as a redundancy/independence diagnostic.

    Parameters
    ----------
    member_probs
        ``(n_samples, n_members)`` array of P(y=1) member estimates (same contract as
        ``odds_ratio_combine``). Requires at least 2 members.
    clip
        Probability clip applied before the logit transform (same default as ``odds_ratio_combine``).

    Returns
    -------
    dict
        ``mean_abs_residual_correlation`` / ``max_abs_residual_correlation``: summary statistics over
        members of ``|corr(member_j_logit, leave_one_out_mean_logit_of_the_rest)|`` (0 = a member's
        logit carries no linear relationship to what the other members already say, ~1 = the member
        is essentially a redundant re-measurement of the same signal as the rest) used to flag
        conditional-independence violations. ``per_member_consensus_correlation``: the raw signed
        per-member ``(k,)`` correlation vector, for callers that want the detail.
    """
    p = np.asarray(member_probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"member_residual_correlation: member_probs must be 2D (n_samples, n_members); got shape {p.shape}")
    _n, k = p.shape
    if k < 2:
        return {"mean_abs_residual_correlation": 0.0, "max_abs_residual_correlation": 0.0, "per_member_consensus_correlation": np.zeros(k)}

    p_c = np.clip(p, clip, 1.0 - clip)
    logits = np.log(p_c / (1.0 - p_c))
    per_member = _member_consensus_correlations(logits)
    per_member = np.nan_to_num(per_member, nan=0.0)  # a constant-logit member (denom~0) yields nan -> treat as 0

    mean_abs = float(np.mean(np.abs(per_member)))
    max_abs = float(np.max(np.abs(per_member)))
    return {"mean_abs_residual_correlation": mean_abs, "max_abs_residual_correlation": max_abs, "per_member_consensus_correlation": per_member}


__all__ = ["member_residual_correlation"]
