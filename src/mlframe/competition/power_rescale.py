"""COMPETITION/EXPLORATORY-ONLY utilities. NOT for production use.

Implements two Kaggle-competition-specific post-hoc rescaling tricks documented in
``MLFRAME_IDEAS_competitions.md``:

1. ``power_rescale_to_target_sum`` — exponent-based probability rescaling that solves
   for an exponent ``p`` such that ``sum(probs ** p) == target_sum``, matching a
   known/estimated positive count (e.g. extrapolated from a public leaderboard).
   Unlike linear/additive rescaling, raising to a power preserves rank order and
   reshapes mostly the low-probability tail.

2. ``asymmetric_scale_by_sign`` — grid-searched asymmetric scale factor applied
   separately to positive vs. negative predictions of a signed/return-like target,
   selected via a 1-D CV sweep over a user-supplied metric function.

Both rely on information that has no production analog (a known/extrapolated true
positive count from leaderboard feedback; a metric-sweep-fit scalar tuned directly
against a validation target). They are useful for Kaggle-style competitions only.
Never import this module from production mlframe code paths and never wire it into
any default pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq


def power_rescale_to_target_sum(probs: np.ndarray, target_sum: float) -> np.ndarray:
    """COMPETITION-ONLY. Not for production use.

    Rescale ``probs`` by raising each element to a common exponent ``p`` such that
    ``sum(probs ** p) == target_sum``, solved via Brent's method (bisection-based
    root finding on a monotonic function of ``p``).

    This is the "exponent-based probability rescaling to match expected positive
    count" trick (2nd place, santander-product-recommendation): the exponent is
    chosen so the *sum* of transformed probabilities matches a known/extrapolated
    true positive count. Because ``x ** p`` is a strictly monotonic transform of
    ``x`` for ``x`` in ``(0, 1)`` and any real ``p``, rank order among probabilities
    is exactly preserved (only isotonic reshaping, unlike linear/additive rescaling
    or Platt/isotonic calibration which can cross ranks).

    Requires ``0 < target_sum < len(probs)`` and all ``probs`` strictly within
    ``(0, 1)`` (exclusive) for the exponent to be well defined and finite; values
    of exactly 0 or 1 are clipped away from the boundary to keep the solve stable.

    Parameters
    ----------
    probs:
        1-D array of predicted probabilities, values expected in ``[0, 1]``.
    target_sum:
        Desired sum of the rescaled probabilities (e.g. an estimated/extrapolated
        expected positive count for the dataset).

    Returns
    -------
    np.ndarray
        ``probs ** p`` for the solved exponent ``p``, same shape/dtype as ``probs``
        cast to float64.

    Raises
    ------
    ValueError
        If ``probs`` is empty, contains values outside ``[0, 1]``, or ``target_sum``
        is not strictly between 0 and ``len(probs)`` (the sum's achievable range as
        ``p -> +inf`` gives 0 and as ``p -> -inf`` (or p -> 0) gives ``len(probs)``).
    """
    p_arr = np.asarray(probs, dtype=np.float64)
    if p_arr.ndim != 1:
        raise ValueError(f"probs must be 1-D, got shape {p_arr.shape}")
    if p_arr.size == 0:
        raise ValueError("probs must be non-empty")
    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError("all probs must lie within [0, 1]")
    n = p_arr.size
    if not (0.0 < target_sum < n):
        raise ValueError(f"target_sum must satisfy 0 < target_sum < len(probs)={n}, got {target_sum}")

    # Keep strictly inside (0, 1) so p ** anything stays finite and monotonic.
    eps = 1e-12
    p_clipped = np.clip(p_arr, eps, 1.0 - eps)

    def sum_at_exponent(exponent: float) -> float:
        return float(np.sum(np.power(p_clipped, exponent))) - target_sum

    # sum_at_exponent is strictly decreasing in exponent: at exponent -> -inf the sum
    # explodes to +inf (since p_clipped < 1), at exponent -> +inf it goes to 0.
    lo, hi = -1.0, 1.0
    max_expand = 200
    expand_iters = 0
    while sum_at_exponent(lo) < 0.0 and expand_iters < max_expand:
        lo *= 2.0
        expand_iters += 1
    expand_iters = 0
    while sum_at_exponent(hi) > 0.0 and expand_iters < max_expand:
        hi *= 2.0
        expand_iters += 1

    exponent = brentq(sum_at_exponent, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    return np.asarray(np.power(p_clipped, exponent), dtype=np.float64)


def asymmetric_scale_by_sign(
    preds: np.ndarray,
    cv_metric_fn: Callable[[np.ndarray], float],
    scale_range: tuple[float, float] = (1.0, 2.0),
    n_steps: int = 101,
) -> tuple[np.ndarray, float]:
    """COMPETITION-ONLY. Not for production use.

    Grid-searched asymmetric scaling of positive vs. negative predictions for a
    signed/return-like target. This is the "bull"/"bear" submission-variant trick
    (8th place, ubiquant-market-prediction): a scalar ``s`` in ``scale_range`` is
    swept and, for each candidate, positive predictions are divided by ``s`` while
    negative predictions are multiplied by ``s`` (an asymmetric transform that
    shrinks the majority-sign magnitude relative to the minority sign, or vice
    versa depending on which side of 1.0 the optimum falls on); the ``s`` maximizing
    ``cv_metric_fn`` on the rescaled predictions is kept.

    This tunes a scalar directly against a CV metric on the very predictions being
    scaled, which risks overfitting the sweep to validation noise / masking a real
    calibration issue rather than fixing it — acceptable for leaderboard-style
    competitions, not for production calibration.

    Parameters
    ----------
    preds:
        1-D array of signed predictions (e.g. predicted returns).
    cv_metric_fn:
        Callable taking the rescaled predictions array and returning a scalar score
        to be *maximized* (e.g. a correlation or a CV-computed metric closing over
        held-out targets).
    scale_range:
        ``(low, high)`` inclusive bounds for the scale factor sweep.
    n_steps:
        Number of grid points (inclusive of both ends) to evaluate.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(rescaled_preds, best_scale)`` — the predictions rescaled with the best
        found scale factor, and that scale factor itself.

    Raises
    ------
    ValueError
        If ``preds`` is empty, ``scale_range`` is not a valid ascending pair with
        strictly positive bounds, or ``n_steps < 2``.
    """
    preds_arr = np.asarray(preds, dtype=np.float64)
    if preds_arr.ndim != 1:
        raise ValueError(f"preds must be 1-D, got shape {preds_arr.shape}")
    if preds_arr.size == 0:
        raise ValueError("preds must be non-empty")
    lo, hi = scale_range
    if not (0.0 < lo <= hi):
        raise ValueError(f"scale_range must satisfy 0 < low <= high, got {scale_range}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")

    is_pos = preds_arr > 0.0
    is_neg = preds_arr < 0.0

    best_scale = 1.0
    best_score = -np.inf
    for scale in np.linspace(lo, hi, n_steps):
        candidate = preds_arr.copy()
        candidate[is_pos] = candidate[is_pos] / scale
        candidate[is_neg] = candidate[is_neg] * scale
        score = cv_metric_fn(candidate)
        if score > best_score:
            best_score = score
            best_scale = float(scale)

    rescaled = preds_arr.copy()
    rescaled[is_pos] = rescaled[is_pos] / best_scale
    rescaled[is_neg] = rescaled[is_neg] * best_scale
    return rescaled, best_scale
