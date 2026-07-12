"""Rank-percentile transform for base-learner OOF/test predictions before stacking.

Base learners in a stacking ensemble are often miscalibrated relative to EACH OTHER (different loss
functions, different amounts of regularization, different score scales) even when each is individually a
good ranker. A meta-learner trained on raw miscalibrated scores can end up worse than the single best base
model -- a failure mode a Home-Credit 5th-place commenter diagnosed and fixed by converting every base
learner's OOF predictions (and test predictions, consistently) to their RANK PERCENTILE before stacking,
putting every base learner on the same [0, 1] scale regardless of its raw score's distribution/calibration.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _gaussian_smoothed_percentile(query: np.ndarray, oof_pred: np.ndarray, bandwidth: float) -> np.ndarray:
    """Kernel-smoothed empirical CDF of ``oof_pred`` evaluated at ``query`` (Gaussian kernel, bandwidth ``h``).

    ``F_h(x) = mean_j Phi((x - x_j) / h)`` -- a continuous alternative to the hard step-function ECDF that the
    interpolated-rank path implicitly uses. Each reference point contributes a smeared-out (not step) increment,
    which trades a small amount of bias for a large cut in sampling variance when ``n_oof`` is small, especially
    near the tails where the hard ECDF has few neighbors to interpolate between. O(n_query * n_oof) -- intended
    for the small-reference-set regime this mode targets, not the million-row bulk path.
    """
    from scipy.special import ndtr

    z = (query[:, None] - oof_pred[None, :]) / bandwidth
    return np.asarray(ndtr(z).mean(axis=1), dtype=np.float64)


def rank_percentile_transform(
    oof_pred: np.ndarray, test_pred: Optional[np.ndarray] = None, smoothing: Optional[float] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert one base learner's OOF predictions (and optionally test predictions) to rank percentiles.

    Parameters
    ----------
    oof_pred
        ``(n_oof,)`` out-of-fold predictions from one base learner.
    test_pred
        Optional ``(n_test,)`` predictions from the SAME base learner on held-out/test rows; ranked against
        the OOF distribution (leak-safe: fit only reads ``oof_pred``) via interpolated rank against the
        sorted OOF values, so a test value between two OOF values gets an interpolated percentile and a
        test value beyond the OOF range clamps to the nearest extreme.
    smoothing
        Opt-in kernel-smoothed mode. ``None`` (default) keeps the exact original hard interpolated-rank
        behavior, bit-identical. When set to a positive float, it is a bandwidth MULTIPLIER of
        ``std(oof_pred)`` (e.g. ``0.3``) used to Gaussian-kernel-smooth the empirical CDF instead of using
        the raw step-function rank -- this reduces discretization/sampling noise near the distribution's
        edges when ``n_oof`` is small (the hard rank has only ``1/n_oof`` resolution there). Raises
        ``ValueError`` if not strictly positive.

    Returns
    -------
    tuple
        ``(oof_percentile, test_percentile)`` -- in hard mode (``smoothing=None``) ``oof_percentile`` is
        ``(rank - 0.5) / n_oof`` (average-tie rank via ``scipy.stats.rankdata``, centered to avoid the 0/1
        boundary); in smoothed mode it is the Gaussian-kernel-smoothed ECDF evaluated at each OOF point.
        ``test_percentile`` is ``None`` when ``test_pred`` is ``None``.
    """
    from scipy.stats import rankdata

    oof_pred = np.asarray(oof_pred, dtype=np.float64)
    n_oof = oof_pred.shape[0]
    if n_oof == 0:
        raise ValueError("rank_percentile_transform: oof_pred is empty")

    if smoothing is not None:
        if smoothing <= 0:
            raise ValueError(f"rank_percentile_transform: smoothing must be strictly positive, got {smoothing}")
        std = np.std(oof_pred)
        bandwidth = smoothing * std if std > 0 else smoothing
        oof_percentile = _gaussian_smoothed_percentile(oof_pred, oof_pred, bandwidth)
        oof_percentile = np.clip(oof_percentile, 0.0, 1.0)

        if test_pred is None:
            return oof_percentile, None

        test_pred = np.asarray(test_pred, dtype=np.float64)
        test_percentile = _gaussian_smoothed_percentile(test_pred, oof_pred, bandwidth)
        test_percentile = np.clip(test_percentile, 0.0, 1.0)
        return oof_percentile, test_percentile

    oof_ranks = rankdata(oof_pred, method="average")
    oof_percentile = (oof_ranks - 0.5) / n_oof

    if test_pred is None:
        return oof_percentile, None

    test_pred = np.asarray(test_pred, dtype=np.float64)
    sorted_oof = np.sort(oof_pred)
    # interpolated rank position of each test value against the sorted OOF distribution, mapped to the same
    # [0, 1] percentile scale (average of left/right insertion position -> handles OOF-value ties gracefully).
    left = np.searchsorted(sorted_oof, test_pred, side="left")
    right = np.searchsorted(sorted_oof, test_pred, side="right")
    test_rank = (left + right) / 2.0
    test_percentile = (test_rank + 0.5) / n_oof
    test_percentile = np.clip(test_percentile, 0.0, 1.0)

    return oof_percentile, test_percentile


__all__ = ["rank_percentile_transform"]
