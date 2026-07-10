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


def rank_percentile_transform(oof_pred: np.ndarray, test_pred: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

    Returns
    -------
    tuple
        ``(oof_percentile, test_percentile)`` -- ``oof_percentile`` is ``(rank - 0.5) / n_oof`` (average-tie
        rank via ``scipy.stats.rankdata``, centered to avoid the 0/1 boundary); ``test_percentile`` is
        ``None`` when ``test_pred`` is ``None``.
    """
    from scipy.stats import rankdata

    oof_pred = np.asarray(oof_pred, dtype=np.float64)
    n_oof = oof_pred.shape[0]
    if n_oof == 0:
        raise ValueError("rank_percentile_transform: oof_pred is empty")

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
