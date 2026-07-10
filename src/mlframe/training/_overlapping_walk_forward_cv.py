"""Overlapping-window walk-forward CV + a CV-stability-across-seeds diagnostic.

``GroupTimeSeriesSplit`` (feature_selection/wrappers/rfecv) already covers gapped forward-chaining CV with an
EXPANDING train window. G-Research-crypto-forecasting's writeup uses a different, complementary shape: FIXED-
length train windows that overlap and step forward by less than the window length (more folds, lower fold-to-
fold variance than non-overlapping windows, at the cost of correlated folds) plus a train/test gap ("embargo")
to avoid near-boundary leakage. This module adds that fixed-window-overlap splitter, and a companion
diagnostic (``cv_stability_check``) that reruns CV across several seeds/hyperparameter values and flags a
metric curve as untrustworthy when it is not reasonably smooth/monotonic -- catching hyperparameter decisions
made on noise before they're acted on.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np


class OverlappingWalkForwardCV:
    """Fixed-length, overlapping-window walk-forward time-series CV splitter.

    Parameters
    ----------
    window_length
        Number of consecutive time-ordered samples in each train fold.
    step
        How far the train window start advances between consecutive folds; ``step < window_length`` makes
        folds overlap (lower fold-to-fold variance, more folds, correlated folds -- the writeup's tradeoff).
    gap
        Number of samples excluded between the end of the train window and the start of the test window
        ("embargo"), guarding against near-boundary leakage (e.g. autocorrelated/overlapping-label data).
    test_length
        Number of samples in each test fold, immediately after the gap.
    """

    def __init__(self, window_length: int, step: int, gap: int = 0, test_length: int = 1):
        if window_length <= 0 or step <= 0 or test_length <= 0 or gap < 0:
            raise ValueError("OverlappingWalkForwardCV: window_length, step, test_length must be > 0 and gap >= 0")
        self.window_length = window_length
        self.step = step
        self.gap = gap
        self.test_length = test_length

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        train_start = 0
        while True:
            train_end = train_start + self.window_length
            test_start = train_end + self.gap
            test_end = test_start + self.test_length
            if test_end > n_samples:
                break
            yield np.arange(train_start, train_end), np.arange(test_start, test_end)
            train_start += self.step

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            raise ValueError("OverlappingWalkForwardCV.get_n_splits requires X to determine n_samples")
        n_samples = len(X)
        span = self.window_length + self.gap + self.test_length
        if span > n_samples:
            return 0
        return (n_samples - span) // self.step + 1


def cv_stability_check(
    metric_curves: Sequence[Sequence[float]],
    max_sign_change_ratio: float = 0.4,
    min_seeds: int = 2,
) -> dict:
    """Flag a hyperparameter-vs-metric curve as noisy/untrustworthy before acting on it.

    Parameters
    ----------
    metric_curves
        One sequence of metric values per seed/repeat, all evaluated at the SAME ordered hyperparameter grid
        (e.g. ``metric_curves[seed_idx][hp_idx]``). At least ``min_seeds`` curves are required.
    max_sign_change_ratio
        A curve's second-difference sign-change count, divided by its length, above this threshold marks that
        seed's curve as jagged/non-smooth (chasing noise rather than a real trend).
    min_seeds
        Minimum number of seed curves required to assess cross-seed agreement.

    Returns
    -------
    dict
        ``mean_curve`` (elementwise mean across seeds), ``jagged_seed_fraction`` (fraction of seed curves
        flagged jagged), ``cross_seed_argmax_agreement`` (fraction of seed curves whose own argmax lands
        within +-1 grid position of the seed-averaged curve's argmax), ``stable`` (bool: low jaggedness AND
        high cross-seed argmax agreement -- safe to act on the mean curve's optimum).
    """
    curves = np.asarray(metric_curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[0] < min_seeds:
        raise ValueError(f"cv_stability_check: need >= {min_seeds} seed curves of equal length; got shape {curves.shape}")

    mean_curve = curves.mean(axis=0)
    n_points = curves.shape[1]

    jagged_flags = []
    for curve in curves:
        if n_points < 3:
            jagged_flags.append(False)
            continue
        diffs = np.diff(curve)
        signs = np.sign(diffs)
        nonzero = signs[signs != 0]
        if len(nonzero) < 2:
            jagged_flags.append(False)
            continue
        sign_changes = int(np.sum(nonzero[1:] != nonzero[:-1]))
        jagged_flags.append((sign_changes / len(nonzero)) > max_sign_change_ratio)

    mean_argmax = int(np.argmax(mean_curve))
    argmax_agreements = [abs(int(np.argmax(curve)) - mean_argmax) <= 1 for curve in curves]

    jagged_seed_fraction = float(np.mean(jagged_flags))
    cross_seed_argmax_agreement = float(np.mean(argmax_agreements))
    stable = jagged_seed_fraction < 0.5 and cross_seed_argmax_agreement >= 0.5

    return {
        "mean_curve": mean_curve,
        "jagged_seed_fraction": jagged_seed_fraction,
        "cross_seed_argmax_agreement": cross_seed_argmax_agreement,
        "stable": stable,
    }


__all__ = ["OverlappingWalkForwardCV", "cv_stability_check"]
