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

from mlframe.training.targets import _max_abs_lag_autocorr


def _resolve_adaptive_gap(
    y_window: np.ndarray,
    base_gap: int,
    autocorr_lags: Tuple[int, ...],
    autocorr_threshold: float,
    max_gap: Optional[int],
) -> int:
    """Widen ``base_gap`` to the largest candidate lag whose |autocorr| on ``y_window`` still exceeds the threshold.

    ``autocorr_lags`` must be ascending. Scans them in order and keeps the last (largest) lag that still clears
    ``autocorr_threshold`` -- for a decaying autocorrelation (AR-like) process this lands near the true decorrelation
    length instead of stopping at the first lag examined. A lag with |autocorr| below threshold does not veto a
    larger lag that happens to spike again (e.g. seasonal series); the scan keeps going through the whole tuple.
    """
    widened = base_gap
    for lag in autocorr_lags:
        if lag <= widened:
            continue
        if y_window.shape[0] < lag + 3:
            break
        corr = _max_abs_lag_autocorr(y_window, lags=(lag,))[0]
        if abs(corr) >= autocorr_threshold:
            widened = lag
    if max_gap is not None:
        widened = min(widened, max_gap)
    return max(widened, base_gap)


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
        ("embargo"), guarding against near-boundary leakage (e.g. autocorrelated/overlapping-label data). Also
        acts as the FLOOR gap when ``adaptive_gap`` is enabled -- the adaptive scheduler only ever widens it.
    test_length
        Number of samples in each test fold, immediately after the gap.
    adaptive_gap
        Opt-in. When True, ``split`` requires ``y`` and, per fold, measures the train window's own label
        autocorrelation (reusing ``_max_abs_lag_autocorr``) and widens that fold's embargo past ``gap`` up to
        the largest lag in ``autocorr_lags`` whose |autocorr| still clears ``autocorr_threshold`` -- instead of
        requiring the caller to hand-tune a single fixed ``gap`` for the whole series. Default False leaves
        ``split``/``get_n_splits`` bit-identical to the pre-existing fixed-gap behavior.
    autocorr_lags
        Ascending candidate lags scanned by the adaptive scheduler (ignored unless ``adaptive_gap`` is True).
    autocorr_threshold
        A lag's |autocorr| at or above this value is treated as meaningful residual leakage risk.
    max_extra_gap
        Optional hard cap on how far the adaptive scheduler may widen the gap above ``gap`` (``None`` = uncapped,
        bounded only by ``max(autocorr_lags)``); guards against a pathological near-unit-root window eating the
        whole window budget.
    """

    def __init__(
        self,
        window_length: int,
        step: int,
        gap: int = 0,
        test_length: int = 1,
        adaptive_gap: bool = False,
        autocorr_lags: Tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20, 25, 30),
        autocorr_threshold: float = 0.2,
        max_extra_gap: Optional[int] = None,
    ):
        if window_length <= 0 or step <= 0 or test_length <= 0 or gap < 0:
            raise ValueError("OverlappingWalkForwardCV: window_length, step, test_length must be > 0 and gap >= 0")
        if adaptive_gap and (not autocorr_lags or list(autocorr_lags) != sorted(autocorr_lags)):
            raise ValueError("OverlappingWalkForwardCV: autocorr_lags must be a non-empty ascending tuple")
        self.window_length = window_length
        self.step = step
        self.gap = gap
        self.test_length = test_length
        self.adaptive_gap = adaptive_gap
        self.autocorr_lags = autocorr_lags
        self.autocorr_threshold = autocorr_threshold
        self.max_extra_gap = max_extra_gap

    def _max_gap_cap(self) -> Optional[int]:
        """Upper bound on the adaptive gap, or None (uncapped) when ``max_extra_gap`` is unset."""
        return None if self.max_extra_gap is None else self.gap + self.max_extra_gap

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) walk-forward windows, widening the gap adaptively if configured."""
        if not self.adaptive_gap:
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
            return

        if y is None:
            raise ValueError("OverlappingWalkForwardCV.split: adaptive_gap=True requires y (labels) to measure autocorrelation")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        n_samples = len(X)
        max_gap_cap = self._max_gap_cap()
        train_start = 0
        while True:
            train_end = train_start + self.window_length
            if train_end > n_samples:
                break
            effective_gap = _resolve_adaptive_gap(y_arr[train_start:train_end], self.gap, self.autocorr_lags, self.autocorr_threshold, max_gap_cap)
            test_start = train_end + effective_gap
            test_end = test_start + self.test_length
            if test_end > n_samples:
                break
            yield np.arange(train_start, train_end), np.arange(test_start, test_end)
            train_start += self.step

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of splits this CV would yield for X (fixed-gap count; adaptive_gap uses the same window arithmetic)."""
        if X is None:
            raise ValueError("OverlappingWalkForwardCV.get_n_splits requires X to determine n_samples")
        if not self.adaptive_gap:
            n_samples = len(X)
            span = self.window_length + self.gap + self.test_length
            if span > n_samples:
                return 0
            return (n_samples - span) // self.step + 1
        # Per-fold gap varies with the adaptive scheduler, so the closed-form fixed-gap formula does not apply --
        # count actual folds directly (still cheap: this class targets moderate n_samples / n_folds).
        return sum(1 for _ in self.split(X, y=y, groups=groups))


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
