"""Distribution-mode + variance-ratio + target-type classifier helpers
carved out of ``mlframe.training._target_distribution_analyzer``.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training._target_distribution_analyzer import _detect_multi_modal``
resolves transparently.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

# Parent-resident threshold constants used as default-arg values in
# ``_detect_multi_modal``. The parent defines these BEFORE its bottom-of-
# module sibling import, so this static cycle resolves at runtime in the
# canonical import order.
from ._target_distribution_analyzer import (
    _MULTI_MODAL_KDE_BINS,
    _MULTI_MODAL_MIN_PEAK_SEP_STDS,
    _MULTI_MODAL_VALLEY_DEPTH_RATIO,
)


def _detect_multi_modal(y: np.ndarray, n_bins: int = _MULTI_MODAL_KDE_BINS,
                        min_peak_sep_stds: float = _MULTI_MODAL_MIN_PEAK_SEP_STDS,
                        valley_depth_ratio: float = _MULTI_MODAL_VALLEY_DEPTH_RATIO,
                        sigma: "float | None" = None) -> tuple[bool, int, float]:
    """Detect >= 2 well-separated peaks via smoothed histogram + antimode check.

    A unimodal but noisy histogram can grow many local maxima from binning
    noise; counting them naively flags gaussian samples as multi-modal. The
    correct test is the *antimode* check: a true bimodal distribution has
    a deep valley BETWEEN the two peaks. We require:

    1. Two distinct local maxima separated by >= ``min_peak_sep_stds`` * std.
    2. The minimum bin between them drops to <= ``valley_depth_ratio`` *
       min(peak_a, peak_b). 0.7 means the antimode must be at least 30%
       below the lower of the two peaks -- gaussian sampling noise rarely
       creates valleys that deep across a wide separation.

    Aggressive smoothing (binomial kernel applied twice) suppresses bin-by-bin
    noise without erasing genuine separations.

    Returns (is_multi_modal, n_peaks_above_min_height, max_qualified_separation).
    """
    if y.size < 50:
        return False, 0, 0.0
    # Reuse the caller's already-computed std when supplied (the analyzer computes it once for the moment stats) to skip
    # a redundant full-n np.std pass here; np.std(y) recomputed gives the identical float, so this is bit-identical.
    sigma = float(np.std(y)) if sigma is None else float(sigma)
    if sigma <= 0.0:
        return False, 0, 0.0
    hist, edges = np.histogram(y, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    # Aggressive smoothing: two passes of binomial(5) kernel kills bin-by-bin noise.
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    smoothed = np.convolve(hist.astype(np.float64), kernel, mode="same")
    smoothed = np.convolve(smoothed, kernel, mode="same")
    if smoothed.max() <= 0:
        return False, 0, 0.0
    min_height = 0.05 * float(smoothed.max())
    peaks: list[int] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= min_height and smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return False, len(peaks), 0.0
    # Antimode-qualified pairs: for each pair, check separation in std-units AND
    # valley-depth between them. Only such pairs count as evidence of multi-modality.
    max_qualified_sep = 0.0
    for ii in range(len(peaks)):
        for jj in range(ii + 1, len(peaks)):
            pi, pj = peaks[ii], peaks[jj]
            sep = abs(centres[pj] - centres[pi]) / sigma
            if sep < min_peak_sep_stds:
                continue
            lo, hi = min(pi, pj), max(pi, pj)
            valley = float(smoothed[lo:hi + 1].min())
            lower_peak = float(min(smoothed[pi], smoothed[pj]))
            if lower_peak <= 0:
                continue
            if valley / lower_peak <= valley_depth_ratio:
                if sep > max_qualified_sep:
                    max_qualified_sep = sep
    return bool(max_qualified_sep > 0.0), len(peaks), float(max_qualified_sep)


def _within_between_group_variance_ratio(y: np.ndarray, group_ids: np.ndarray) -> float:
    """Return within-group std / between-group std.

    Ratios near 0 indicate strongly clustered target (group fully determines y).
    Ratios near or above 1 indicate group label adds no information.
    """
    uniq_groups = np.unique(group_ids)
    if len(uniq_groups) < 2:
        return float("nan")
    group_means = np.zeros(len(uniq_groups), dtype=np.float64)
    within_sq_sum = 0.0
    within_n = 0
    for k, g in enumerate(uniq_groups):
        mask = group_ids == g
        if not np.any(mask):
            continue
        yk = y[mask]
        group_means[k] = float(np.mean(yk))
        within_sq_sum += float(np.sum((yk - group_means[k]) ** 2))
        within_n += int(yk.size)
    within_std = math.sqrt(within_sq_sum / max(within_n, 1))
    between_std = float(np.std(group_means))
    if between_std <= 0.0:
        return float("inf")
    return within_std / between_std


def _classify_target_type(y: np.ndarray) -> Literal["regression", "classification"]:
    """Heuristic: integer-typed AND <= 50 unique values -> classification."""
    if y.dtype.kind in ("i", "u", "b"):
        if np.unique(y).size <= 50:
            return "classification"
    if y.dtype.kind == "f":
        # Floats with very few unique values are encoded classification.
        if np.unique(y).size <= 50:
            return "classification"
    return "regression"
