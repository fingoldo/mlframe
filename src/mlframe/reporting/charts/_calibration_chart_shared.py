"""Shared per-bin/per-group reliability helpers for the calibration chart family
(calibration_by_feature.py, fairness_calibration.py): independently duplicated across those
modules, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.calibration import standard_ece


def null_ece_scale(n_rows: int, prevalence: float, n_bins: int) -> float:
    """ECE a PERFECTLY calibrated sample of this size would still show from sampling noise alone.

    ECE is a mean ABSOLUTE deviation, so it is bounded away from zero at finite n and shrinks only as
    ``1/sqrt(n)``. With roughly equal bin occupancy each bin's observed rate has standard error
    ``sqrt(p(1-p)/(n/B))``, and the expected absolute deviation of a near-normal quantity is ``sqrt(2/pi)`` times
    its standard error; averaging that over bins leaves the same expression, hence ``sqrt(2*p*(1-p)*B/(pi*n))``.

    Every consumer of an ECE in this package needs it: an ECE, an ECE GAP between groups, or a per-cell ECE on a
    grid is uninterpretable without the floor its own sample size imposes. Graded against a fixed constant instead,
    a perfectly calibrated model reads RED wherever the data happens to be thin.
    """
    if n_rows <= 0 or n_bins <= 0:
        return float("inf")
    var = max(prevalence * (1.0 - prevalence), 0.0)
    return float(np.sqrt(2.0 * var * n_bins / (np.pi * n_rows)))


def is_single_class(y_true: np.ndarray) -> bool:
    """True iff the binary labels are all-0 or all-1 -- a reliability curve needs both classes. O(n), no sort/unique."""
    s = float(y_true.sum())
    return s == 0.0 or s == float(y_true.size)


def reliability_points(y_true: np.ndarray, y_score: np.ndarray, n_bins: int) -> "tuple[np.ndarray, np.ndarray, float] | None":
    """Per-bin (mean-pred, observed-freq) + standard ECE for one slice (feature-bin or group), via the shared
    njit binning.

    Returns ``(freqs_predicted, freqs_true, ece)`` or ``None`` when the slice is degenerate (single class /
    all-equal scores / no populated bin). Reuses ``fast_calibration_binning`` so binning matches the suite's
    reliability diagram exactly.
    """
    from mlframe.metrics.calibration import fast_calibration_binning

    fp, ft, hits = fast_calibration_binning(y_true, y_score, nbins=n_bins)
    if fp.size == 0:
        return None
    ece = standard_ece(fp, ft, hits)
    if not np.isfinite(ece):
        return None
    return fp, ft, ece
