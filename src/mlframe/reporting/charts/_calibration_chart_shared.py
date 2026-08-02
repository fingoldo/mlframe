"""Shared per-bin/per-group reliability helpers for the calibration chart family
(calibration_by_feature.py, fairness_calibration.py): independently duplicated across those
modules, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.calibration import standard_ece


def is_single_class(y_true: np.ndarray) -> bool:
    """True iff the binary labels are all-0 or all-1 -- a reliability curve needs both classes. O(n), no sort/unique."""
    s = float(y_true.sum())
    return s == 0.0 or s == float(y_true.size)


def reliability_points(y_true: np.ndarray, y_score: np.ndarray, n_bins: int):
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
