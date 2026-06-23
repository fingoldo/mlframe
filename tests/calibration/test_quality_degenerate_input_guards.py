"""Regression tests for degenerate-input guards + logging fix in mlframe.calibration.quality.

Pre-fix:
- EDGE4: ``estimate_calibration_quality_binned`` with n_samples < nbins gave bin_size==0 -> empty
  non-final pockets -> np.nanmean of empty slices -> NaN-laden ECE/CRPS report (silent garbage).
- EDGE-P2 (:270): ``show_classifier_calibration`` with nintervals==0 raised ZeroDivisionError on
  ``s // nintervals``.
- EDGE-P2 (:529): ``chi_square_statistic`` on empty pit_values returned silent NaN via 0/0 (its sibling
  ``anderson_darling_statistic`` was already guarded).
- LOG2 (:296): ``logging.exception(e)`` used the ROOT logger with the exception object as the message;
  fixed to the module logger with a static message.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("properscoring")

from mlframe.calibration.quality import (
    estimate_calibration_quality_binned,
    show_classifier_calibration,
    chi_square_statistic,
)


def test_small_n_produces_finite_report_no_nan_pockets():
    # n=4 samples but nbins=20: pre-fix bin_size==0 -> empty pockets -> NaN metrics.
    y_true = np.array([0, 1, 0, 1], dtype=np.float64)
    y_pred = np.array([0.2, 0.7, 0.3, 0.9], dtype=np.float64)
    pockets_pred, pockets_true, data, metrics = estimate_calibration_quality_binned(
        y_true, y_pred, nbins=20
    )
    assert np.all(np.isfinite(pockets_pred)), "predicted pockets must be finite"
    assert np.all(np.isfinite(pockets_true)), "true pockets must be finite"
    for name, val in metrics.items():
        assert np.isfinite(val), f"metric {name} is not finite: {val}"


def test_show_classifier_calibration_nintervals_zero_raises_clear_error():
    y_true = np.array([0, 1, 0, 1], dtype=np.float64)
    y_pred = np.array([0.2, 0.7, 0.3, 0.9], dtype=np.float64)
    with pytest.raises(ValueError, match="nintervals"):
        show_classifier_calibration(
            y_true, y_pred, title="t", nintervals=0, skip_plotting=True
        )


def test_chi_square_statistic_empty_pit_returns_nan_not_crash():
    res = chi_square_statistic(np.array([], dtype=np.float64))
    assert np.isnan(res)
