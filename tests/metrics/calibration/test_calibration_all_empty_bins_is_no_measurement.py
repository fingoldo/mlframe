"""With every bin empty there is nothing to measure: the weighted calibration MAE is NaN, not 0.

The weights are the bin hit counts. When they are all zero, the normalisation was skipped and the weighted MAE collapsed to 0.0, which reads as a
perfectly calibrated model. The other kernels in this module return NaN for the same no-data case.
"""

from __future__ import annotations

import numpy as np

from mlframe.metrics.calibration._calibration_metrics import calibration_metrics_from_freqs


def test_all_empty_bins_with_weights_gives_nan_not_zero():
    """Zero hits in every bin leaves no measurement: MAE and its spread are NaN."""
    freqs_predicted = np.array([0.1, 0.5, 0.9])
    freqs_true = np.array([0.2, 0.4, 0.8])
    hits = np.zeros(3)
    mae, std, _ = calibration_metrics_from_freqs(freqs_predicted, freqs_true, hits, 3, use_weights=True)
    assert np.isnan(mae)
    assert np.isnan(std)


def test_populated_bins_still_measure_normally():
    """A populated calibration curve keeps a finite MAE equal to the weighted mean of the bin gaps."""
    freqs_predicted = np.array([0.1, 0.5, 0.9])
    freqs_true = np.array([0.2, 0.4, 0.8])
    hits = np.array([10.0, 10.0, 10.0])
    mae, _, _ = calibration_metrics_from_freqs(freqs_predicted, freqs_true, hits, 3, use_weights=True, use_power_weighting=False)
    assert np.isfinite(mae)
    assert mae > 0.0
