"""Cluster 20 regression: one NaN label keeps the calibration-drift base rate finite (audit 2026-11-20, finding 20-F11).

A NaN label made the base rate NaN, which turned every per-window noise band NaN. The base rate is now the mean of the finite labels.
"""

import numpy as np

from mlframe.reporting.charts.calibration_drift import calibration_drift


def test_one_nan_label_keeps_the_base_rate_finite():
    """The base rate with a NaN label matches the mean of the other labels, and stays finite (20-F11)."""
    rng = np.random.default_rng(0)
    n = 2000
    score = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < score).astype(float)
    ts = np.arange(n, dtype=np.float64)
    y_nan = y.copy()
    y_nan[0] = np.nan
    result = calibration_drift(y_nan, score, ts, n_windows=4)
    assert np.isfinite(result.base_rate)
    assert abs(result.base_rate - float(np.mean(y[1:]))) < 1e-12
