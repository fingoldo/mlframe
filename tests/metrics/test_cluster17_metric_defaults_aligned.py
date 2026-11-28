"""Cluster 17 regression: metric siblings share their defaults (audit 2026-11-20, finding 17-F22).

fast_max_error returned per-output values by default while every sibling returned the uniform average, and fast_calibration_metrics binned into 100 bins
while fast_calibration_report uses 10.
"""

import inspect

import numpy as np

from mlframe.metrics.calibration._calibration_metrics import fast_calibration_metrics
from mlframe.metrics.regression._regression_metrics import fast_max_error


def test_max_error_defaults_to_the_uniform_average():
    """For two outputs with maxima 1 and 3 the default is 2.0, and raw_values still returns both."""
    yt = np.zeros((4, 2))
    yp = np.array([[1.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert fast_max_error(yt, yp) == 2.0
    np.testing.assert_array_equal(fast_max_error(yt, yp, multioutput="raw_values"), [1.0, 3.0])


def test_calibration_metrics_default_to_ten_bins():
    """The signature default is 10 bins, matching fast_calibration_report."""
    default = inspect.signature(getattr(fast_calibration_metrics, "py_func", fast_calibration_metrics)).parameters["nbins"].default
    assert default == 10
