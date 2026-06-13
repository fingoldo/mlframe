"""Regression test: maximum_absolute_percentage_error must propagate NaN on
non-finite input instead of silently dropping the bad row via np.nanmax.

Pre-fix, a NaN in y_true (or y_pred) was skipped by np.nanmax / the parallel
``err == err`` guard, so the metric returned a misleadingly-finite value
(e.g. 0.0) that looked like a clean score. The other percentage metrics
(smape/wmape/mdape/pinball) already return NaN on non-finite input; MAPE was
the inconsistent outlier that masked a corrupt row.
"""

import numpy as np

from mlframe.metrics._core_precision_mape import maximum_absolute_percentage_error as mape


def test_mape_nan_in_y_true_propagates_nan():
    # All non-NaN rows are exact (error 0). Pre-fix nanmax dropped the NaN row
    # and returned 0.0 -- a perfect-looking score on corrupt data.
    y = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(mape(y, p))


def test_mape_nan_in_y_pred_propagates_nan():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, np.nan, 4.0])
    assert np.isnan(mape(y, p))


def test_mape_inf_in_pred_propagates_nan():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, np.inf, 3.0, 4.0])
    assert np.isnan(mape(y, p))


def test_mape_clean_input_unchanged():
    # No non-finite values: behaviour must be unchanged (known answer).
    y = np.array([1.0, 2.0, 4.0])
    p = np.array([1.1, 2.0, 5.0])  # errors: 0.1, 0, 0.25 -> max 0.25
    assert abs(mape(y, p) - 0.25) < 1e-12


def test_mape_parallel_path_propagates_nan():
    # Force the >=500k parallel kernel path; one NaN row must still yield NaN.
    n = 500_000
    y = np.ones(n)
    p = np.ones(n)
    y[123] = np.nan
    assert np.isnan(mape(y, p))
