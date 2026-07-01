"""Unit + biz_value tests for RMSPE (PZAD minfunc)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.regression._regression_benchmark import fast_rmspe


def test_zero_error_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert fast_rmspe(y, y.copy()) == 0.0


def test_known_value():
    y = np.array([100.0, 100.0])
    a = np.array([110.0, 90.0])  # ±10% each -> RMSPE 0.1
    assert abs(fast_rmspe(y, a) - 0.1) < 1e-12


def test_excludes_zero_targets():
    y = np.array([0.0, 100.0])
    a = np.array([5.0, 110.0])  # first excluded; only the 10% counts
    assert abs(fast_rmspe(y, a) - 0.1) < 1e-12


def test_all_zero_and_empty_and_mismatch():
    assert np.isnan(fast_rmspe(np.zeros(3), np.ones(3)))
    assert np.isnan(fast_rmspe(np.array([]), np.array([])))
    with pytest.raises(ValueError):
        fast_rmspe(np.zeros(3), np.zeros(2))


def test_biz_val_rmspe_penalizes_relative_error_scale_invariant():
    """RMSPE is scale-free: the same relative error on small and large targets scores the same, unlike RMSE which
    is dominated by the large-target absolute error. This is why forecasting competitions (Rossmann) use it."""
    small_y = np.array([10.0, 10.0])
    large_y = np.array([10000.0, 10000.0])
    small_pred = small_y * 1.2  # 20% off
    large_pred = large_y * 1.2  # 20% off
    assert abs(fast_rmspe(small_y, small_pred) - fast_rmspe(large_y, large_pred)) < 1e-9
    # a model 20% off everywhere scores 0.2 regardless of target magnitude
    assert abs(fast_rmspe(large_y, large_pred) - 0.2) < 1e-9
