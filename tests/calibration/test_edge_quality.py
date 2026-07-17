"""Edge-case coverage for ``mlframe.calibration.quality`` binning + GoF statistics.

Covers ``bin_predictions`` / ``estimate_calibration_quality_binned`` degenerate shapes,
``anderson_darling_statistic`` at n=1 / all-tied / empty, the Miller-Madow ECI correction,
and ``mean_squared_deviation`` on empty / known inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("properscoring")

from mlframe.calibration.quality import (
    bin_predictions,
    estimate_calibration_quality_binned,
    anderson_darling_statistic,
    entropy_calibration_index,
    mean_squared_deviation,
)


def test_bin_predictions_single_sample_one_bin():
    y_true = np.array([1.0])
    y_pred = np.array([0.5])
    idx = np.argsort(y_pred)
    pockets_pred, pockets_true, data = bin_predictions(y_true, y_pred, idx, nbins=1)
    assert pockets_pred.tolist() == [0.5]
    assert pockets_true.tolist() == [1.0]
    # data row = [avg_pred, sum_true, count, avg_true]
    assert data.shape == (1, 4)
    np.testing.assert_allclose(data[0], [0.5, 1.0, 1.0, 1.0])


def test_estimate_calibration_quality_binned_nbins_gt_n_caps_and_stays_finite():
    # nbins=50 but only 5 samples: nbins is capped to n so every pocket holds >=1 row and
    # no NaN leaks into the reliability curve or the metrics.
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    pockets_pred, pockets_true, data, metrics = estimate_calibration_quality_binned(y_true, y_pred, nbins=50)
    assert len(pockets_pred) == 5, "nbins must be capped to the sample count"
    assert np.isfinite(pockets_pred).all() and np.isfinite(pockets_true).all()
    for name, val in metrics.items():
        assert np.isfinite(val), f"metric {name} must be finite, got {val}"


def test_estimate_calibration_quality_binned_empty_raises():
    with pytest.raises(ValueError, match="empty y_pred"):
        estimate_calibration_quality_binned(np.array([]), np.array([]))


def test_anderson_darling_single_sample_finite():
    # n=1: the (1/n) factor and the single-term accumulation must stay finite (not inf/NaN).
    res = anderson_darling_statistic(np.array([0.5]))
    assert np.isfinite(res)


def test_anderson_darling_all_tied_finite_and_large():
    # All PIT values identical (far from uniform) -> a large but finite A-D statistic.
    res = anderson_darling_statistic(np.full(20, 0.3))
    assert np.isfinite(res)
    assert res > 1.0


def test_anderson_darling_empty_returns_nan():
    assert np.isnan(anderson_darling_statistic(np.array([])))


def test_eci_miller_madow_reduces_bias_vs_plugin():
    # MM adds (k_obs-1)/(2N) to the plug-in entropy, which strictly lowers ECI toward its
    # perfect-calibration floor of 0. On near-uniform (calibrated) PIT values MM must be
    # <= the raw plug-in and closer to 0.
    pit = np.clip(np.random.default_rng(0).random(500), 0.0, 1.0)
    eci_mm = entropy_calibration_index(pit, bins=10, miller_madow=True)
    eci_plain = entropy_calibration_index(pit, bins=10, miller_madow=False)
    assert eci_mm >= 0.0 and eci_plain >= 0.0
    assert eci_mm <= eci_plain
    assert abs(eci_mm) <= abs(eci_plain)


def test_eci_empty_returns_zero():
    # total==0 short-circuit: no PIT mass -> ECI defined as 0.0 (perfect-calibration floor).
    assert entropy_calibration_index(np.array([]), bins=10) == 0.0


def test_msd_empty_is_nan_and_known_value():
    assert np.isnan(mean_squared_deviation(np.array([])))
    # deviation of {0.0, 1.0} from the uniform mean 0.5 -> mean(0.25, 0.25) == 0.25
    assert mean_squared_deviation(np.array([0.0, 1.0])) == pytest.approx(0.25)
    # {0.5, 0.5} sits exactly on the uniform mean -> 0.
    assert mean_squared_deviation(np.array([0.5, 0.5])) == pytest.approx(0.0)
