"""Regression (wave6 P1/P2): the calibration binning njit kernels advertised "NaN-safe" but seeded min/max at
(1.0, 0.0) and indexed the histogram with floor((p-min)*mult). A NaN prediction survived the min/max scan
(NaN>max and NaN<min are both False), then floor(NaN) produced a garbage int64 index -> out-of-bounds write in a
numba kernel with bounds-checking off. Out-of-[0,1] predictions were also mis-binned against the stale (1.0, 0.0)
seed. Fix: drop non-finite predictions at the Python boundary, seed span from the data, and clamp the index.
"""
import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_plot import (
    fast_calibration_binning,
    calibration_binning,
    _fast_calibration_binning_prange,
    _fast_calibration_binning_serial,
)


def _clean(n, seed):
    rng = np.random.default_rng(seed)
    yp = rng.uniform(0, 1, n)
    yt = (rng.uniform(0, 1, n) < yp).astype(np.int64)
    return yt, yp


def test_nan_pred_does_not_crash_and_is_excluded():
    yt, yp = _clean(500, 0)
    yp = yp.copy()
    yp[::17] = np.nan  # inject NaNs
    yp[3] = np.inf
    fp, ft, hits = fast_calibration_binning(yt, yp, nbins=20)
    n_finite = int(np.isfinite(yp).sum())
    assert hits.sum() == n_finite, "non-finite predictions must be excluded, not counted"
    assert np.isfinite(fp).all() and np.isfinite(ft).all()


def test_all_nan_returns_empty():
    yt = np.array([0, 1, 0], dtype=np.int64)
    yp = np.array([np.nan, np.nan, np.inf])
    fp, ft, hits = fast_calibration_binning(yt, yp, nbins=10)
    assert len(fp) == 0 and len(ft) == 0 and len(hits) == 0


def test_out_of_range_predictions_bin_without_collapse():
    # Predictions all > 1 (a logit fed in by mistake): the old (1.0, 0.0) seed collapsed the lower bound to 1.0.
    yt = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    yp = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
    fp, ft, hits = fast_calibration_binning(yt, yp, nbins=5)
    assert hits.sum() == 5
    # The five distinct predictions must spread across more than one bin (not all collapse into one).
    assert len(hits) >= 2, "out-of-[0,1] predictions collapsed into a single bin"


def test_serial_and_prange_agree_on_clean_data():
    yt, yp = _clean(4000, 1)
    fp_s, ft_s, h_s = _fast_calibration_binning_serial(yt, yp, 50)
    fp_p, ft_p, h_p = _fast_calibration_binning_prange(yt, yp, 50)
    assert np.array_equal(h_s, h_p), "hit counts must be identical across serial/prange"
    assert np.array_equal(ft_s, ft_p)
    assert np.allclose(fp_s, fp_p, atol=1e-12), "mean-pred differs beyond FP reduction-order"


def test_nan_safe_quantile_strategy():
    yt, yp = _clean(600, 2)
    yp = yp.copy()
    yp[::13] = np.nan
    fp, ft, hits = calibration_binning(yt, yp, nbins=20, strategy="quantile")
    assert hits.sum() == int(np.isfinite(yp).sum())
    assert np.isfinite(fp).all()
