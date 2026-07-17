"""Regression sensor for the size-aware calibration-binning dispatcher (iter128).

``fast_calibration_binning`` dispatches to a parallel prange kernel for large n
(>= ``_CALIB_BINNING_PRANGE_THRESHOLD``) and the serial njit kernel below it. The
parallel kernel must be bit-identical to serial on the integer outputs (hits,
pockets_true) and within a FP reduction-order ULP on ``freqs_predicted``.

Pre-fix this module had a single njit ``fast_calibration_binning`` and no
``_fast_calibration_binning_prange`` symbol — the import below fails on pre-fix code.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_plot import (
    _fast_calibration_binning_serial,
    _fast_calibration_binning_prange,
)


@pytest.mark.parametrize("n", [50_000, 250_000, 1_500_000])
@pytest.mark.parametrize("nbins", [10, 100])
def test_prange_matches_serial(n, nbins):
    """Prange matches serial."""
    rng = np.random.default_rng(123)
    y_pred = rng.random(n).astype(np.float64)
    y_true = (rng.random(n) < y_pred).astype(np.int64)

    fp_s, ft_s, h_s = _fast_calibration_binning_serial(y_true, y_pred, nbins)
    fp_p, ft_p, h_p = _fast_calibration_binning_prange(y_true, y_pred, nbins)

    assert np.array_equal(h_s, h_p), "hits (int populations) must be bit-identical"
    assert np.array_equal(ft_s, ft_p), "freqs_true (int/int) must be bit-identical"
    # freqs_predicted differs only by per-thread partial-sum reduction order.
    assert np.max(np.abs(fp_s - fp_p)) < 1e-9, "freqs_predicted divergence must be < 1e-9 (FP order only)"


def test_dispatcher_routes_by_size(monkeypatch):
    """Dispatcher routes by size."""
    import mlframe.metrics.calibration._calibration_plot as cp

    calls = {"serial": 0, "prange": 0}
    real_serial = cp._fast_calibration_binning_serial
    real_prange = cp._fast_calibration_binning_prange

    def spy_serial(*a, **k):
        """Spy serial."""
        calls["serial"] += 1
        return real_serial(*a, **k)

    def spy_prange(*a, **k):
        """Spy prange."""
        calls["prange"] += 1
        return real_prange(*a, **k)

    monkeypatch.setattr(cp, "_fast_calibration_binning_serial", spy_serial)
    monkeypatch.setattr(cp, "_fast_calibration_binning_prange", spy_prange)
    monkeypatch.setattr(cp, "_CALIB_BINNING_PRANGE_THRESHOLD", 1000)

    rng = np.random.default_rng(0)
    small_p = rng.random(500)
    small_t = (rng.random(500) < small_p).astype(np.int64)
    big_p = rng.random(5000)
    big_t = (rng.random(5000) < big_p).astype(np.int64)

    cp.fast_calibration_binning(small_t, small_p, nbins=10)
    cp.fast_calibration_binning(big_t, big_p, nbins=10)

    assert calls["serial"] == 1, "n below threshold must use the serial kernel"
    assert calls["prange"] == 1, "n at/above threshold must use the prange kernel"


def test_span_zero_path_matches():
    """All-equal y_pred (span == 0) routes both kernels into the single-bin branch."""
    n = 1_500_000
    y_pred = np.full(n, 0.42, dtype=np.float64)
    rng = np.random.default_rng(5)
    y_true = (rng.random(n) < 0.42).astype(np.int64)

    fp_s, ft_s, h_s = _fast_calibration_binning_serial(y_true, y_pred, 100)
    fp_p, ft_p, h_p = _fast_calibration_binning_prange(y_true, y_pred, 100)
    assert np.array_equal(h_s, h_p)
    assert np.array_equal(ft_s, ft_p)
    assert np.max(np.abs(fp_s - fp_p)) < 1e-9
