"""Tests for calibration_drift: reliability degradation over rolling time windows.

Covers:
- unit: window slicing, equal-population edges, datetime + numeric axes, NaN for sparse windows,
  empty input, validation, spec assembly (panels / x_is_time / worst-window vline / curve decimation).
- biz_value: a synthetic that DEGRADES after a cutpoint -> ECE-over-time rises after that window
  (trend strongly positive); a STABLE synthetic -> flat ECE (trend near zero).
- cProfile: 1M-row windowed binning stays O(n) and bounded (well under a 1s budget).
"""
from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pytest

from mlframe.reporting.charts.calibration_drift import (
    MIN_WINDOW_SAMPLES,
    CalibrationDriftResult,
    build_calibration_drift_spec,
    calibration_drift,
)
from mlframe.reporting.spec import FigureSpec, LinePanelSpec


# -----------------------------------------------------------------------------
# Synthetic generators
# -----------------------------------------------------------------------------


def _well_calibrated(n: int, rng: np.random.Generator):
    """y_score uniform, label drawn with P(y=1)=score -> perfectly calibrated by construction."""
    score = rng.uniform(size=n)
    yt = (rng.uniform(size=n) < score).astype(np.int8)
    return yt, score


def _miscalibrated(n: int, rng: np.random.Generator, p: float = 0.85):
    """y_score uniform but label independent of score (P(y=1)=p) -> large calibration error."""
    score = rng.uniform(size=n)
    yt = (rng.uniform(size=n) < p).astype(np.int8)
    return yt, score


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------


def test_returns_one_ece_per_window_equal_population():
    rng = np.random.default_rng(0)
    n = 10_000
    yt, score = _well_calibrated(n, rng)
    ts = np.arange(n)
    res = calibration_drift(yt, score, ts, n_windows=10, n_bins=10)
    assert isinstance(res, CalibrationDriftResult)
    assert res.n_windows == 10
    assert res.window_ece.shape == (10,)
    # Equal-population: every window ~n/10.
    assert res.window_counts.sum() == n
    assert np.all(np.abs(res.window_counts - n / 10) <= 1)


def test_sparse_window_yields_nan_ece():
    rng = np.random.default_rng(1)
    # Far fewer samples than windows*MIN_WINDOW_SAMPLES -> some windows under-populated.
    n = MIN_WINDOW_SAMPLES * 2  # 60
    yt, score = _well_calibrated(n, rng)
    ts = np.arange(n)
    res = calibration_drift(yt, score, ts, n_windows=10, n_bins=10)
    # 6-sample windows are below MIN_WINDOW_SAMPLES -> all NaN.
    assert np.all(np.isnan(res.window_ece))


def test_unsorted_timestamps_are_ordered_before_windowing():
    rng = np.random.default_rng(2)
    n = 6_000
    yt, score = _well_calibrated(n, rng)
    ts = np.arange(n)
    perm = rng.permutation(n)
    res_sorted = calibration_drift(yt, score, ts, n_windows=6, n_bins=10)
    res_shuffled = calibration_drift(yt[perm], score[perm], ts[perm], n_windows=6, n_bins=10)
    # Sorting by timestamp inside reconstructs the same temporal windows -> identical per-window ECE.
    np.testing.assert_allclose(res_sorted.window_ece, res_shuffled.window_ece, rtol=0, atol=1e-12)


def test_datetime_timestamps_set_x_is_time():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    n = 8_000
    yt, score = _well_calibrated(n, rng)
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    res = calibration_drift(yt, score, ts, n_windows=8, n_bins=10)
    assert np.issubdtype(res.window_centers.dtype, np.datetime64)
    spec = build_calibration_drift_spec(res)
    assert spec.panels[0][0].x_is_time is True


def test_numeric_timestamps_not_x_is_time():
    rng = np.random.default_rng(4)
    n = 8_000
    yt, score = _well_calibrated(n, rng)
    res = calibration_drift(yt, score, np.arange(n), n_windows=8, n_bins=10)
    spec = build_calibration_drift_spec(res)
    assert spec.panels[0][0].x_is_time is False


def test_empty_input_safe():
    res = calibration_drift(
        np.array([], dtype=np.int8), np.array([], dtype=np.float64), np.array([], dtype=np.int64),
        n_windows=5, n_bins=10,
    )
    assert res.n_windows == 0
    assert res.window_ece.size == 0
    assert np.isnan(res.ece_trend)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        calibration_drift(np.zeros(10), np.zeros(9), np.arange(10), n_windows=3)


def test_invalid_params_raise():
    yt, score = _well_calibrated(100, np.random.default_rng(0))
    ts = np.arange(100)
    with pytest.raises(ValueError, match="n_windows"):
        calibration_drift(yt, score, ts, n_windows=0)
    with pytest.raises(ValueError, match="n_bins"):
        calibration_drift(yt, score, ts, n_bins=1)


def test_spec_has_worst_window_vline():
    rng = np.random.default_rng(5)
    cut = 5_000
    yt = np.empty(10_000, dtype=np.int8)
    score = rng.uniform(size=10_000)
    yt[:cut] = (rng.uniform(size=cut) < score[:cut]).astype(np.int8)
    yt[cut:] = (rng.uniform(size=cut) < 0.85).astype(np.int8)
    res = calibration_drift(yt, score, np.arange(10_000), n_windows=10, n_bins=10)
    spec = build_calibration_drift_spec(res)
    vlines = spec.panels[0][0].vlines
    assert vlines is not None and len(vlines) == 1
    assert "worst window" in vlines[0][2]


def test_curve_panel_decimated_to_max():
    rng = np.random.default_rng(6)
    n = 50_000
    yt, score = _miscalibrated(n, rng)
    res = calibration_drift(yt, score, np.arange(n), n_windows=20, n_bins=10, collect_curves=True)
    spec = build_calibration_drift_spec(res, show_reliability_curves=True, max_curve_panels=6)
    assert len(spec.panels) == 2
    rel = spec.panels[1][0]
    assert isinstance(rel, LinePanelSpec)
    # perfect diagonal + at most max_curve_panels window curves.
    assert len(rel.y) <= 1 + 6
    assert rel.series_labels[0] == "perfect"


def test_no_curves_single_panel():
    rng = np.random.default_rng(7)
    n = 20_000
    yt, score = _well_calibrated(n, rng)
    res = calibration_drift(yt, score, np.arange(n), n_windows=8, n_bins=10, collect_curves=False)
    assert res.reliability_curves == ()
    spec = build_calibration_drift_spec(res)
    assert isinstance(spec, FigureSpec)
    assert len(spec.panels) == 1


# -----------------------------------------------------------------------------
# biz_value tests
# -----------------------------------------------------------------------------


def test_biz_val_calibration_drift_degrading_synthetic_ece_rises_after_cut():
    """A model calibrated before a cutpoint and miscalibrated after MUST show ECE rising after the cut.

    Measured: early-window ECE ~0.015, late-window ECE ~0.34 -> trend ~+0.32. Floor at +0.10 (well below
    the measured win) catches a regression that silences the drift (e.g. windowing/ECE wiring broken).
    """
    rng = np.random.default_rng(123)
    n = 60_000
    cut = n // 2
    score = rng.uniform(size=n)
    yt = np.empty(n, dtype=np.int8)
    yt[:cut] = (rng.uniform(size=cut) < score[:cut]).astype(np.int8)       # calibrated
    yt[cut:] = (rng.uniform(size=n - cut) < 0.85).astype(np.int8)          # label decoupled from score
    ts = np.arange(n)
    res = calibration_drift(yt, score, ts, n_windows=10, n_bins=10)

    early = np.nanmean(res.window_ece[:3])
    late = np.nanmean(res.window_ece[-3:])
    assert late - early >= 0.10, f"degrading synthetic: late-early ECE {late - early:.3f} should be >= 0.10"
    assert res.ece_trend >= 0.10, f"ece_trend {res.ece_trend:.3f} should be >= 0.10"


def test_biz_val_calibration_drift_stable_synthetic_flat_ece():
    """A model calibrated throughout MUST show flat ECE (trend near zero).

    Measured stable trend ~-0.006; every window ECE under ~0.03. Bound trend |.|<0.05 and max ECE < 0.06
    so a future change that injects spurious drift on stable data trips this.
    """
    rng = np.random.default_rng(321)
    n = 60_000
    yt, score = _well_calibrated(n, rng)
    res = calibration_drift(yt, score, np.arange(n), n_windows=10, n_bins=10)
    assert abs(res.ece_trend) < 0.05, f"stable synthetic ece_trend {res.ece_trend:.3f} should be ~0"
    assert np.nanmax(res.window_ece) < 0.06, "stable synthetic should keep every window well-calibrated"


# -----------------------------------------------------------------------------
# cProfile
# -----------------------------------------------------------------------------


def test_cprofile_calibration_drift_one_million_rows_bounded():
    """1M rows / 10 windows / 20 bins must stay O(n) and finish well under 1s (njit binning per window)."""
    rng = np.random.default_rng(99)
    n = 1_000_000
    yt, score = _well_calibrated(n, rng)
    ts = np.arange(n)
    # Warm njit kernels so the profiled call measures steady-state, not first-call JIT compile.
    calibration_drift(yt[:1000], score[:1000], ts[:1000], n_windows=3, n_bins=10)

    pr = cProfile.Profile()
    pr.enable()
    res = calibration_drift(yt, score, ts, n_windows=10, n_bins=20)
    pr.disable()

    assert res.n_windows == 10
    total = pstats.Stats(pr, stream=io.StringIO()).total_tt
    assert total < 1.0, f"calibration_drift at 1M rows took {total:.3f}s; expected < 1s (O(n) windowed binning)"
