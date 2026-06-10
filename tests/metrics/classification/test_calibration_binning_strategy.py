"""Tests for calibration binning corrections + strategies.

Covers:
- INV-3: reliability x-positions are the mean predicted probability per bin,
  not the wrong-width bin centre.
- INV-21: strategy = uniform / quantile / auto (auto -> quantile for rare events)
  + biz_value: on a tightly-concentrated rare-event synthetic, quantile yields
  >= 5 non-empty bins vs <= 2 for uniform.
- INV-8: non-finite probabilities are dropped before binning (no OOB write under
  numba) and the dropped count is logged.
- INV-22: Wilson binomial CI half-width on a known n / p.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_plot import (
    calibration_binning,
    fast_calibration_binning,
)


# --------------------------------------------------------------------------- #
# INV-3: mean predicted prob per bin (not bin centre)
# --------------------------------------------------------------------------- #
def _reference_uniform_binning(y_true, y_pred, nbins):
    """Plain-python reference: assign each sample to a uniform bin, return
    (mean_pred, observed_freq, count) per non-empty bin in ascending bin order."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    lo, hi = y_pred.min(), y_pred.max()
    span = hi - lo
    if span > 0:
        mult = (nbins - 1) / span
        idx = np.floor((y_pred - lo) * mult).astype(int)
    else:
        idx = np.zeros(len(y_pred), dtype=int)
    means, freqs, counts = [], [], []
    for b in range(nbins):
        mask = idx == b
        if mask.any():
            means.append(y_pred[mask].mean())
            freqs.append(y_true[mask].mean())
            counts.append(int(mask.sum()))
    return np.array(means), np.array(freqs), np.array(counts)


def test_freqs_predicted_is_bin_mean_not_centre():
    """freqs_predicted equals the per-bin mean predicted probability, matching a
    plain-python reference -- NOT the wrong-width bin centre."""
    y_true = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0.02, 0.06, 0.91, 0.93, 0.95, 0.97], dtype=np.float64)
    fp, ft, hits = fast_calibration_binning(y_true, y_pred, nbins=10)
    ref_mean, ref_freq, ref_cnt = _reference_uniform_binning(y_true, y_pred, 10)
    np.testing.assert_allclose(fp, ref_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ft, ref_freq, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(hits, ref_cnt)
    # The lowest bin holds {0.02, 0.06}; its x is their mean (0.04), not a bin
    # centre near 0.045/0.05 from the wrong-width formula.
    assert fp[0] == pytest.approx(0.04, abs=1e-9)
    # Every reported x lies within the global prediction support.
    assert fp.min() >= y_pred.min() - 1e-9
    assert fp.max() <= y_pred.max() + 1e-9


def test_freqs_predicted_within_observed_support_random():
    rng = np.random.default_rng(0)
    n = 5000
    y_pred = rng.random(n)
    y_true = (rng.random(n) < y_pred).astype(np.int64)
    fp, ft, hits = fast_calibration_binning(y_true, y_pred, nbins=20)
    # Mean-per-bin x positions are bounded by the global pred range.
    assert fp.min() >= y_pred.min() - 1e-9
    assert fp.max() <= y_pred.max() + 1e-9
    assert hits.sum() == n


# --------------------------------------------------------------------------- #
# INV-21: strategy dispatcher + biz_value
# --------------------------------------------------------------------------- #
def _rare_event_synthetic(seed: int = 7, n: int = 20000):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(0.5, 120.0, n), 0.0, 1.0)
    p[:2] = 0.99  # two high preds stretch the uniform span; the rest crowd bin 0
    y = (rng.random(n) < np.clip(p, 0.0, 1.0)).astype(np.int64)
    return y, p


def test_strategy_validation():
    y = np.array([0, 1], dtype=np.int64)
    p = np.array([0.2, 0.8], dtype=np.float64)
    with pytest.raises(ValueError):
        calibration_binning(y, p, nbins=10, strategy="nope")


def test_biz_quantile_spreads_rare_event_mass():
    """biz_value: uniform collapses a rare-event distribution into <=2 bins;
    quantile spreads it across >=5. Measured uniform=2 / quantile=10."""
    y, p = _rare_event_synthetic()
    base_rate = float(y.mean())
    assert base_rate < 0.10, f"synthetic must be rare-event, got base_rate={base_rate}"

    _, _, hits_uniform = calibration_binning(y, p, nbins=10, strategy="uniform")
    _, _, hits_quantile = calibration_binning(y, p, nbins=10, strategy="quantile")

    assert len(hits_uniform) <= 2, f"uniform should collapse rare event, got {len(hits_uniform)} bins"
    assert len(hits_quantile) >= 5, f"quantile should spread mass, got {len(hits_quantile)} bins"


def test_auto_picks_quantile_for_rare_event():
    y, p = _rare_event_synthetic()
    _, _, hits_auto = calibration_binning(y, p, nbins=10, strategy="auto")
    _, _, hits_quantile = calibration_binning(y, p, nbins=10, strategy="quantile")
    # auto -> quantile when base rate < 10%.
    assert len(hits_auto) == len(hits_quantile)
    assert len(hits_auto) >= 5


def test_auto_picks_uniform_for_balanced():
    rng = np.random.default_rng(3)
    n = 10000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(np.int64)  # ~50% base rate
    fp_auto, _, _ = calibration_binning(y, p, nbins=10, strategy="auto")
    fp_uniform, _, _ = calibration_binning(y, p, nbins=10, strategy="uniform")
    # auto -> uniform when base rate >= 10%; identical output.
    np.testing.assert_allclose(fp_auto, fp_uniform, rtol=1e-12, atol=1e-12)


def test_quantile_bins_equal_population():
    """Equal-population edges -> bins of near-equal size."""
    rng = np.random.default_rng(5)
    n = 10000
    p = np.clip(rng.beta(2.0, 5.0, n), 0, 1)
    y = (rng.random(n) < p).astype(np.int64)
    _, _, hits = calibration_binning(y, p, nbins=10, strategy="quantile")
    assert len(hits) >= 5
    # Each non-empty quantile bin holds roughly n/nbins; allow a wide tolerance.
    assert hits.max() / hits.min() < 3.0


# --------------------------------------------------------------------------- #
# INV-8: NaN guard
# --------------------------------------------------------------------------- #
def test_nan_probs_dropped_before_binning(caplog):
    from mlframe.metrics.classification._classification_report import fast_calibration_report
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(np.int64)
    p[:25] = np.nan
    p[25:30] = np.inf
    with caplog.at_level(logging.WARNING):
        out = fast_calibration_report(y, p, nbins=10, show_plots=False, plot_file="")
    # The report must complete (finite ICE) instead of corrupting memory.
    assert np.isfinite(out[10])
    assert any("non-finite" in r.message for r in caplog.records)


def test_all_nan_probs_does_not_crash():
    from mlframe.metrics.classification._classification_report import fast_calibration_report
    n = 100
    y = np.zeros(n, dtype=np.int64)
    p = np.full(n, np.nan)
    out = fast_calibration_report(y, p, nbins=10, show_plots=False, plot_file="")
    # All-NaN -> degenerate but no exception.
    assert out is not None


# --------------------------------------------------------------------------- #
# INV-22: Wilson CI half-width
# --------------------------------------------------------------------------- #
def test_wilson_ci_known_value():
    """Wilson 95% CI for p_hat=0.5, n=100 has a known half-width.

    center = (0.5 + z^2/200) / (1 + z^2/100)
    half   = z*sqrt(0.25/100 + z^2/40000) / (1 + z^2/100)
    With z=1.959964: center=0.5, half ~= 0.0980 (the canonical Wilson(0.5,100)
    interval is approximately [0.4020, 0.5980])."""
    from mlframe.reporting.charts.calibration import wilson_ci
    lower, upper = wilson_ci(np.array([0.5]), np.array([100]))
    z = 1.959963984540054
    denom = 1.0 + z * z / 100.0
    center = (0.5 + z * z / 200.0) / denom
    half = z * np.sqrt(0.25 / 100.0 + z * z / 40000.0) / denom
    assert lower[0] == pytest.approx(center - half, abs=1e-9)
    assert upper[0] == pytest.approx(center + half, abs=1e-9)
    # Sanity vs the textbook Wilson(0.5, 100) interval ~= [0.4038, 0.5962]
    # (symmetric about 0.5 at p_hat=0.5; half-width ~= 0.0962).
    assert lower[0] == pytest.approx(0.40383, abs=1e-3)
    assert upper[0] == pytest.approx(0.59617, abs=1e-3)
    assert (upper[0] - lower[0]) == pytest.approx(0.19234, abs=1e-3)


def test_wilson_ci_clips_and_handles_zero_n():
    from mlframe.reporting.charts.calibration import wilson_ci
    lower, upper = wilson_ci(np.array([0.0, 1.0, 0.5]), np.array([10, 10, 0]))
    # p_hat=0 / p_hat=1 stay inside [0,1].
    assert lower[0] >= 0.0 and upper[0] <= 1.0
    assert lower[1] >= 0.0 and upper[1] <= 1.0
    # n==0 -> nan (no interval).
    assert np.isnan(lower[2]) and np.isnan(upper[2])
    # Wilson half-width shrinks as n grows.
    lo_small, hi_small = wilson_ci(np.array([0.5]), np.array([10]))
    lo_big, hi_big = wilson_ci(np.array([0.5]), np.array([1000]))
    assert (hi_big[0] - lo_big[0]) < (hi_small[0] - lo_small[0])
