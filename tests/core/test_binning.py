"""Unit + biz_value tests for binning-smoothing (PZAD datapreprocessing)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.binning import (
    apply_bin_smoother,
    bin_smooth,
    fit_bin_smoother,
)


# ---------------------------------------------------------------- unit
def test_bin_means_han_kamber_example():
    # Han & Kamber classic: 4,8,15,21,21,24,25,28,34 into 3 equal-depth bins -> bin means
    """Bin means han kamber example."""
    x = np.array([4, 8, 15, 21, 21, 24, 25, 28, 34], dtype=float)
    out = bin_smooth(x, n_bins=3, strategy="mean", binning="quantile")
    # bin1 {4,8,15}->9, bin2 {21,21,24}->22, bin3 {25,28,34}->29
    assert np.allclose(np.unique(out), [9.0, 22.0, 29.0])


def test_boundary_strategy_snaps_to_nearer_edge():
    """Boundary strategy snaps to nearer edge."""
    x = np.array([4, 8, 15, 21, 21, 24, 25, 28, 34], dtype=float)
    sm = fit_bin_smoother(x, n_bins=3, binning="quantile")
    out = apply_bin_smoother(x, sm, strategy="boundary")
    # every value maps to one of the (quantile) bin edges
    assert set(np.round(out, 6)).issubset(set(np.round(sm["edges"], 6)))
    # 8 is in the first bin [edges0, edges1]; nearer boundary is the low edge (=4, the min)
    assert out[1] == sm["edges"][0]


def test_median_strategy():
    """Median strategy."""
    x = np.array([1, 2, 100, 3, 4, 200], dtype=float)  # median robust to the big values within a bin
    sm = fit_bin_smoother(x, n_bins=2, binning="quantile")
    out = apply_bin_smoother(x, sm, strategy="median")
    assert np.all(np.isfinite(out))


def test_nan_passthrough():
    """Nan passthrough."""
    x = np.array([1.0, np.nan, 3.0, 4.0])
    out = bin_smooth(x, n_bins=2, strategy="mean")
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


def test_fit_apply_on_heldout():
    """Fit apply on heldout."""
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=500)
    test = rng.normal(0, 1, size=100)
    sm = fit_bin_smoother(train, n_bins=5, binning="uniform")
    out = apply_bin_smoother(test, sm, strategy="mean")
    assert out.shape == test.shape
    # every output value is one of the (<=5) fitted bin means
    assert set(np.round(out, 6)).issubset(set(np.round(sm["bin_mean"], 6)))


def test_invalid_args():
    """Invalid args."""
    with pytest.raises(ValueError):
        fit_bin_smoother(np.arange(10.0), binning="nope")
    with pytest.raises(ValueError):
        fit_bin_smoother(np.arange(10.0), n_bins=0)
    with pytest.raises(ValueError):
        apply_bin_smoother(np.arange(10.0), fit_bin_smoother(np.arange(10.0)), strategy="bad")
    with pytest.raises(ValueError):
        fit_bin_smoother(np.array([np.nan, np.nan]))


# ---------------------------------------------------------------- biz_value
def test_biz_val_rank_preserving_quantizer():
    """Binning-smoothing is a MONOTONE (rank-preserving) quantizer: it maps a continuous feature to <= n_bins
    levels on the original scale without inverting any pairwise order (Spearman == 1), unlike an arbitrary recode."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=1000)
    out = bin_smooth(x, n_bins=10, strategy="mean", binning="quantile")
    assert len(np.unique(out)) <= 10  # quantized to at most n_bins levels
    # rank-preserving == no pairwise INVERSIONS: sorted by x, the output is non-decreasing (ties are expected, so
    # Spearman < 1 from the ties alone; the real guarantee is monotonicity, not a perfect rank correlation)
    ordered = out[np.argsort(x, kind="stable")]
    assert np.all(np.diff(ordered) >= -1e-12), "binning-smoothing must not invert any pair's order"
    assert spearmanr(x, out).correlation > 0.99  # high despite tie-induced loss


def test_biz_val_recovers_discretized_then_noised_feature():
    """The classic denoising win: a feature that is DISCRETE at heart (few true levels) but OBSERVED with additive
    jitter is recovered by binning — the noisy values cluster around the true levels, quantile bins align with the
    clusters, and each bin's mean returns the true level. MSE to the true levels drops far below the noisy input."""
    rng = np.random.default_rng(4)
    n = 5000
    levels = rng.integers(0, 10, size=n).astype(float)  # true feature takes 10 integer levels
    noisy = levels + rng.normal(0, 0.15, size=n)  # jitter << the 1.0 gap between levels
    # uniform (equal-width) bins align cleanly to equally-spaced integer levels regardless of cluster counts
    smoothed = bin_smooth(noisy, n_bins=10, strategy="mean", binning="uniform")
    mse_noisy = np.mean((noisy - levels) ** 2)
    mse_smoothed = np.mean((smoothed - levels) ** 2)
    assert mse_smoothed < mse_noisy * 0.3, f"smoothed MSE {mse_smoothed:.4f} should be <0.3x noisy {mse_noisy:.4f}"


def test_biz_val_median_robust_to_within_bin_spikes():
    """Median-smoothing caps within-bin outliers better than mean: on the discretized feature with rare huge spikes,
    median recovers the true levels closer than mean (which is pulled by the spike inside each bin)."""
    rng = np.random.default_rng(5)
    n = 5000
    levels = rng.integers(0, 10, size=n).astype(float)
    noisy = levels + rng.normal(0, 0.15, size=n)
    spike = rng.random(n) < 0.03
    noisy[spike] += rng.uniform(30, 60, size=spike.sum())  # rare huge spikes pull bin means up
    # quantile bins are percentile-based, so the spikes cluster into the top bin (they don't blow up the range as
    # equal-width edges would); that bin's mean is dragged toward the spikes while its median stays near the true level
    mean_sm = bin_smooth(noisy, n_bins=10, strategy="mean", binning="quantile")
    med_sm = bin_smooth(noisy, n_bins=10, strategy="median", binning="quantile")
    mse_mean = np.mean((mean_sm - levels) ** 2)
    mse_med = np.mean((med_sm - levels) ** 2)
    assert mse_med < mse_mean * 0.5, f"median MSE {mse_med:.3f} should beat mean MSE {mse_mean:.3f} under within-bin spikes"
