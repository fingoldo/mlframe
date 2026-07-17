"""Unit + biz_value tests for Nadaraya-Watson kernel smoothing (PZAD traffic lecture)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.nadaraya_watson import (
    nadaraya_watson_smooth,
    per_group_nadaraya_watson_smooth,
)


# ---------------------------------------------------------------- unit


def test_constant_signal_is_recovered():
    """Constant signal is recovered."""
    x = np.linspace(0, 10, 50)
    y = np.full(50, 3.5)
    out = nadaraya_watson_smooth(x, y, bandwidth=1.0)
    assert np.allclose(out, 3.5)


def test_empty_input():
    """Empty input."""
    assert nadaraya_watson_smooth(np.array([]), np.array([])).shape == (0,)


def test_boxcar_zero_outside_support_is_nan():
    """Boxcar zero outside support is nan."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    out = nadaraya_watson_smooth(x, y, x_query=np.array([100.0]), bandwidth=0.5, kernel="boxcar")
    assert np.isnan(out[0])


def test_sample_weight_biases_toward_weighted_points():
    """Sample weight biases toward weighted points."""
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 10.0])
    # equal kernel weight at query 0; sample_weight tilts fully to the second point
    out = nadaraya_watson_smooth(x, y, x_query=np.array([0.0]), bandwidth=1.0, sample_weight=np.array([0.0, 1.0]))
    assert np.isclose(out[0], 10.0)


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov", "boxcar", "tricube"])
def test_all_kernels_recover_linear_midrange(kernel):
    """All kernels recover linear midrange."""
    x = np.linspace(0, 10, 200)
    y = 2.0 * x + 1.0
    out = nadaraya_watson_smooth(x, y, x_query=np.array([5.0]), bandwidth=0.8, kernel=kernel)
    assert abs(out[0] - 11.0) < 0.5, f"{kernel} should track a line mid-range"


def test_invalid_kernel_raises():
    """Invalid kernel raises."""
    with pytest.raises(ValueError):
        nadaraya_watson_smooth(np.arange(3.0), np.arange(3.0), kernel="nope")


# ---------------------------------------------------------------- biz_value


def test_biz_val_nw_denoises_smooth_curve_better_than_raw():
    """NW smoothing recovers a known smooth curve from noisy samples with much lower RMSE than the raw noisy signal.

    Measured: raw RMSE ~ noise std (2.0), NW RMSE ~ 0.4-0.7 => >3x reduction. Floor at 2x to absorb seed noise.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4 * np.pi, 400)
    truth = np.sin(x) * 5 + 10
    noisy = truth + rng.normal(0, 2.0, size=x.shape)
    smoothed = nadaraya_watson_smooth(x, noisy, bandwidth=0.4)
    rmse_raw = float(np.sqrt(np.mean((noisy - truth) ** 2)))
    rmse_nw = float(np.sqrt(np.mean((smoothed - truth) ** 2)))
    assert rmse_nw < 0.5 * rmse_raw, f"NW RMSE {rmse_nw:.3f} should beat raw {rmse_raw:.3f} by >=2x"


def test_biz_val_nw_beats_boxcar_moving_average_on_curved_signal():
    """A proximity-weighted (gaussian) NW tracks a curved signal better than an equal-weight moving average
    of the same effective width (the lecture's motivation for a weighted vs flat average)."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 500)
    truth = np.exp(-((x - 5) ** 2) / 2.0) * 20  # a bump
    noisy = truth + rng.normal(0, 1.0, size=x.shape)
    gauss = nadaraya_watson_smooth(x, noisy, bandwidth=0.3, kernel="gaussian")
    box = nadaraya_watson_smooth(x, noisy, bandwidth=0.6, kernel="boxcar")  # wider flat window, similar effective smoothing
    rmse_g = float(np.sqrt(np.nanmean((gauss - truth) ** 2)))
    rmse_b = float(np.sqrt(np.nanmean((box - truth) ** 2)))
    assert rmse_g <= rmse_b + 1e-9, f"gaussian NW {rmse_g:.3f} should be <= boxcar {rmse_b:.3f} on a curved peak"


def test_biz_val_per_group_smoothing_denoises_each_entity():
    """Per-entity NW smoothing denoises each road-arc's speed series independently (no cross-entity leakage)."""
    rng = np.random.default_rng(2)
    hist = 120
    vals, groups, order, truths = [], [], [], []
    for e in range(30):
        t = np.linspace(0, 6, hist)
        truth = 50 + 20 * np.sin(t + e)  # each arc a different phase
        noisy = truth + rng.normal(0, 4.0, size=hist)
        vals.append(noisy)
        groups.append(np.full(hist, e))
        order.append(t)
        truths.append(truth)
    vals = np.concatenate(vals)
    groups = np.concatenate(groups)
    order = np.concatenate(order)
    truths = np.concatenate(truths)
    smoothed = per_group_nadaraya_watson_smooth(vals, groups, order=order, bandwidth=0.3)
    rmse_raw = float(np.sqrt(np.mean((vals - truths) ** 2)))
    rmse_sm = float(np.sqrt(np.mean((smoothed - truths) ** 2)))
    assert rmse_sm < 0.6 * rmse_raw, f"per-group NW {rmse_sm:.3f} should beat raw {rmse_raw:.3f}"
