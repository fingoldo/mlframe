"""Unit + biz_value tests for robust location estimators (PZAD probweights)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.robust_location import geometric_median, robust_mean_mestimator


# ---------------------------------------------------------------- unit
def test_robust_mean_no_outliers_matches_mean_closely():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, size=500)
    r = robust_mean_mestimator(x, weight="huber", param=3.0)  # large k -> near mean
    assert abs(r - x.mean()) < 0.15


def test_robust_mean_single_and_empty():
    assert robust_mean_mestimator(np.array([7.0])) == 7.0
    assert np.isnan(robust_mean_mestimator(np.array([])))


@pytest.mark.parametrize("weight", ["meshalkin", "huber", "tukey"])
def test_robust_mean_ignores_extreme_outlier(weight):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    r = robust_mean_mestimator(x, weight=weight)
    assert 2.0 < r < 4.5, f"{weight} should resist the 1000 outlier, got {r}"


def test_invalid_weight_and_param():
    with pytest.raises(ValueError):
        robust_mean_mestimator(np.arange(5.0), weight="nope")
    with pytest.raises(ValueError):
        robust_mean_mestimator(np.arange(5.0), param=-1.0)


def test_geometric_median_1d_matches_median():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    gm = geometric_median(x)
    assert abs(gm[0] - np.median(x)) < 1e-3


def test_geometric_median_coincident_points_safe():
    X = np.zeros((10, 2))  # all identical -> distance 0 for all; must not divide by zero
    gm = geometric_median(X)
    assert np.allclose(gm, 0.0)


def test_geometric_median_empty():
    gm = geometric_median(np.empty((0, 3)))
    assert gm.shape == (3,) and np.all(np.isnan(gm))


# ---------------------------------------------------------------- biz_value
def test_biz_val_robust_mean_beats_mean_under_contamination():
    """Under 15% gross contamination, the redescending M-estimator recovers the true center with far lower
    error than the arithmetic mean. Measured: mean err ~ 3-6, robust err ~ 0.1-0.3 => >10x. Floor at 3x."""
    rng = np.random.default_rng(1)
    true_center = 10.0
    clean = rng.normal(true_center, 1.0, size=850)
    outliers = rng.uniform(40, 80, size=150)  # 15% gross outliers pulling the mean up
    x = np.concatenate([clean, outliers])
    err_mean = abs(x.mean() - true_center)
    err_robust = abs(robust_mean_mestimator(x, weight="tukey") - true_center)
    assert err_robust < err_mean / 3.0, f"robust err {err_robust:.3f} should beat mean err {err_mean:.3f} by >=3x"


def test_biz_val_geometric_median_beats_coordinate_mean_under_contamination():
    """The geometric median recovers a 2-D center under contamination better than the coordinate-wise mean."""
    rng = np.random.default_rng(2)
    center = np.array([5.0, -3.0])
    clean = rng.normal(center, 1.0, size=(800, 2))
    outliers = rng.uniform(50, 100, size=(200, 2))
    X = np.vstack([clean, outliers])
    err_mean = np.linalg.norm(X.mean(axis=0) - center)
    err_gm = np.linalg.norm(geometric_median(X) - center)
    assert err_gm < err_mean / 3.0, f"geomedian err {err_gm:.3f} should beat mean err {err_mean:.3f} by >=3x"


def test_biz_val_meshalkin_lambda_controls_robustness():
    """Larger Meshalkin lambda -> more aggressive outlier rejection -> closer to the clean center."""
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0.0, 1.0, size=180), rng.uniform(20, 30, size=20)])
    err_weak = abs(robust_mean_mestimator(x, weight="meshalkin", param=0.2))
    err_strong = abs(robust_mean_mestimator(x, weight="meshalkin", param=3.0))
    assert err_strong < err_weak, "stronger lambda should reject outliers harder"
