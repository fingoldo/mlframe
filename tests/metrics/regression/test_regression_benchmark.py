"""Unit + biz_value tests for benchmark-relative / threshold regression functionals (PZAD err_regression)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.regression._regression_benchmark import (
    fast_epsilon_band_accuracy,
    fast_logcosh_loss,
    fast_mrae,
    fast_percent_better,
    fast_rel_mae,
)


# ---------------------------------------------------------------- unit
def test_epsilon_band_accuracy_basic():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    a = np.array([0.5, 1.5, -0.5, 2.0])
    # within eps=1.0: |0.5|,|−0.5| -> 2 of 4
    assert fast_epsilon_band_accuracy(y, a, 1.0) == 0.5


def test_epsilon_band_perfect_and_empty():
    y = np.arange(5.0)
    assert fast_epsilon_band_accuracy(y, y.copy(), 0.0) == 1.0
    assert np.isnan(fast_epsilon_band_accuracy(np.array([]), np.array([]), 1.0))


def test_epsilon_negative_raises():
    with pytest.raises(ValueError):
        fast_epsilon_band_accuracy(np.zeros(3), np.zeros(3), -0.1)


def test_rel_mae_ratio_of_maes():
    y = np.array([0.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0])  # MAE 1
    bench = np.array([2.0, 2.0, 2.0])  # MAE 2
    assert abs(fast_rel_mae(y, pred, bench) - 0.5) < 1e-12


def test_percent_better_counts_wins():
    y = np.zeros(4)
    pred = np.array([0.1, 0.1, 5.0, 0.1])  # 3 wins vs bench
    bench = np.array([1.0, 1.0, 1.0, 1.0])
    assert fast_percent_better(y, pred, bench) == 0.75


def test_mrae_floors_benchmark_error():
    y = np.array([0.0, 0.0])
    pred = np.array([1.0, 1.0])
    bench = np.array([0.0, 2.0])  # first benchmark error is 0 -> floored
    # element 1: 1/eps (huge), element 2: 1/2 -> mean is dominated by first; just check finite & positive
    v = fast_mrae(y, pred, bench, eps=1.0)
    assert v > 0 and np.isfinite(v)


def test_logcosh_zero_and_symmetry():
    y = np.arange(5.0)
    assert abs(fast_logcosh_loss(y, y.copy())) < 1e-12
    a = y + 3.0
    b = y - 3.0
    assert abs(fast_logcosh_loss(y, a) - fast_logcosh_loss(y, b)) < 1e-12


def test_logcosh_overflow_safe():
    # large residuals must not overflow (uses the |z|+log(...) identity)
    y = np.array([0.0])
    a = np.array([1e6])
    v = fast_logcosh_loss(y, a)
    assert np.isfinite(v) and abs(v - (1e6 - np.log(2.0))) < 1e-3


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fast_rel_mae(np.zeros(3), np.zeros(2), np.zeros(3))


# ---------------------------------------------------------------- biz_value
def test_biz_val_epsilon_band_rewards_the_mode_on_skewed_spend():
    """The eB functional (predict within eps) is maximized near the mode, not the mean, on skewed spend.
    A mode-like constant beats the mean-like constant on within-band accuracy."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(50, 2.0, size=850)
    spikes = rng.uniform(150, 300, size=150)  # rare large spikes pull the mean up
    y = np.concatenate([cluster, spikes])
    mean_pred = np.full_like(y, y.mean())  # ~87
    mode_pred = np.full_like(y, 50.0)  # the cluster center
    acc_mean = fast_epsilon_band_accuracy(y, mean_pred, 10.0)
    acc_mode = fast_epsilon_band_accuracy(y, mode_pred, 10.0)
    assert acc_mode > acc_mean + 0.3, f"mode within-band {acc_mode:.2f} should beat mean {acc_mean:.2f}"


def test_biz_val_rel_mae_and_pb_detect_model_beating_benchmark():
    """A model closer to truth than a naive benchmark yields REL_MAE < 1 and Percent-Better > 0.5."""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, size=1000)
    good = y + rng.normal(0, 0.3, size=1000)  # tight around truth
    naive = np.full_like(y, y.mean())  # constant-mean benchmark
    assert fast_rel_mae(y, good, naive) < 0.6, "good model should beat naive by >40% MAE"
    assert fast_percent_better(y, good, naive) > 0.7, "good model should win most points"
