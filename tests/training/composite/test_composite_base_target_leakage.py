"""Unit + biz_value tests for base-column target-leakage detection.

Covers detect_base_target_leakage + screen_base_pool: flags base==y and
base==y+tiny-noise and monotone(y) as leaky, does NOT flag a genuine lag-1 base
nor a merely-correlated feature, and the batch helper partitions a pool correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._leakage import (
    detect_base_target_leakage,
    screen_base_pool,
)


@pytest.fixture
def y_signal() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=2000).cumsum()  # a non-trivial 1-D series


# ---------------------------------------------------------------------------
# Unit: leaky cases
# ---------------------------------------------------------------------------
def test_base_equals_y_is_leaky(y_signal):
    v = detect_base_target_leakage(y_signal, y_signal.copy())
    assert v["is_leaky"] is True
    assert v["score"] > 0.99
    assert "near-identity" in v["reason"]


def test_base_equals_y_plus_tiny_noise_is_leaky(y_signal):
    base = y_signal + 1e-9 * np.random.default_rng(1).normal(size=y_signal.shape[0])
    v = detect_base_target_leakage(y_signal, base)
    assert v["is_leaky"] is True


def test_monotone_transform_of_y_is_leaky(y_signal):
    # cubic monotone re-encoding -> not linearly identical but fully monotone.
    base = np.sign(y_signal) * np.abs(y_signal) ** 3
    v = detect_base_target_leakage(y_signal, base)
    assert v["is_leaky"] is True, v


def test_decreasing_monotone_is_leaky(y_signal):
    v = detect_base_target_leakage(y_signal, -y_signal.copy())
    assert v["is_leaky"] is True


# ---------------------------------------------------------------------------
# Unit: non-leaky cases
# ---------------------------------------------------------------------------
def test_merely_correlated_feature_not_leaky(y_signal):
    rng = np.random.default_rng(3)
    base = y_signal + rng.normal(scale=y_signal.std(), size=y_signal.shape[0])
    v = detect_base_target_leakage(y_signal, base)
    assert v["is_leaky"] is False, v


def test_unrelated_feature_not_leaky(y_signal):
    base = np.random.default_rng(99).normal(size=y_signal.shape[0])
    v = detect_base_target_leakage(y_signal, base)
    assert v["is_leaky"] is False


def test_genuine_lag1_not_leaky_with_time_ordering():
    # AR-ish series; lag-1 base is strongly correlated but shifted in time.
    rng = np.random.default_rng(5)
    n = 2000
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + rng.normal(scale=0.5)
    base = np.empty(n)
    base[1:] = y[:-1]  # yesterday's value
    base[0] = y[0]
    t = np.arange(n)
    v = detect_base_target_leakage(y, base, time_ordering=t)
    assert v["is_leaky"] is False, v
    # The same lag base WITHOUT time info may look identity-ish but should still leave
    # a real same-time residual, so it must not be flagged on the residual gate alone.


def test_few_rows_not_leaky():
    v = detect_base_target_leakage(np.arange(4.0), np.arange(4.0))
    assert v["is_leaky"] is False
    assert "too few" in v["reason"]


def test_nonfinite_rows_dropped(y_signal):
    base = y_signal.copy()
    base[::50] = np.nan
    yv = y_signal.copy()
    yv[5::50] = np.inf
    v = detect_base_target_leakage(yv, base)
    assert v["is_leaky"] is True  # remaining finite rows are still identity


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------
def test_screen_base_pool_partitions(y_signal):
    rng = np.random.default_rng(11)
    pool = {
        "identity": y_signal.copy(),
        "tiny_noise": y_signal + 1e-9 * rng.normal(size=y_signal.shape[0]),
        "cubic": np.sign(y_signal) * np.abs(y_signal) ** 3,
        "noisy_feature": y_signal + rng.normal(scale=y_signal.std(), size=y_signal.shape[0]),
        "unrelated": rng.normal(size=y_signal.shape[0]),
    }
    res = screen_base_pool(y_signal, pool)
    assert set(res["leaky"]) == {"identity", "tiny_noise", "cubic"}, res["leaky"]
    assert set(res["safe"]) == {"noisy_feature", "unrelated"}, res["safe"]
    assert set(res["verdicts"]) == set(pool)


# ---------------------------------------------------------------------------
# biz_value: quantitative win -- leakage screen keeps a useless base out
# ---------------------------------------------------------------------------
def test_biz_val_leakage_score_separates_leaky_from_lag():
    """Leaky base scores ~1.0 while a genuine lag-1 base scores well below the cutoff.

    Measured: identity score ~1.0; lag-1 same-time residual leaves score <0.9.
    Floor margin >=0.1 between the leaky identity and the legitimate lag base.
    """
    rng = np.random.default_rng(21)
    n = 3000
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.6 * y[i - 1] + rng.normal(scale=1.0)
    lag = np.empty(n)
    lag[1:] = y[:-1]
    lag[0] = y[0]
    t = np.arange(n)

    leaky = detect_base_target_leakage(y, y.copy(), time_ordering=t)
    lagged = detect_base_target_leakage(y, lag, time_ordering=t)

    assert leaky["is_leaky"] is True
    assert lagged["is_leaky"] is False
    assert leaky["score"] - lagged["score"] >= 0.1, (leaky["score"], lagged["score"])
