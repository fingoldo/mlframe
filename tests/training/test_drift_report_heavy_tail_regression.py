"""Regression test for the training-log audit 2026-09-20, ``SEN-02``.

Regression label drift was tested only as a mean shift in TRAIN-SIGMA units. On a heavy-tailed target sigma is set by
the tail while the shift happens in the bulk, so a production target whose mean fell 59%, std fell 63% and p99 fell
63% scored 0.09 sigma and the report printed ``(no drift warnings - splits within threshold)``. The binary branch of
the same module correctly flagged a 6.6pp move in the same run.
"""
from __future__ import annotations

import numpy as np

from mlframe.training.drift_report import (
    DEFAULT_REGRESSION_REL_SHIFT_WARN_THRESHOLD,
    compute_label_distribution_drift,
    format_drift_report,
)


def _zero_inflated_pareto(n: int, scale: float, seed: int) -> np.ndarray:
    """Zero-inflated heavy-tailed draw with the production shape: median 0, huge sigma, mass in the tail."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    k = n // 3
    y[:k] = rng.pareto(1.2, k) * scale
    return y


def _prod_like_splits():
    """train / val / test with the production ratio of level shrinkage (85.65 / 57.57 / 35.07)."""
    return (
        _zero_inflated_pareto(120_000, 60.0, 0),
        _zero_inflated_pareto(20_000, 40.0, 1),
        _zero_inflated_pareto(20_000, 24.0, 2),
    )


def test_heavy_tail_level_collapse_is_reported():
    train, val, test = _prod_like_splits()
    report = compute_label_distribution_drift(train, val, test, "regression")

    # The sigma statistic that used to be the only test is still computed, and still tiny -- that is the point.
    assert abs(report["drifts"]["test_mean_z_vs_train"]) < 0.5

    warnings = report["warnings"]
    assert warnings, "a 60%+ collapse in level, dispersion and upper tail must not be reported as no drift"
    assert any("TEST mean=" in w and "level shift" in w for w in warnings), warnings
    assert any("TEST std=" in w and "dispersion shift" in w for w in warnings), warnings
    assert any("TEST p99=" in w and "upper tail shift" in w for w in warnings), warnings
    assert "(no drift warnings" not in format_drift_report(report, "target_total_charge")


def test_ratios_are_recorded_for_every_split_and_statistic():
    train, val, test = _prod_like_splits()
    drifts = compute_label_distribution_drift(train, val, test, "regression")["drifts"]
    for split in ("val", "test"):
        for stat in ("mean", "std", "p99"):
            key = f"{split}_{stat}_ratio_vs_train"
            assert key in drifts, key
            assert 0.0 < drifts[key] < 1.0, (key, drifts[key])
    assert drifts["test_mean_ratio_vs_train"] < drifts["val_mean_ratio_vs_train"]


def test_a_stable_regression_target_stays_silent():
    """The new checks must not fire on splits drawn from one distribution, or they replace one useless verdict
    with a different useless one."""
    rng = np.random.default_rng(7)
    train = rng.gamma(2.0, 50.0, 120_000)
    val = rng.gamma(2.0, 50.0, 20_000)
    test = rng.gamma(2.0, 50.0, 20_000)
    report = compute_label_distribution_drift(train, val, test, "regression")
    assert report["warnings"] == [], report["warnings"]
    assert "(no drift warnings" in format_drift_report(report, "stable_target")


def test_shift_just_below_threshold_stays_silent_and_just_above_warns():
    """The threshold is the contract: scale the target by exactly +/- a few points around it."""
    rng = np.random.default_rng(11)
    train = rng.gamma(2.0, 50.0, 80_000)
    thr = DEFAULT_REGRESSION_REL_SHIFT_WARN_THRESHOLD
    quiet = compute_label_distribution_drift(train, None, train * (1.0 - thr * 0.5), "regression")
    loud = compute_label_distribution_drift(train, None, train * (1.0 - thr * 1.5), "regression")
    assert not [w for w in quiet["warnings"] if "level shift" in w]
    assert [w for w in loud["warnings"] if "level shift" in w]


def test_near_zero_train_reference_does_not_manufacture_a_warning():
    """A target legitimately centred on ~0 must not warn on every split from a division blow-up."""
    rng = np.random.default_rng(3)
    train = rng.normal(0.0, 1.0, 50_000)
    test = rng.normal(0.0, 1.0, 20_000)
    report = compute_label_distribution_drift(train, None, test, "regression")
    assert not [w for w in report["warnings"] if "level shift" in w], report["warnings"]


def test_binary_branch_is_untouched():
    rng = np.random.default_rng(5)
    train = (rng.random(100_000) < 0.293).astype(int)
    test = (rng.random(50_000) < 0.360).astype(int)
    report = compute_label_distribution_drift(train, None, test, "binary_classification")
    assert any("prior-shift suspected" in w for w in report["warnings"]), report["warnings"]
