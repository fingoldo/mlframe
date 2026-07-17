"""Unit + biz_value tests for conformal prediction SETS on
``CompositeClassificationEstimator`` (LAC + APS split-conformal).

Coverage contract: with a held-out calibration set the true label lands in the
returned set with marginal probability ``>= 1 - alpha``; set size shrinks where
the model is confident.
"""

from __future__ import annotations

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training.composite.classification import CompositeClassificationEstimator
from mlframe.training.composite.conformal_classification import (
    conformal_set_threshold,
)


def _make_estimator():
    return CompositeClassificationEstimator(
        base_estimator=lgb.LGBMClassifier(n_estimators=40, num_leaves=15, verbose=-1),
    )


def _fit_multiclass(n=2400, k=4, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    centers = rng.normal(0, sep, size=(k, 6))
    X = centers[y] + rng.normal(0, 1.0, size=(n, 6))
    n_tr = n // 3
    n_cal = n // 3
    sl_tr = slice(0, n_tr)
    sl_cal = slice(n_tr, n_tr + n_cal)
    sl_te = slice(n_tr + n_cal, n)
    est = _make_estimator().fit(X[sl_tr], y[sl_tr])
    return est, X, y, sl_cal, sl_te


def _coverage(est, X_te, y_te, alpha, score):
    sets = est.predict_set(X_te, alpha=alpha, score=score)
    hit = np.array([y_te[i] in set(sets[i].tolist()) for i in range(len(sets))])
    size = np.array([len(s) for s in sets])
    return hit.mean(), size.mean(), sets


# -- unit: threshold helper -------------------------------------------------
def test_threshold_inf_when_too_few_points():
    # rank ceil((n+1)(1-alpha)) > n => +inf (valid, uninformative).
    assert conformal_set_threshold(np.array([0.3]), alpha=0.1) == float("inf")
    assert np.isfinite(conformal_set_threshold(np.linspace(0, 1, 100), alpha=0.1))


def test_threshold_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal_set_threshold(np.array([0.1, 0.2, 0.3]), alpha=0.0)


# -- unit: before-fit raise -------------------------------------------------
def test_calibrate_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    est = _make_estimator()
    with pytest.raises(NotFittedError):
        est.calibrate_conformal_set(np.zeros((5, 6)), np.zeros(5))


def test_predict_set_without_calibration_raises():
    est, X, y, sl_cal, sl_te = _fit_multiclass()
    with pytest.raises(RuntimeError):
        est.predict_set(X[sl_te], alpha=0.1)


def test_predict_set_unknown_alpha_raises():
    est, X, y, sl_cal, sl_te = _fit_multiclass()
    est.calibrate_conformal_set(X[sl_cal], y[sl_cal], alpha=0.1, score="lac")
    with pytest.raises(RuntimeError):
        est.predict_set(X[sl_te], alpha=0.2, score="lac")


# -- unit: singleton sets on easy data --------------------------------------
def test_singleton_sets_on_easy_data():
    # Very well-separated classes => the model is sure => mostly size-1 sets.
    est, X, y, sl_cal, sl_te = _fit_multiclass(sep=6.0, seed=1)
    est.calibrate_conformal_set(X[sl_cal], y[sl_cal], alpha=0.1, score="lac")
    cov, size, sets = _coverage(est, X[sl_te], y[sl_te], 0.1, "lac")
    assert cov >= 0.85, f"easy-data coverage {cov:.3f} below ~0.9"
    assert size <= 1.2, f"easy-data mean set size {size:.3f} should be ~1"


# -- unit: full-label set on hard / tiny-n ----------------------------------
def test_full_label_set_on_tiny_calibration():
    # n_cal too small to certify the level => threshold +inf => all labels in set.
    est, X, y, sl_cal, sl_te = _fit_multiclass(seed=2)
    cal_idx = np.arange(2)  # 2 calibration rows, alpha=0.1 -> rank 3 > 2 -> inf
    est.calibrate_conformal_set(X[cal_idx], y[cal_idx], alpha=0.1, score="lac")
    sets = est.predict_set(X[sl_te][:20], alpha=0.1, score="lac")
    k = est.n_classes_
    assert all(len(s) == k for s in sets), "tiny-n must yield the full label set"


def test_full_label_set_on_hard_low_alpha():
    # Overlapping classes + tiny alpha => near-full sets, high coverage.
    est, X, y, sl_cal, sl_te = _fit_multiclass(sep=0.4, seed=3)
    est.calibrate_conformal_set(X[sl_cal], y[sl_cal], alpha=0.01, score="aps")
    cov, size, _ = _coverage(est, X[sl_te], y[sl_te], 0.01, "aps")
    assert cov >= 0.95, f"hard-data low-alpha coverage {cov:.3f} below 0.95"
    assert size >= 2.0, f"hard data should need large sets, got {size:.3f}"


# -- biz_value: marginal coverage >= 1-alpha across alphas (LAC + APS) -------
@pytest.mark.parametrize("score", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_biz_val_marginal_coverage_holds(score, alpha):
    """Empirical set-coverage >= 1-alpha (minus a small finite-sample slack)."""
    est, X, y, sl_cal, sl_te = _fit_multiclass(seed=7)
    est.calibrate_conformal_set(X[sl_cal], y[sl_cal], alpha=alpha, score=score)
    cov, size, _ = _coverage(est, X[sl_te], y[sl_te], alpha, score)
    assert cov >= (1.0 - alpha) - 0.04, f"{score} coverage {cov:.3f} below target {1 - alpha:.2f} (alpha={alpha})"
    assert size <= est.n_classes_


# -- biz_value: set size shrinks as confidence grows ------------------------
def test_biz_val_set_size_shrinks_with_confidence():
    """Confident rows (high max-proba) get strictly smaller sets than the
    uncertain rows -- the core adaptivity win of conformal sets."""
    est, X, y, sl_cal, sl_te = _fit_multiclass(sep=2.0, seed=5)
    est.calibrate_conformal_set(X[sl_cal], y[sl_cal], alpha=0.1, score="aps")
    proba = est.predict_proba(X[sl_te])
    conf = proba.max(axis=1)
    sets = est.predict_set(X[sl_te], alpha=0.1, score="aps")
    size = np.array([len(s) for s in sets])
    hi = conf >= np.quantile(conf, 0.75)
    lo = conf <= np.quantile(conf, 0.25)
    assert size[hi].mean() + 0.3 <= size[lo].mean(), f"confident mean size {size[hi].mean():.2f} not smaller than uncertain {size[lo].mean():.2f}"


# -- biz_value: LAC sets are no larger than APS at equal coverage -----------
def test_biz_val_lac_sets_no_larger_than_aps():
    """LAC yields the smallest average set size (its design property)."""
    est, X, y, sl_cal, sl_te = _fit_multiclass(seed=9)
    est.calibrate_conformal_set(X[sl_cal], y[sl_cal], alpha=0.1, score="lac")
    est.calibrate_conformal_set(X[sl_cal], y[sl_cal], alpha=0.1, score="aps")
    _, size_lac, _ = _coverage(est, X[sl_te], y[sl_te], 0.1, "lac")
    _, size_aps, _ = _coverage(est, X[sl_te], y[sl_te], 0.1, "aps")
    assert size_lac <= size_aps + 0.05, f"LAC mean size {size_lac:.3f} should be <= APS {size_aps:.3f}"


# -- binary path ------------------------------------------------------------
def test_binary_conformal_sets_cover():
    rng = np.random.default_rng(11)
    n = 2400
    y = rng.integers(0, 2, size=n)
    X = (y[:, None] * 1.5) + rng.normal(0, 1.0, size=(n, 5))
    est = _make_estimator().fit(X[:800], y[:800])
    est.calibrate_conformal_set(X[800:1600], y[800:1600], alpha=0.1, score="lac")
    cov, size, _ = _coverage(est, X[1600:], y[1600:], 0.1, "lac")
    assert cov >= 0.86, f"binary coverage {cov:.3f}"
    assert size <= 2
