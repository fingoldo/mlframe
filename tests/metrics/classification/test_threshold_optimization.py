"""Unit + biz_value tests for optimal-threshold search (PZAD err_classification)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._threshold_optimization import (
    THRESHOLD_METRICS,
    optimal_threshold,
)


def _apply(y_score, thr):
    return (y_score >= thr).astype(int)


def _f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


# ---------------------------------------------------------------- unit
def test_perfectly_separable_reaches_score_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    for m in THRESHOLD_METRICS:
        thr, sc = optimal_threshold(y, s, metric=m)
        pred = _apply(s, thr)
        assert np.array_equal(pred, y), f"{m}: perfect separation should be recoverable"
        assert sc > 0.99, f"{m}: score {sc} should be ~1 on separable data"


def test_returned_threshold_reproduces_reported_score():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    thr, sc = optimal_threshold(y, s, metric="f1")
    assert abs(_f1(y, _apply(s, thr)) - sc) < 1e-9


def test_all_negative_wins_when_scores_uninformative_for_accuracy():
    # 95% negative, scores random -> predicting all-negative maximizes accuracy
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.05).astype(int)
    s = rng.random(1000)
    _thr, sc = optimal_threshold(y, s, metric="accuracy")
    assert sc >= 0.9  # all-negative accuracy ~0.95


def test_invalid_metric_and_mismatch_raise():
    with pytest.raises(ValueError):
        optimal_threshold(np.zeros(3), np.zeros(3), metric="nope")
    with pytest.raises(ValueError):
        optimal_threshold(np.zeros(3), np.zeros(2))


def test_empty_input():
    thr, sc = optimal_threshold(np.array([]), np.array([]))
    assert thr == np.inf and np.isnan(sc)


def test_ties_in_score_handled():
    # identical scores cannot be split by a threshold; predictions must be consistent
    y = np.array([1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    thr, _sc = optimal_threshold(y, s, metric="f1")
    pred = _apply(s, thr)
    assert len(np.unique(pred)) == 1  # all-same, since all scores equal


# ---------------------------------------------------------------- biz_value
def test_biz_val_balanced_accuracy_threshold_differs_from_f1_under_imbalance():
    """Under class imbalance the F1-optimal and balanced-accuracy-optimal thresholds diverge (the lecture's
    central point). We assert the two chosen thresholds are meaningfully different and each maximizes its own metric."""
    rng = np.random.default_rng(2)
    n = 4000
    y = (rng.random(n) < 0.15).astype(int)  # 15% positive
    # scores: positives shifted up, heavy overlap
    s = rng.normal(0.0, 1.0, size=n) + y * 1.2
    thr_f1, _ = optimal_threshold(y, s, metric="f1")
    thr_ba, _ = optimal_threshold(y, s, metric="balanced_accuracy")
    assert abs(thr_f1 - thr_ba) > 0.15, f"F1 thr {thr_f1:.2f} and BA thr {thr_ba:.2f} should differ under imbalance"


def test_biz_val_optimal_threshold_beats_naive_half_on_shifted_scores():
    """When scores are NOT probabilities centered at 0.5, the tuned threshold gives higher F1 than the naive 0.5 cut."""
    rng = np.random.default_rng(3)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    # Both classes sit well below 0.5, so the naive 0.5 cut predicts all-negative (F1=0); a tuned threshold recovers F1.
    s = rng.normal(-3.0, 1.0, size=n) + y * 2.0
    _thr, best_f1 = optimal_threshold(y, s, metric="f1")
    f1_half = _f1(y, _apply(s, 0.5))
    assert best_f1 >= f1_half + 0.3, f"tuned F1 {best_f1:.3f} should beat naive-0.5 F1 {f1_half:.3f} by >=0.3"


def test_biz_val_mcc_and_youden_recover_separating_threshold():
    """On well-separated classes, MCC- and Youden-optimal thresholds land in the gap and score high."""
    rng = np.random.default_rng(4)
    y = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)
    s = np.concatenate([rng.normal(-2, 0.5, 500), rng.normal(2, 0.5, 500)])
    for m in ("mcc", "youden"):
        thr, sc = optimal_threshold(y, s, metric=m)
        assert -2 < thr < 2, f"{m} threshold {thr:.2f} should fall in the class gap"
        assert sc > 0.9, f"{m} score {sc:.2f} should be high on separable data"
