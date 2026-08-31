"""Coverage for metrics.classification._threshold_optimization.optimal_threshold, previously
untested. Validated by brute-force: sweep every candidate threshold, recompute the target metric via
sklearn, and confirm the O(n log n) incremental-sweep kernel's arg-max matches."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._threshold_optimization import (
    THRESHOLD_METRICS,
    optimal_threshold,
)

pytestmark = pytest.mark.fast


def _brute_force_best(y_true, y_score, metric):
    """O(n^2)-ish reference: score every candidate threshold with sklearn/direct formulas, return the best."""
    from sklearn.metrics import f1_score, matthews_corrcoef

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    candidates = np.concatenate([[np.inf], np.unique(y_score)])
    best_val = -np.inf
    best_thr = np.inf
    for thr in candidates:
        pred = (y_score >= thr).astype(int)
        if metric == "f1":
            val = f1_score(y_true, pred, zero_division=0)
        elif metric == "accuracy":
            val = (pred == y_true).mean()
        elif metric == "balanced_accuracy":
            from sklearn.metrics import balanced_accuracy_score

            val = balanced_accuracy_score(y_true, pred) if len(np.unique(y_true)) > 1 else 0.0
        elif metric == "mcc":
            den_ok = len(np.unique(pred)) > 1 and len(np.unique(y_true)) > 1
            val = matthews_corrcoef(y_true, pred) if den_ok else 0.0
        elif metric == "youden":
            tp = ((pred == 1) & (y_true == 1)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            tn = ((pred == 0) & (y_true == 0)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            val = tpr + tnr - 1.0
        elif metric == "cost":
            # Negated average cost per row, matching the kernel: every functional here is higher-is-better so
            # one arg-max serves them all. Default costs are 1.0 / 1.0, i.e. plain error rate negated.
            fn = ((pred == 0) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            val = -(fp * 1.0 + fn * 1.0) / len(y_true)
        else:
            raise ValueError(metric)
        if val > best_val:
            best_val = val
            best_thr = thr
    return best_thr, best_val


@pytest.mark.parametrize("metric", THRESHOLD_METRICS)
def test_optimal_threshold_matches_brute_force(metric):
    """Every supported metric: the incremental-sweep kernel's arg-max matches a brute-force threshold scan."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    y_score = y_true * 0.6 + rng.random(n) * 0.5  # informative but noisy score

    _thr, val = optimal_threshold(y_true, y_score, metric=metric)
    _, ref_val = _brute_force_best(y_true, y_score, metric)
    assert val == pytest.approx(ref_val, abs=1e-9)


def test_optimal_threshold_invalid_metric_raises():
    """An unsupported metric name raises ValueError naming the supported set."""
    with pytest.raises(ValueError, match="f1"):
        optimal_threshold(np.array([0, 1]), np.array([0.1, 0.9]), metric="not_a_metric")


def test_optimal_threshold_length_mismatch_raises():
    """y_true and y_score must have matching lengths."""
    with pytest.raises(ValueError, match="length mismatch"):
        optimal_threshold(np.array([0, 1, 0]), np.array([0.1, 0.9]))


def test_optimal_threshold_empty_input():
    """Empty input returns (+inf, nan) rather than raising."""
    thr, val = optimal_threshold(np.array([]), np.array([]))
    assert thr == np.inf
    assert np.isnan(val)


def test_optimal_threshold_perfect_separation():
    """A perfectly separable score must find a threshold achieving the max value (1.0) for f1/accuracy."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr, val = optimal_threshold(y_true, y_score, metric="f1")
    assert val == pytest.approx(1.0)
    assert 0.3 < thr <= 0.7
