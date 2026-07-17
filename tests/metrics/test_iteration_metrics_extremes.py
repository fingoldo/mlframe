"""Extreme-input coverage for ``mlframe.metrics.compute_all_metrics`` (per-iteration aggregator).

Complements ``test_iteration_metrics_aggregator`` (key sets, single-class, NaN scores, empty) with
perfect / worst separation (ROC_AUC 1.0 / 0.0, accuracy 1.0 / 0.0) and extreme class imbalance
(a handful of positives among thousands of negatives -> ranking metrics still finite, not NaN).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics import compute_all_metrics


def test_binary_perfect_separation():
    # Negatives score < 0.5, positives score > 0.5, all distinct -> perfect ranking + threshold.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90])
    d = compute_all_metrics(y, s, "binary_classification")
    assert d["ROC_AUC"] == pytest.approx(1.0, abs=1e-9)
    assert d["PR_AUC"] == pytest.approx(1.0, abs=1e-9)
    assert d["accuracy"] == pytest.approx(1.0, abs=1e-9)


def test_binary_worst_separation():
    # Same labels, inverted scores: positives now rank BELOW negatives -> ROC_AUC 0, accuracy 0.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.90, 0.80, 0.70, 0.60, 0.40, 0.30, 0.20, 0.10])
    d = compute_all_metrics(y, s, "binary_classification")
    assert d["ROC_AUC"] == pytest.approx(0.0, abs=1e-9)
    assert d["accuracy"] == pytest.approx(0.0, abs=1e-9)


def test_binary_extreme_imbalance_ranking_metrics_finite():
    # 3 positives among 1000 rows; positives get the top scores -> AUC ~1.0, metrics finite (not NaN).
    n = 1000
    rng = np.random.default_rng(20260702)
    y = np.zeros(n, dtype=int)
    pos_idx = np.array([100, 500, 900])
    y[pos_idx] = 1
    s = rng.uniform(0.0, 0.5, size=n)
    s[pos_idx] = np.array([0.90, 0.95, 0.99])  # positives ranked above every negative
    d = compute_all_metrics(y, s, "binary_classification")
    assert np.isfinite(d["ROC_AUC"]) and d["ROC_AUC"] >= 0.99
    assert np.isfinite(d["PR_AUC"]) and d["PR_AUC"] > 0.5
    assert np.isfinite(d["KS"]) and np.isfinite(d["log_loss"])
