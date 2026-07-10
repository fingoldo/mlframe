"""biz_value test for ``calibration.optimize_decision_threshold`` / ``apply_decision_threshold``.

The win: on an imbalanced binary classification problem, the naive 0.5 decision threshold under-predicts the
rare positive class (a well-calibrated model's probabilities for true positives can legitimately stay below
0.5 when positives are rare), giving poor F1. Sweeping the threshold on a validation fold and picking the
F1-maximizing cutoff should measurably improve F1 over the naive default.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from mlframe.calibration.threshold_optimizer import apply_decision_threshold, optimize_decision_threshold


def test_biz_val_optimize_decision_threshold_beats_naive_half_cutoff_on_imbalanced_data():
    rng = np.random.default_rng(0)
    n_pos, n_neg = 60, 6000
    X_pos = rng.normal(2.0, 1.2, (n_pos, 1))
    X_neg = rng.normal(-0.2, 1.2, (n_neg, 1))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.6 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split : split + 1500], y[split : split + 1500]
    X_test, y_test = X[split + 1500 :], y[split + 1500 :]

    model = LogisticRegression().fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    naive_pred = apply_decision_threshold(test_proba, 0.5)
    naive_f1 = f1_score(y_test, naive_pred)

    result = optimize_decision_threshold(y_val, val_proba, metric_fn=f1_score)
    optimized_pred = apply_decision_threshold(test_proba, result["best_threshold"])
    optimized_f1 = f1_score(y_test, optimized_pred)

    assert optimized_f1 > naive_f1 + 0.1, (
        f"threshold optimization should measurably improve F1 on imbalanced data: optimized={optimized_f1:.4f} naive={naive_f1:.4f}"
    )
    assert result["best_threshold"] != 0.5


def test_optimize_decision_threshold_sweep_shape():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    result = optimize_decision_threshold(y_true, y_proba, metric_fn=f1_score, n_thresholds=50)
    assert result["thresholds"].shape == (50,)
    assert result["scores"].shape == (50,)
    assert 0.0 <= result["best_threshold"] <= 1.0
