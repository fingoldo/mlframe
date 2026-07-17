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
    """Optimize decision threshold beats naive half cutoff on imbalanced data."""
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
    """Optimize decision threshold sweep shape."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    result = optimize_decision_threshold(y_true, y_proba, metric_fn=f1_score, n_thresholds=50)
    assert result["thresholds"].shape == (50,)
    assert result["scores"].shape == (50,)
    assert 0.0 <= result["best_threshold"] <= 1.0


def test_biz_val_optimize_decision_threshold_per_segment_beats_global_on_heterogeneous_cohorts():
    """Two cohorts with genuinely different optimal cutoffs: a single global threshold is a compromise
    that is suboptimal for both, while a per-segment threshold (opt-in via ``groups=``) fits each cohort's
    own operating point."""
    rng = np.random.default_rng(0)

    # Cohort A: well separated scores, F1-optimal cutoff sits near 0.5.
    nA_pos, nA_neg = 400, 1600
    segA_pos = rng.normal(0.75, 0.10, nA_pos)
    segA_neg = rng.normal(0.25, 0.10, nA_neg)

    # Cohort B: whole score distribution shifted low (e.g. a stricter sub-model/channel), F1-optimal
    # cutoff sits well below cohort A's -- a threshold tuned on the pooled data misses it.
    nB_pos, nB_neg = 400, 1600
    segB_pos = rng.normal(0.35, 0.08, nB_pos)
    segB_neg = rng.normal(0.12, 0.08, nB_neg)

    y = np.concatenate([np.ones(nA_pos), np.zeros(nA_neg), np.ones(nB_pos), np.zeros(nB_neg)])
    proba = np.clip(np.concatenate([segA_pos, segA_neg, segB_pos, segB_neg]), 0.0, 1.0)
    groups = np.array(["A"] * nA_pos + ["A"] * nA_neg + ["B"] * nB_pos + ["B"] * nB_neg)

    idx = rng.permutation(len(y))
    y, proba, groups = y[idx], proba[idx], groups[idx]
    split = len(y) // 2
    y_val, proba_val, groups_val = y[:split], proba[:split], groups[:split]
    y_test, proba_test, groups_test = y[split:], proba[split:], groups[split:]

    global_result = optimize_decision_threshold(y_val, proba_val, metric_fn=f1_score)
    global_pred = apply_decision_threshold(proba_test, global_result["best_threshold"])
    global_f1 = f1_score(y_test, global_pred)

    segment_result = optimize_decision_threshold(y_val, proba_val, metric_fn=f1_score, groups=groups_val)
    segment_pred = apply_decision_threshold(
        proba_test, global_result["best_threshold"], groups=groups_test, group_thresholds=segment_result["group_thresholds"]
    )
    segment_f1 = f1_score(y_test, segment_pred)

    assert segment_f1 > global_f1 + 0.15, (
        f"per-segment thresholding should measurably beat one global threshold on heterogeneous cohorts: segment_f1={segment_f1:.4f} global_f1={global_f1:.4f}"
    )
    # the two cohorts genuinely disagree on the optimal cutoff
    assert segment_result["group_thresholds"]["A"] > segment_result["group_thresholds"]["B"] + 0.1


def test_biz_val_optimize_decision_threshold_cv_report_flags_unstable_threshold():
    """The CV threshold-stability report should tell apart a threshold that generalizes (large, clean,
    single-mode data) from one that is overfit to whichever fold got swept (small, heterogeneous data)."""
    rng = np.random.default_rng(1)

    n_pos, n_neg = 2000, 2000
    stable_pos = rng.normal(0.75, 0.10, n_pos)
    stable_neg = rng.normal(0.25, 0.10, n_neg)
    y_stable = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    proba_stable = np.clip(np.concatenate([stable_pos, stable_neg]), 0.0, 1.0)
    idx = rng.permutation(len(y_stable))
    y_stable, proba_stable = y_stable[idx], proba_stable[idx]

    stable_result = optimize_decision_threshold(y_stable, proba_stable, metric_fn=f1_score, cv=5, cv_seed=0)
    stable_report = stable_result["cv_report"]

    rng2 = np.random.default_rng(0)
    nA_pos, nA_neg = 400, 1600
    segA_pos = rng2.normal(0.75, 0.10, nA_pos)
    segA_neg = rng2.normal(0.25, 0.10, nA_neg)
    nB_pos, nB_neg = 400, 1600
    segB_pos = rng2.normal(0.35, 0.08, nB_pos)
    segB_neg = rng2.normal(0.12, 0.08, nB_neg)
    y_mixed = np.concatenate([np.ones(nA_pos), np.zeros(nA_neg), np.ones(nB_pos), np.zeros(nB_neg)])
    proba_mixed = np.clip(np.concatenate([segA_pos, segA_neg, segB_pos, segB_neg]), 0.0, 1.0)
    idx2 = rng2.permutation(len(y_mixed))
    y_mixed, proba_mixed = y_mixed[idx2], proba_mixed[idx2]
    split = len(y_mixed) // 2
    mixed_result = optimize_decision_threshold(y_mixed[:split], proba_mixed[:split], metric_fn=f1_score, cv=5, cv_seed=0)
    mixed_report = mixed_result["cv_report"]

    assert stable_report["is_stable"] is True
    assert stable_report["cv"] < 0.05
    assert mixed_report["is_stable"] is False
    assert mixed_report["cv"] > stable_report["cv"] * 2
