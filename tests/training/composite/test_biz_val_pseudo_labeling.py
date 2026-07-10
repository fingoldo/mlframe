"""biz_value test for ``training.composite.PseudoLabelingLoop``.

The win: with a very small labeled set (a decision tree overfits badly at n=35) and a much larger unlabeled
pool from the SAME distribution, leakage-safe fold-ensemble pseudo-labeling with confidence filtering should
recover a modest but real generalization improvement over training on the labeled data alone -- the
realistic, literature-consistent magnitude of semi-supervised self-training gains (this is NOT a dramatic
win like some other techniques; pseudo-labeling gains are small and noisy per-trial, which is why this test
averages over 10 seeds rather than asserting a single-trial threshold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from mlframe.training.composite import PseudoLabelingLoop


def _make_dataset(n: int, seed: int, d: int = 6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:3] = [1.5, -1.0, 0.5]
    logit = X @ w
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(float)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y


def test_biz_val_pseudo_labeling_loop_beats_labeled_only_baseline_mean_auc():
    aucs_base = []
    aucs_pl = []
    for seed in range(10):
        X_labeled, y_labeled = _make_dataset(35, 1000 + seed)
        X_unlabeled, _ = _make_dataset(3000, 2000 + seed)
        X_test, y_test = _make_dataset(3000, 3000 + seed)

        baseline = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_labeled, y_labeled)
        aucs_base.append(roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1]))

        loop = PseudoLabelingLoop(
            estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
            task="classification", n_rounds=2, n_splits=5, confidence_threshold=0.8, pseudo_label_weight=0.4, random_state=0,
        )
        loop.fit(X_labeled, y_labeled, X_unlabeled)
        aucs_pl.append(roc_auc_score(y_test, loop.predict(X_test)))

    mean_base = float(np.mean(aucs_base))
    mean_pl = float(np.mean(aucs_pl))
    improvement = mean_pl - mean_base
    assert improvement > 0.008, f"expected >0.008 mean AUC improvement across 10 seeds, got {improvement:.4f} (base={mean_base:.4f}, pl={mean_pl:.4f})"


def test_pseudo_labeling_loop_confidence_filtering_rejects_low_confidence_rows():
    X_labeled, y_labeled = _make_dataset(40, 1)
    X_unlabeled, _ = _make_dataset(500, 2)
    loop = PseudoLabelingLoop(
        estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
        task="classification", n_rounds=1, n_splits=5, confidence_threshold=0.9, random_state=0,
    )
    loop.fit(X_labeled, y_labeled, X_unlabeled)
    accepted, _, _ = loop.pseudo_labels_history_[0]
    assert 0 < accepted.sum() < len(accepted), "expected confidence filtering to accept SOME but not ALL unlabeled rows at a strict threshold"


def test_pseudo_labeling_loop_regression_task_end_to_end():
    rng = np.random.default_rng(0)
    X_labeled = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    y_labeled = X_labeled["a"].to_numpy() * 2 + rng.normal(scale=0.3, size=30)
    X_unlabeled = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])

    from sklearn.linear_model import LinearRegression

    loop = PseudoLabelingLoop(estimator_factory=lambda: LinearRegression(), task="regression", n_rounds=1, n_splits=3, confidence_threshold=1.0, random_state=0)
    loop.fit(X_labeled, y_labeled, X_unlabeled)
    pred = loop.predict(X_labeled)
    assert pred.shape == (30,)
    assert np.isfinite(pred).all()
