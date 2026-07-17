"""biz_value test for "KNN as a diverse first-level metafeature generator" (2nd_otto-group-product-
classification.md's "never underestimate nearest neighbours algorithm" recommendation).

This idea is a documented base-learner-set recommendation, not a new mechanism -- the actual validation tool
is `mlframe.votenrank.correlation_diversity_ablation.diversity_ablation_report` (built earlier this session).
This test demonstrates the recommendation concretely using EXISTING mlframe tooling: on a target with genuine
local/nonlinear structure (a spatial decision boundary a KNN's neighborhood-averaging captures differently
than tree/linear models), a KNN's OOF predictions -- despite being individually the WORST of the three
candidate models by log-loss -- get correctly flagged as low-correlation-but-lower-accuracy AND confirmed via
ablation to genuinely improve a blend of the other models, validating the source's claim in mlframe's own
existing quantitative framework.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from mlframe.votenrank.correlation_diversity_ablation import diversity_ablation_report


def _log_loss(y_true, y_pred):
    """Helper that log loss."""
    return float(log_loss(y_true, np.clip(y_pred, 1e-6, 1 - 1e-6)))


def _make_local_structure_dataset(n: int, seed: int):
    """Helper that make local structure dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    y = ((np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * 2) + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, y


def test_biz_val_knn_flagged_as_diverse_lower_accuracy_and_genuinely_improves_blend():
    """Knn flagged as diverse lower accuracy and genuinely improves blend."""
    X, y = _make_local_structure_dataset(n=2000, seed=0)

    rf_oof = cross_val_predict(RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0), X, y, cv=5, method="predict_proba")[:, 1]
    lr_oof = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=5, method="predict_proba")[:, 1]
    knn_oof = cross_val_predict(KNeighborsClassifier(n_neighbors=15), X, y, cv=5, method="predict_proba")[:, 1]

    oof_preds = {"rf": rf_oof, "lr": lr_oof, "knn": knn_oof}
    individual_scores = {name: -_log_loss(y, pred) for name, pred in oof_preds.items()}

    # KNN must genuinely be the worst individual model for this to be a real test of the "diverse but
    # weaker" claim, not a coincidence.
    assert individual_scores["knn"] < individual_scores["rf"], "expected KNN to be individually weaker than RF for this to test the diversity claim"

    report = diversity_ablation_report(oof_preds, individual_scores, y, _log_loss, correlation_threshold=0.85, higher_score_is_better=True)
    flagged = {entry["model"]: entry for entry in report}

    assert "knn" in flagged, f"expected KNN to be flagged as a low-correlation-but-lower-accuracy diversity candidate, got {list(flagged)}"
    assert (
        flagged["knn"]["ablation_improvement"] > 0
    ), f"expected including KNN to genuinely improve the blend despite its worse individual score, got {flagged['knn']['ablation_improvement']:.4f}"
