"""biz_value test for ``votenrank.correlation_diversity_ablation.recommend_diversity_additions``.

Gap: ``diversity_ablation_report`` flags low-correlation-but-lower-accuracy candidates but leaves the caller
to scan the raw table for which flagged entries actually paid off and in what order to add them (a flagged
candidate can still have zero or negative ``ablation_improvement``). This test builds a small model zoo --
several mutually-correlated tree/linear learners plus one genuinely diverse-but-individually-mediocre KNN
learner (local/nonlinear decision boundary the correlated learners all miss the same way) -- and confirms the
recommender both correctly surfaces the KNN learner as its top-ranked shortlist pick and correctly excludes a
near-duplicate model with zero ablation payoff.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from mlframe.votenrank.correlation_diversity_ablation import diversity_ablation_report, recommend_diversity_additions


def _log_loss(y_true, y_pred):
    """Helper that log loss."""
    return float(log_loss(y_true, np.clip(y_pred, 1e-6, 1 - 1e-6)))


def _make_local_structure_dataset(n: int, seed: int):
    """Helper that make local structure dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    y = ((np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * 2) + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, y


def _build_zoo(X, y):
    # rf/et: two mutually-correlated tree learners (same architecture family, different randomization). lr: a
    # correlated-enough linear learner. knn: the diverse-but-mediocre local-neighborhood learner this test is
    # actually about.
    """Helper that build zoo."""
    rf_oof = cross_val_predict(RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0), X, y, cv=5, method="predict_proba")[:, 1]
    et_oof = cross_val_predict(ExtraTreesClassifier(n_estimators=200, max_depth=4, random_state=1), X, y, cv=5, method="predict_proba")[:, 1]
    lr_oof = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=5, method="predict_proba")[:, 1]
    knn_oof = cross_val_predict(KNeighborsClassifier(n_neighbors=15), X, y, cv=5, method="predict_proba")[:, 1]
    return {"rf": rf_oof, "et": et_oof, "lr": lr_oof, "knn": knn_oof}


def test_biz_val_recommend_diversity_additions_surfaces_knn_top_pick():
    """Recommend diversity additions surfaces knn top pick."""
    X, y = _make_local_structure_dataset(n=2000, seed=0)
    oof_preds = _build_zoo(X, y)
    individual_scores = {name: -_log_loss(y, pred) for name, pred in oof_preds.items()}

    # KNN must genuinely be mediocre, not the individually strongest model -- the whole point is surfacing a
    # MEDIOCRE-but-diverse learner, not coincidentally the best one.
    assert individual_scores["knn"] < max(individual_scores.values()), "expected KNN to not be the individually strongest learner in this zoo"

    shortlist = recommend_diversity_additions(oof_preds, individual_scores, y, _log_loss, correlation_threshold=0.85, higher_score_is_better=True)

    assert len(shortlist) >= 1, f"expected at least one recommended diversity addition, got empty shortlist"
    assert shortlist[0]["model"] == "knn", f"expected KNN ranked first (largest genuine blend improvement), got {shortlist[0]['model']}"
    assert shortlist[0]["recommended_rank"] == 1
    # Measured ablation_improvement=0.0115 (seed=0, n=2000) -- threshold set ~13% below the measured value.
    assert shortlist[0]["ablation_improvement"] > 0.01, f"expected a real measured blend improvement from KNN, got {shortlist[0]['ablation_improvement']:.4f}"

    # Ranking is sorted descending by ablation_improvement.
    improvements = [entry["ablation_improvement"] for entry in shortlist]
    assert improvements == sorted(improvements, reverse=True)


def test_biz_val_recommend_diversity_additions_excludes_non_improving_flagged_candidate():
    """Recommend diversity additions excludes non improving flagged candidate."""
    rng = np.random.default_rng(3)
    n = 1500
    shared = rng.normal(size=n)
    y_true = shared

    best = shared + 0.05 * rng.standard_normal(n)
    # Low-correlation-but-lower-accuracy AND genuinely useless: pure noise uncorrelated with everything,
    # carries no recoverable signal about y_true -- diversity_ablation_report should still flag it (it clears
    # both the correlation and accuracy criteria), but its ablation_improvement must not be positive enough to
    # make the recommender's shortlist.
    noise_candidate = rng.normal(size=n) * 3.0

    oof_preds = {"best": best, "noise": noise_candidate}
    individual_scores = {"best": -abs(np.mean(best - y_true)), "noise": -abs(np.mean(noise_candidate - y_true)) - 10.0}

    def _mae(yt, yp):
        """Helper that mae."""
        return float(np.mean(np.abs(yt - yp)))

    report = diversity_ablation_report(oof_preds, individual_scores, y_true, _mae, correlation_threshold=0.85, higher_score_is_better=True)
    assert any(entry["model"] == "noise" for entry in report), "expected 'noise' to be flagged by the raw report (low-correlation, lower-accuracy)"

    shortlist = recommend_diversity_additions(oof_preds, individual_scores, y_true, _mae, correlation_threshold=0.85, higher_score_is_better=True)
    shortlisted_names = {entry["model"] for entry in shortlist}
    assert (
        "noise" not in shortlisted_names
    ), f"expected the non-improving flagged 'noise' candidate excluded from the recommender shortlist, got {shortlisted_names}"


def test_recommend_diversity_additions_top_k_caps_shortlist():
    """Recommend diversity additions top k caps shortlist."""
    rng = np.random.default_rng(7)
    n = 2000
    shared = rng.normal(size=n)
    u1 = rng.normal(size=n)
    u2 = rng.normal(size=n)
    u3 = rng.normal(size=n)
    y_true = shared + u1 + u2 + u3

    best = shared + 0.2 * rng.standard_normal(n)
    c1 = shared + 3.0 * u1 + 0.3 * rng.standard_normal(n)
    c2 = shared + 3.0 * u2 + 0.3 * rng.standard_normal(n)
    c3 = shared + 3.0 * u3 + 0.3 * rng.standard_normal(n)
    oof_preds = {"best": best, "c1": c1, "c2": c2, "c3": c3}

    def _rmse(yt, yp):
        """Helper that rmse."""
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    individual_scores = {name: -_rmse(y_true, pred) for name, pred in oof_preds.items()}

    full_shortlist = recommend_diversity_additions(oof_preds, individual_scores, y_true, _rmse, correlation_threshold=0.9, higher_score_is_better=True)
    assert len(full_shortlist) >= 2, f"expected multiple diverse candidates flagged, got {len(full_shortlist)}"

    capped_shortlist = recommend_diversity_additions(
        oof_preds, individual_scores, y_true, _rmse, correlation_threshold=0.9, higher_score_is_better=True, top_k=1
    )
    assert len(capped_shortlist) == 1
    assert capped_shortlist[0]["model"] == full_shortlist[0]["model"]
