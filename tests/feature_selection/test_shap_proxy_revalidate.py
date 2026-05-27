"""Tests for honest re-validation, the proxy-trust guard, and the importance-top-k ablation.

These exercise the guards that turn the cheap proxy into a defensible selector: a disjoint-holdout
honest retrain (winner's curse), a measured proxy-vs-honest fidelity report, and the unique-value
gate vs plain SHAP importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture
def planted():
    rng = np.random.default_rng(0)
    n, f = 1200, 8
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    # target is an exact linear combo of features 0,1,2 -> a linear model recovers it perfectly
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    # SHAP-like phi for a linear model on standardized-ish data: phi_j ~ coef_j * (x_j - mean)
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"], -0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    return X, y, phi, base


def test_revalidate_recovers_planted_subset(planted):
    from mlframe.feature_selection._shap_proxy_revalidate import revalidate_top_n

    X, y, phi, base = planted
    idx_search = np.arange(900)
    idx_hold = np.arange(900, 1200)
    Xs, ys = X.iloc[idx_search].reset_index(drop=True), y[idx_search]
    Xh, yh = X.iloc[idx_hold].reset_index(drop=True), y[idx_hold]

    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]
    best, ranked, baseline = revalidate_top_n(
        candidates, LinearRegression(), Xs, ys, Xh, yh,
        classification=False, metric="rmse", n_models=1, lambda_stab=0.0)
    assert set(best) == {0, 1, 2}
    assert ranked[0]["honest_loss"] < ranked[-1]["honest_loss"]
    assert baseline["honest_loss"] > ranked[0]["honest_loss"]  # the chosen subset beats a random one


def test_trust_guard_high_fidelity_on_clean_proxy(planted):
    from mlframe.feature_selection._shap_proxy_revalidate import proxy_trust_guard

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    # phi was computed on full X; slice to search rows to match ys
    rep = proxy_trust_guard(phi[:900], base[:900], ys, LinearRegression(), Xs, Xh, yh,
                            classification=False, metric="rmse", n_anchors=25,
                            rng=np.random.default_rng(0))
    assert rep["n_anchors"] >= 10
    assert rep["spearman"] > 0.5  # clean linear proxy -> good fidelity
    assert rep["trustworthy"]


def test_active_learning_respects_budget_and_not_worse_than_proxy_top1(planted):
    from mlframe.feature_selection._shap_proxy_revalidate import _honest_loss, active_learning_revalidate

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]

    # Candidate pool: proxy-best is the true {0,1,2}; several decoys with lower proxy_loss-ranks.
    candidates = [(0.05, (0, 1, 2)), (0.10, (0, 1)), (0.20, (0, 1, 2, 5)),
                  (0.30, (4, 5, 6)), (0.40, (3, 4)), (0.50, (5, 6, 7))]
    # Enough anchors for the corrector to fit (proxy ~ honest with a redundancy wobble).
    rng = np.random.default_rng(0)
    cd = dict(proxy=list(rng.uniform(0.1, 0.6, 16)), honest=list(rng.uniform(0.1, 0.6, 16)),
              cards=list(rng.integers(2, 5, 16).astype(float)), redund=list(rng.uniform(0, 1, 16)))

    budget = 4
    best_idx, ranked, n_eval = active_learning_revalidate(
        candidates, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        corrector_data=cd, phi=phi, budget=budget, batch=2, n_models=1, rng=np.random.default_rng(1))

    assert n_eval <= budget and n_eval >= 1
    assert len(ranked) == n_eval
    # The proxy top-1 is always evaluated early, so the active best is never worse than it.
    proxy_top1_honest = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, [0, 1, 2], False, "rmse")
    best_honest = min(d["honest_loss"] for d in ranked)
    assert best_honest <= proxy_top1_honest + 1e-9
    assert set(best_idx) <= {0, 1, 2, 3, 4, 5, 6, 7}


def test_importance_ablation_runs(planted):
    from mlframe.feature_selection._shap_proxy_revalidate import importance_topk_ablation

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    out = importance_topk_ablation(phi[:900], (0, 1, 2), LinearRegression(), Xs, ys, Xh, yh,
                                   classification=False, metric="rmse")
    assert out["proxy_features"] == (0, 1, 2)
    assert "proxy_honest_loss" in out and "importance_honest_loss" in out
    assert isinstance(out["proxy_wins"], bool)
