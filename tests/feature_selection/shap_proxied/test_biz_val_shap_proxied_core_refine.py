"""biz_val: ``refine_mode="core"`` (gt_02) end-to-end through the real ``ShapProxiedFS.fit()`` pipeline.

Three scenarios per the gt_02 plan sec 5: core recovers more weak-but-real recall than legacy greedy
at non-inferior AUC, core still prunes true (near-duplicate) redundancy rather than "keeping
everything", and core's honest-gate fallback fires (and matches greedy exactly) under an adversarial
``core_drop_threshold`` that would otherwise drop nearly everything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from tests.feature_selection.shap_proxied.test_biz_val_shap_proxied_parsimony_tol_recall import _make_mixed_strength_fixture


def _fit_selected(X, y, refine_mode, seed=0, **kw):
    """Fit ShapProxiedFS with the given refine_mode and return (selected_feature_names, fitted_estimator)."""
    s = ShapProxiedFS(classification=True, random_state=seed, verbose=False, prescreen_ladder_mode="off", n_jobs=1, refine_mode=refine_mode, **kw)
    s.fit(X, y)
    return set(s.selected_features_), s


def _downstream_auc(X, y, selected, seed=0):
    """Train a fresh XGBClassifier on ``selected`` columns, return holdout AUC (higher=better)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    cols = sorted(selected)
    if not cols:
        return 0.5
    clf = XGBClassifier(n_estimators=100, max_depth=3, random_state=seed, eval_metric="logloss")
    clf.fit(Xtr[cols], ytr)
    proba = clf.predict_proba(Xte[cols])[:, 1]
    return float(roc_auc_score(yte, proba))


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_core_refine_recovers_weak_over_greedy():
    """refine_mode='core' recovers more weak-but-real features than legacy 'greedy' at non-inferior AUC.

    At the DEFAULT mixed-strength setting (6 strong w=1.0 + 6 weak w=0.25) the strong block alone
    saturates log-loss near its floor, so EVERY unit's least-core share (weak or pure-noise alike)
    falls under the credit-share drop_threshold and core degrades to greedy's 0/6 -- measured directly
    while implementing this test. A less strong-dominated regime (3 strong w=0.8 + 6 weak w=0.35) keeps
    loss further from saturation, so the LP's leave-one-out / coalition-blocking signal actually
    differentiates weak-but-real units from pure noise: measured greedy 4-5/6, core 5-6/6 across seeds
    0-2 (this session), core AUC never below greedy's on any seed.
    """
    X, y, _strong, weak = _make_mixed_strength_fixture(n_strong=3, strong_weight=0.8, weak_weight=0.35)
    weak_names = {f"f{i}" for i in weak}

    sel_greedy, _ = _fit_selected(X, y, refine_mode="greedy")
    sel_core, _ = _fit_selected(X, y, refine_mode="core")

    greedy_weak_recall = len(weak_names & sel_greedy)
    core_weak_recall = len(weak_names & sel_core)
    assert core_weak_recall > greedy_weak_recall, (
        f"core recall ({core_weak_recall}/6) did not exceed greedy recall ({greedy_weak_recall}/6) -- "
        "core should keep weak-but-real features greedy's parsimony_tol threshold drops"
    )
    assert core_weak_recall >= 5, f"core recovered only {core_weak_recall}/6 weak features, expected >= 5/6 (measured 6/6 on this seed)"

    auc_greedy = _downstream_auc(X, y, sel_greedy)
    auc_core = _downstream_auc(X, y, sel_core)
    assert auc_core >= auc_greedy - 0.005, f"core downstream AUC ({auc_core:.4f}) regressed more than 0.005 below greedy ({auc_greedy:.4f})"


def test_biz_val_core_refine_drops_true_redundancy():
    """core_refine still prunes exact-duplicate redundancy -- it doesn't just 'keep everything'.

    Bed: 4 informative features + 4 near-exact duplicate copies (rho=0.98). Each duplicate's
    leave-one-out coalition barely changes the coalition value (v(N \\ dup) approx= v(N)), so its
    least-core share collapses toward zero and core_refine drops it, same as greedy would.
    """
    rng = np.random.default_rng(1)
    n, n_info = 2000, 4
    X_info = rng.standard_normal((n, n_info)).astype(np.float32)
    noise = rng.standard_normal((n, n_info)).astype(np.float32) * np.sqrt(1 - 0.98**2)
    X_dup = (0.98 * X_info + noise).astype(np.float32)
    X_noise = rng.standard_normal((n, 20)).astype(np.float32)
    X = np.concatenate([X_info, X_dup, X_noise], axis=1)
    cols = [f"info{i}" for i in range(n_info)] + [f"dup{i}" for i in range(n_info)] + [f"noise{i}" for i in range(20)]
    Xdf = pd.DataFrame(X, columns=cols)
    logit = X_info.sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # cluster_features=True forces the clustering stage to engage on this small (p=28) fixture -- it
    # only fires "auto"-mode by default above cluster_auto_threshold=40 -- which is what surfaces
    # unit_to_members / member_groups (and hence any per-unit intra-cluster redundancy) for refine to see.
    sel_core, _ = _fit_selected(Xdf, pd.Series(y), refine_mode="core", cluster_features=True)
    assert len(sel_core) <= 6, (
        f"core_refine kept {len(sel_core)} features on a 4-informative + 4-exact-duplicate bed, " "expected duplicates to be pruned (n_selected <= 6)"
    )


def test_biz_val_core_refine_honest_fallback():
    """An adversarial core_drop_threshold=0.9 forces the core proposal to fail the honest gate;
    core_refine must fall back to the legacy greedy path and produce the SAME selection greedy would."""
    X, y, _strong, _weak = _make_mixed_strength_fixture(n=800, p=200, n_strong=4, n_weak=2)

    sel_greedy, _ = _fit_selected(X, y, refine_mode="greedy")
    sel_core_adversarial, fs_core = _fit_selected(X, y, refine_mode="core", core_drop_threshold=0.9)

    refine_report = fs_core.shap_proxy_report_["within_cluster_refine"]
    assert refine_report.get("fallback") is True, "adversarial core_drop_threshold=0.9 should have tripped the honest-gate fallback"
    assert sel_core_adversarial == sel_greedy, (
        "on honest-gate fallback, core_refine's output must equal the legacy greedy result exactly "
        f"(core={sorted(sel_core_adversarial)}, greedy={sorted(sel_greedy)})"
    )
