"""biz_val: gt_09 two-phase residual attribution recovers weak-signal recall that ``parsimony_tol``
(see ``test_biz_val_shap_proxied_parsimony_tol_recall.py``) documents as lost by default.

Same mixed-strength fixture (6 strong w=1.0 at cols 0-5, 6 weak w=0.25 at cols 50-55): the weak
features carry real signal but the additive proxy under-credits them because the strong features
absorb most of the shared SHAP credit, so ``within_cluster_refine``'s ``parsimony_tol`` greedy
pruner drops them (measured baseline: weak recall 0/6 at defaults). A second SHAP pass on pass-1's
residual re-attributes what the strong features failed to explain, which is dominated by the weak
features once the strong signal is subtracted out -- this pins that recovery end-to-end.
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


def _make_mixed_strength_fixture(seed=0, n=3000, p=3000, n_strong=6, n_weak=6, strong_weight=1.0, weak_weight=0.25):
    """Same generator as ``test_biz_val_shap_proxied_parsimony_tol_recall``: 6 strong + 6 weak features."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    strong = list(range(n_strong))
    weak = list(range(50, 50 + n_weak))
    logit = strong_weight * X[:, strong].sum(axis=1) + weak_weight * X[:, weak].sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(p)]
    Xdf = pd.DataFrame(X, columns=cols)
    return Xdf, pd.Series(y), strong, weak


def _make_pure_strong_fixture(seed=0, n=3000, p=3000, n_strong=6, strong_weight=1.0):
    """Pure-strong bed: 6 strong features + pure noise, NO weak signal anywhere."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    strong = list(range(n_strong))
    logit = strong_weight * X[:, strong].sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(p)]
    Xdf = pd.DataFrame(X, columns=cols)
    return Xdf, pd.Series(y), strong


def _fit_selected(X, y, seed=0, **kwargs):
    """Fit a ShapProxiedFS with the given kwargs and return the selected feature-name set."""
    s = ShapProxiedFS(classification=True, random_state=seed, verbose=False, prescreen_ladder_mode="off", n_jobs=1, **kwargs)
    s.fit(X, y)
    return set(s.selected_features_)


def _downstream_auc(X, y, selected_names, seed=0):
    """Retrain an xgboost classifier on the selected columns and return holdout AUC."""
    cols = sorted(selected_names)
    Xs = X[cols]
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=seed, stratify=y)
    clf = XGBClassifier(n_estimators=300, random_state=seed, eval_metric="logloss")
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_residual_passes_recovers_weak_recall():
    """residual_passes=1 recovers >=3/6 weak features vs the 0/6 measured default baseline, without
    materially hurting downstream AUC (>= baseline - 0.005)."""
    X, y, _strong, weak = _make_mixed_strength_fixture()
    weak_names = {f"f{i}" for i in weak}

    sel_default = _fit_selected(X, y, residual_passes=0)
    sel_residual = _fit_selected(X, y, residual_passes=1, residual_merge="rescue")

    default_weak_recall = len(weak_names & sel_default)
    residual_weak_recall = len(weak_names & sel_residual)
    assert residual_weak_recall >= 3, (
        f"residual_passes=1 recovered {residual_weak_recall}/6 weak features, expected >=3/6 " f"(default recall was {default_weak_recall}/6)"
    )

    auc_default = _downstream_auc(X, y, sel_default)
    auc_residual = _downstream_auc(X, y, sel_residual)
    assert auc_residual >= auc_default - 0.005, (
        f"residual_passes=1 downstream AUC ({auc_residual:.4f}) regressed vs default " f"({auc_default:.4f}) beyond the -0.005 tolerance"
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_residual_passes_no_noise_inflation():
    """On a pure-strong bed (no real weak signal anywhere), residual_passes=1 must not inflate the
    selection with noise columns: n_selected grows by at most 1 vs default, and zero noise columns
    (outside the strong set) are selected -- the residual of a well-explained target is noise, and
    pass 2's top-k rescue candidates must fail refine/revalidation arbitration."""
    X, y, strong = _make_pure_strong_fixture()
    strong_names = {f"f{i}" for i in strong}

    sel_default = _fit_selected(X, y, residual_passes=0)
    sel_residual = _fit_selected(X, y, residual_passes=1, residual_merge="rescue")

    assert len(sel_residual) <= len(sel_default) + 1, (
        f"residual_passes=1 selected {len(sel_residual)} features vs default {len(sel_default)} "
        "-- expected at most +1 (precision guard against residual-of-noise inflation)"
    )
    noise_selected = sel_residual - strong_names
    assert not noise_selected, f"residual_passes=1 selected noise columns on a pure-strong bed: {sorted(noise_selected)}"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_residual_hard_vs_soft():
    """residual_exclude_top=6 (hard residual: pass 2 never sees the strong features) recovers weak
    recall at least as well as the soft variant (residual_exclude_top=0, pass 2 sees everything but
    the strong features' phi already dominates pass-1 loss). Informational floor per gt_09 sec 4.3 --
    if this consistently fails the plan calls for recording the numbers and adjusting, not forcing it."""
    X, y, _strong, weak = _make_mixed_strength_fixture()
    weak_names = {f"f{i}" for i in weak}

    sel_soft = _fit_selected(X, y, residual_passes=1, residual_merge="rescue", residual_exclude_top=0)
    sel_hard = _fit_selected(X, y, residual_passes=1, residual_merge="rescue", residual_exclude_top=6)

    soft_recall = len(weak_names & sel_soft)
    hard_recall = len(weak_names & sel_hard)
    assert hard_recall >= soft_recall, (
        f"residual_exclude_top=6 (hard) recovered {hard_recall}/6 weak features, expected >= "
        f"residual_exclude_top=0 (soft)'s {soft_recall}/6 -- see gt_09 sec 4.3/sec 5 for the recorded verdict"
    )
