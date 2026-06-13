"""Cross-model shadow-voting all-relevant selection (agent idea BorutaShap-B7).

A single-model shadow gate (Boruta) leaks the top finite-sample-spurious noise column: that column's
in-bag importance beats the structure-less shadows for THAT model. But a noise column that fools one
model's geometry rarely fools structurally-different models. ``heterogeneous_relevance_vote`` runs the
shadow comparison across a PANEL of estimators (tree / linear / distance by default) and accepts a feature
only if it beats the max-shadow importance in a MAJORITY of the panel. Cross-model disagreement drops the
model-specific spurious leak while keeping genuinely-relevant features.

POSITIONING (measured on the 6-scenario x 2-seed fs_hybrid bed, round-2): this is a HIGH-PRECISION /
PARSIMONY selector, NOT a downstream-AUC maximiser. The cross-model majority drives accepted-noise to ~0
(vs single-fit gini-Boruta's ~1.6) and yields compact sets (~7-9 vs ~17 features), but the same strict
agreement also drops weakly-relevant features, so its mean honest-holdout AUC (~0.74) trails single-fit
Boruta (~0.76) and it wins only ~2/12 cells. Use it when you want a clean, compact, low-false-positive
all-relevant set (e.g. interpretability, or a denoise pre-stage feeding a downstream that tolerates lost
weak signal); use plain Boruta when downstream AUC is the objective. n_shadow_trials: cross-MODEL
disagreement is the mechanism (not cross-trial), so 2-3 trials match 5 at lower cost (default 3).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _importance(est, X, y, *, n_perm_repeats: int = 3, random_state: int = 0) -> np.ndarray:
    """Per-feature importance for a fitted-from-scratch ``est`` on (X, y): native feature_importances_,
    else |coef_| (max over classes), else permutation importance on a subsample (model-agnostic fallback)."""
    from sklearn.base import clone
    m = clone(est)
    m.fit(X, y)
    if hasattr(m, "feature_importances_"):
        return np.abs(np.asarray(m.feature_importances_, dtype=float))
    if hasattr(m, "coef_"):
        c = np.abs(np.asarray(m.coef_, dtype=float))
        return c.max(axis=0) if c.ndim > 1 else c
    from sklearn.inspection import permutation_importance
    n = X.shape[0]
    idx = np.arange(n) if n <= 1000 else np.random.default_rng(random_state).choice(n, 1000, replace=False)
    pi = permutation_importance(m, X[idx], np.asarray(y)[idx], n_repeats=n_perm_repeats, random_state=random_state)
    return np.asarray(pi.importances_mean, dtype=float)


def _default_panel(classification: bool):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if classification:
        return {
            "tree": RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=0),
            "linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "distance": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        }
    return {
        "tree": RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=0),
        "linear": make_pipeline(StandardScaler(), Ridge()),
        "distance": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25)),
    }


def _cv_skill(est, X, y, *, classification: bool, folds: int, random_state: int) -> float:
    """Above-chance CV skill of a panel member: ROC-AUC - 0.5 (classification) or R2 - 0 (regression),
    clamped to >= 0. Used to downweight a structurally-blind member (e.g. a linear model on monotone-but-
    nonlinear signal) so its veto cannot sink a feature the rest of the panel confirms."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    if classification:
        scoring = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr_weighted"
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        chance = 0.5
    else:
        scoring = "r2"
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        chance = 0.0
    try:
        score = float(np.mean(cross_val_score(clone(est), X, y, cv=cv, scoring=scoring)))
    except Exception as e:
        # A genuine scoring failure (too few rows/folds, single-class fold) legitimately maps to
        # zero above-chance skill, but a silent 0.0 also hides real bugs (shape/dtype/wiring) that
        # would otherwise mis-weight the panel vote. Log so the failure is visible, then fall back.
        logger.warning("_cv_skill: CV scoring failed for %s (%s: %s); treating skill as 0.0",
                       type(est).__name__, type(e).__name__, e)
        return 0.0
    return max(0.0, score - chance)


def heterogeneous_relevance_vote(
    X, y, *, models=None, classification: bool = True, n_shadow_trials: int = 3,
    percentile: float = 100.0, per_model_hit_frac: float = 0.5, vote_threshold: float = 0.5,
    weight_by_cv_skill: bool = False, cv_skill_folds: int = 3, cv_skill_floor: float = 0.05,
    random_state: int = 0,
):
    """All-relevant selection by cross-model shadow voting.

    For each model in the panel and each of ``n_shadow_trials`` permuted-shadow draws, fit on [X | shadows]
    and mark a feature a "hit" when its importance exceeds the ``percentile`` of the shadow importances. A
    feature PASSES a model if it hits in >= ``per_model_hit_frac`` of that model's trials; it is ACCEPTED if
    its (optionally skill-weighted) panel vote-fraction is >= ``vote_threshold``. Returns ``(accepted, info)``
    where ``accepted`` is the kept feature-name list and ``info['vote_fraction']`` maps feature -> the panel
    fraction that passed.

    weight_by_cv_skill : when True, each member's vote is weighted by its above-chance CV skill (clamped to
        >= ``cv_skill_floor``) instead of counting equally. The intent was to fix the panel's recall failure
        mode by downweighting a member blind to a feature's functional form. MEASURED (round2_hetero_skillweight
        _bench.py, 6 scenarios x 2 seeds): it changed the selection in 0/12 cells (mean AUC identical to equal
        weighting, 0.7418), because on this bed every panel member keeps a BALANCED CV skill (~0.20-0.29) even on
        monotone / weakmix -- no member is actually near-chance, so there is nothing to downweight. hetero_vote's
        recall deficit is structural to the cross-model AGREEMENT requirement (a weak feature is not confidently
        above-shadow in a majority of models), not the fault of one blind voter, so skill-weighting cannot close
        it. Kept as an off-by-default option for datasets that DO contain a near-chance member; not a recall fix.
        ``cv_skill_folds`` sets the CV used to estimate skill; ``cv_skill_floor`` keeps a weak-but-not-useless
        member from being fully silenced. Equal weighting (the default) keeps the cheap, calibration-free path.

    models : dict name->estimator (cloned + fit from scratch). Defaults to a tree/linear/distance panel.
    """
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(np.asarray(X).shape[1])]
    Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    yv = np.asarray(y)
    P = Xv.shape[1]
    panel = models if models is not None else _default_panel(classification)
    # The shadow seed (random_state + tr) is model-INDEPENDENT, so every panel member redraws the same
    # n_shadow_trials shadow matrices from scratch. Build each [X | shadow] once and reuse across the panel:
    # bit-identical (same seed -> same permutation), removing (n_models-1)/n_models of the shadow work.
    augmented = []
    for tr in range(n_shadow_trials):
        rng = np.random.default_rng(random_state + tr)
        shadow = np.column_stack([rng.permutation(Xv[:, j]) for j in range(P)])
        augmented.append(np.hstack([Xv, shadow]))
    passes, weights = [], []
    for est in panel.values():
        hits = np.zeros(P)
        for tr in range(n_shadow_trials):
            imp = _importance(est, augmented[tr], yv, random_state=random_state + tr)
            thr = np.percentile(imp[P:], percentile)
            hits += (imp[:P] > thr).astype(float)
        passes.append((hits / max(1, n_shadow_trials) >= per_model_hit_frac).astype(float))
        if weight_by_cv_skill:
            weights.append(max(cv_skill_floor, _cv_skill(est, Xv, yv, classification=classification,
                                                          folds=cv_skill_folds, random_state=random_state)))
        else:
            weights.append(1.0)
    W = np.asarray(weights, dtype=float)
    vote_frac = (W[:, None] * np.vstack(passes)).sum(axis=0) / max(W.sum(), 1e-12)
    accepted = [cols[i] for i in range(P) if vote_frac[i] >= vote_threshold]
    info = {"vote_fraction": {cols[i]: float(vote_frac[i]) for i in range(P)}, "n_models": len(panel),
            "model_weights": {name: float(w) for name, w in zip(panel.keys(), W)}}
    return accepted, info
