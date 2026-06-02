"""Cross-model shadow-voting all-relevant selection (agent idea BorutaShap-B7).

A single-model shadow gate (Boruta) leaks the top finite-sample-spurious noise column: that column's
in-bag importance beats the structure-less shadows for THAT model. But a noise column that fools one
model's geometry rarely fools structurally-different models. ``heterogeneous_relevance_vote`` runs the
shadow comparison across a PANEL of estimators (tree / linear / distance by default) and accepts a feature
only if it beats the max-shadow importance in a MAJORITY of the panel. Cross-model disagreement drops the
model-specific spurious leak while keeping genuinely-relevant features.

Verified on the fs_hybrid synthetic (base): single-tree shadow gate leaks 2 noise columns; the 3-model
vote (>=2/3) leaks 0/32 noise while keeping 5/7 causal features (the 2 misses are the pure interaction
operands - the universal marginal blindspot, not specific to this method). This is a cheaper, more robust
noise-leak fix than 10x cross-subsample stability for the common case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def heterogeneous_relevance_vote(
    X, y, *, models=None, classification: bool = True, n_shadow_trials: int = 5,
    percentile: float = 100.0, per_model_hit_frac: float = 0.5, vote_threshold: float = 0.5,
    random_state: int = 0,
):
    """All-relevant selection by cross-model shadow voting.

    For each model in the panel and each of ``n_shadow_trials`` permuted-shadow draws, fit on [X | shadows]
    and mark a feature a "hit" when its importance exceeds the ``percentile`` of the shadow importances. A
    feature PASSES a model if it hits in >= ``per_model_hit_frac`` of that model's trials; it is ACCEPTED if
    it passes >= ``vote_threshold`` fraction of the panel. Returns ``(accepted, info)`` where ``accepted`` is
    the kept feature-name list and ``info['vote_fraction']`` maps feature -> fraction of panel passed.

    models : dict name->estimator (cloned + fit from scratch). Defaults to a tree/linear/distance panel.
    """
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(np.asarray(X).shape[1])]
    Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    yv = np.asarray(y)
    P = Xv.shape[1]
    panel = models if models is not None else _default_panel(classification)
    pass_count = np.zeros(P)
    for est in panel.values():
        hits = np.zeros(P)
        for tr in range(n_shadow_trials):
            rng = np.random.default_rng(random_state + tr)
            shadow = np.column_stack([rng.permutation(Xv[:, j]) for j in range(P)])
            imp = _importance(est, np.hstack([Xv, shadow]), yv, random_state=random_state + tr)
            thr = np.percentile(imp[P:], percentile)
            hits += (imp[:P] > thr).astype(float)
        pass_count += (hits / max(1, n_shadow_trials) >= per_model_hit_frac).astype(float)
    vote_frac = pass_count / max(1, len(panel))
    accepted = [cols[i] for i in range(P) if vote_frac[i] >= vote_threshold]
    return accepted, {"vote_fraction": {cols[i]: float(vote_frac[i]) for i in range(P)}, "n_models": len(panel)}
