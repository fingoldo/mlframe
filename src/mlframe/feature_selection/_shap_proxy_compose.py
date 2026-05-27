"""Novel composition patterns for ShapProxiedFS (Batch D, novel A/B).

A) ``proposal_generator``: run the cheap proxy (no honest re-validation) to emit its top-N candidate
   subsets as a SHAP-guided seed set for an expensive honest search (RFECV's MBH optimiser, a genetic
   wrapper, ...). "SHAP proposes, honest retraining disposes" -- turns blind search into guided search.

B) ``per_fold_stability_select``: run ShapProxiedFS once per outer CV fold and aggregate how often
   each feature survives. The frequency table is a Boruta-like confidence signal; a majority vote
   yields a final subset robust to any single fold's winner's-curse. Each fold's measured proxy
   fidelity (trust Spearman) weights its vote.

Both are thin orchestration over the existing selector -- self-contained (no training-suite coupling).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def proposal_generator(X, y, *, classification=True, optimizer="genetic", top_n=30, **shap_kwargs):
    """Return up to ``top_n`` ``(proxy_loss, feature_names)`` candidate subsets from the cheap proxy
    (honest re-validation / trust guard / ablation all disabled for speed). Use as seeds for an
    honest wrapper search."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(
        classification=classification, optimizer=optimizer, revalidate=False, trust_guard=False,
        run_importance_ablation=False, use_bias_corrector=False, within_cluster_refine=False,
        top_n=top_n, verbose=False, **shap_kwargs)
    sel.fit(X, y)
    cands = sel.shap_proxy_report_.get("candidates", [])
    return [(c["proxy_loss"], tuple(c["features"])) for c in cands]


def per_fold_stability_select(
    X, y, *, classification=True, n_folds=5, vote_threshold=0.5, weight_by_fidelity=True,
    random_state=0, **shap_kwargs):
    """Run ShapProxiedFS on each of ``n_folds`` train splits; return a stability report.

    Returns dict with:
      - ``frequency``: feature -> fraction of folds that selected it (fidelity-weighted if enabled),
      - ``ensemble``: features whose (weighted) frequency >= ``vote_threshold``, sorted,
      - ``per_fold``: each fold's selected features + its trust Spearman.
    """
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold

    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    splitter = (StratifiedKFold(n_folds, shuffle=True, random_state=random_state) if classification
                else KFold(n_folds, shuffle=True, random_state=random_state))
    split = splitter.split(X, y) if classification else splitter.split(X)

    weighted = defaultdict(float)
    total_w = 0.0
    per_fold = []
    for fold, (tr, _) in enumerate(split):
        sel = ShapProxiedFS(classification=classification, random_state=random_state + fold,
                            verbose=False, **shap_kwargs)
        sel.fit(X.iloc[tr].reset_index(drop=True), y[tr])
        fidelity = float(sel.shap_proxy_report_.get("trust", {}).get("spearman", 1.0) or 0.0)
        w = max(fidelity, 0.0) if weight_by_fidelity else 1.0
        w = w if np.isfinite(w) else 0.0
        total_w += w
        for f in sel.selected_features_:
            weighted[str(f)] += w
        per_fold.append(dict(fold=fold, selected=list(map(str, sel.selected_features_)), fidelity=fidelity))

    total_w = total_w or 1.0
    frequency = {f: v / total_w for f, v in weighted.items()}
    ensemble = sorted([f for f, fr in frequency.items() if fr >= vote_threshold])
    return dict(frequency=frequency, ensemble=ensemble, per_fold=per_fold, n_folds=n_folds)
