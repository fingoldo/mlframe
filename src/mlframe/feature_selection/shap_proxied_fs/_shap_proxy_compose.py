"""Novel composition patterns for ShapProxiedFS (Batch D, novel A/B).

A) ``proposal_generator``: run the cheap proxy (no honest re-validation) to emit its top-N candidate
   subsets as a SHAP-guided seed set for an expensive honest search (RFECV's MBH optimiser, a genetic
   wrapper, ...). "SHAP proposes, honest retraining disposes" - turns blind search into guided search.

B) ``per_fold_stability_select``: run ShapProxiedFS once per outer CV fold and aggregate how often
   each feature survives. The frequency table is a Boruta-like confidence signal; a majority vote
   yields a final subset robust to any single fold's winner's-curse. Each fold's measured proxy
   fidelity (trust Spearman) weights its vote.

Both are thin orchestration over the existing selector - self-contained (no training-suite coupling).
"""

from __future__ import annotations

from collections import defaultdict

from typing import Any

import numpy as np
import pandas as pd


def proposal_generator(
    X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series, *, classification: bool = True, optimizer: str = "genetic", top_n: int = 30, **shap_kwargs
) -> list[tuple[float, tuple[Any, ...]]]:
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


def fold_vote_weights(fidelities: list, weight_by_fidelity: bool) -> list:
    """Per-fold vote weights. A fold whose trust fidelity is not finite (too few anchors to rank) gets the mean weight of the finite folds, not 0, so it
    is neither silenced nor allowed to dominate; negative fidelity counts as 0; when every weight is 0 the vote falls back to one fold, one vote."""
    if not weight_by_fidelity or not fidelities:
        return [1.0] * len(fidelities)
    finite = [max(f, 0.0) for f in fidelities if np.isfinite(f)]
    neutral = float(np.mean(finite)) if finite else 1.0
    weights = [max(f, 0.0) if np.isfinite(f) else neutral for f in fidelities]
    return weights if sum(weights) > 0 else [1.0] * len(fidelities)


def per_fold_stability_select(
    X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series, *, classification: bool = True, n_folds: int = 5, vote_threshold: float = 0.5,
    weight_by_fidelity: bool = True, random_state: int = 0, **shap_kwargs
) -> dict[str, Any]:
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
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=random_state) if classification else KFold(n_folds, shuffle=True, random_state=random_state)
    split = splitter.split(X, y) if classification else splitter.split(X)

    selected_by_fold: list = []
    fidelities: list = []
    for fold, (tr, _) in enumerate(split):
        sel = ShapProxiedFS(classification=classification, random_state=random_state + fold, verbose=False, **shap_kwargs)
        sel.fit(X.iloc[tr].reset_index(drop=True), y[tr])
        fidelities.append(float(sel.shap_proxy_report_.get("trust", {}).get("spearman", 1.0) or 0.0))
        selected_by_fold.append([str(f) for f in sel.selected_features_])

    weights = fold_vote_weights(fidelities, weight_by_fidelity)
    weighted: defaultdict = defaultdict(float)
    per_fold = []
    for fold, (chosen, w) in enumerate(zip(selected_by_fold, weights)):
        for f in chosen:
            weighted[f] += w
        per_fold.append(dict(fold=fold, selected=list(chosen), fidelity=fidelities[fold]))

    total_w = float(sum(weights))
    if total_w == 0.0:
        total_w = 1.0
    frequency = {f: v / total_w for f, v in weighted.items()}
    ensemble = sorted([f for f, fr in frequency.items() if fr >= vote_threshold])
    return dict(frequency=frequency, ensemble=ensemble, per_fold=per_fold, n_folds=n_folds)
