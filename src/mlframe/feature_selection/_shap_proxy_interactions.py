"""Interaction-aware coalition value for ShapProxiedFS (lever #5).

The plain coalition value ``base + sum_{j in S} phi_j`` folds each pairwise interaction into the two
features' main effects, so it cannot tell that dropping ONE partner of an interacting pair destroys
the joint signal -- and for a pure interaction (XOR) each partner's main effect is ~0, so the
informative pair looks like noise. Using SHAP *interaction* values fixes this: by the additivity of
interaction values ``phi_j = sum_k Phi_jk``, the coalition value restricted to S is

    base + sum_{i in S} sum_{k in S} Phi_ik   (interactions WITH excluded features dropped)

which correctly credits a subset for the joint signal it can actually produce. This is the user's
research ``get_total_contributions`` formula, done as the search objective.

Interaction tensors are O(P^2) memory and slower to compute, so this is opt-in and bounded to a small
proxy-column count (post-clustering / pre-screen, P is already small). We compute the tensor with one
in-sample model (a ranking refinement; the honest re-validation downstream stays OOF-disjoint), and
search with an exhaustive interaction scan for small P plus a greedy-forward pass that recovers
interacting pairs the main-effect search would miss.
"""

from __future__ import annotations

import itertools

import numpy as np

from mlframe.feature_selection._shap_proxy_explain import _unwrap_estimator, _fit_one
from mlframe.feature_selection._shap_proxy_objective import proxy_loss, resolve_metric


def compute_interaction_tensor(model_template, X, y, *, classification, rng=None):
    """Fit one model on X and return ``(Phi, base)`` where Phi is (n, P, P) SHAP interaction values
    (positive class for binary) and base is the scalar expected value broadcast to (n,)."""
    import shap

    rng = np.random.default_rng(0) if rng is None else rng
    est = _fit_one(model_template, X, y, classification, int(rng.integers(0, 2**31 - 1)))
    explainer = shap.TreeExplainer(_unwrap_estimator(est), feature_perturbation="tree_path_dependent")
    Phi = explainer.shap_interaction_values(X)
    base = explainer.expected_value
    if isinstance(Phi, list):  # binary -> positive class
        Phi = Phi[1] if len(Phi) == 2 else Phi[0]
        base = base[1] if (np.ndim(base) > 0 and len(Phi) == 2) else (base[0] if np.ndim(base) > 0 else base)
    Phi = np.asarray(Phi, dtype=np.float64)
    if Phi.ndim == 4:  # (n, P, P, classes) -> positive class
        Phi = Phi[:, :, :, -1]
    base = float(np.asarray(base, dtype=np.float64).ravel()[0]) if np.ndim(base) > 0 else float(base)
    n = Phi.shape[0]
    return Phi, np.full(n, base, dtype=np.float64)


def interaction_margin(Phi: np.ndarray, base: np.ndarray, idx) -> np.ndarray:
    """``base + sum_{i,k in S} Phi_ik`` -- coalition value keeping only within-subset interactions."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return base.copy()
    sub = Phi[:, idx][:, :, idx]
    return base + sub.sum(axis=(1, 2))


def interaction_subset_loss(Phi, base, y, idx, metric) -> float:
    return proxy_loss(interaction_margin(Phi, base, idx), y, metric)


def interaction_top_n(
    Phi, base, y, *, classification, metric=None, min_card=1, max_card=None, top_n=30,
    exhaustive_max=16,
):
    """Rank subsets by the interaction-aware coalition loss. Exhaustive for small P; always also runs
    a greedy-forward pass (which recovers interacting pairs a main-effect search would miss)."""
    metric = resolve_metric(classification, metric)
    P = Phi.shape[1]
    max_card = P if max_card is None else min(max_card, P)
    cache: dict[tuple[int, ...], float] = {}

    def loss(idx):
        key = tuple(sorted(int(i) for i in idx))
        if not key:
            return float("inf")
        v = cache.get(key)
        if v is None:
            v = interaction_subset_loss(Phi, base, y, list(key), metric)
            cache[key] = v
        return v

    if P <= exhaustive_max:
        for r in range(min_card, max_card + 1):
            for comb in itertools.combinations(range(P), r):
                loss(comb)
    else:  # greedy forward only when too wide to enumerate
        current = ()
        best = float("inf")
        remaining = set(range(P))
        while remaining and len(current) < max_card:
            cand, cl = None, float("inf")
            for j in remaining:
                l = loss(current + (j,))
                if l < cl:
                    cl, cand = l, j
            if cand is None or cl >= best:
                break
            current = tuple(sorted(current + (cand,)))
            best = cl
            remaining.discard(cand)
    # Always run greedy-forward to surface interacting pairs (cheap, dedup via cache).
    current = ()
    best = float("inf")
    remaining = set(range(P))
    while remaining and len(current) < max_card:
        cand, cl = None, float("inf")
        for j in remaining:
            l = loss(current + (j,))
            if l < cl:
                cl, cand = l, j
        if cand is None or cl >= best:
            break
        current = tuple(sorted(current + (cand,)))
        best = cl
        remaining.discard(cand)

    items = [(v, k) for k, v in cache.items() if np.isfinite(v)]
    items.sort(key=lambda t: t[0])
    return items[:top_n]
