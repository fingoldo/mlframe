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

# Min proxy width at which the custom numba interaction kernel is selected over the ``shap`` library.
# Measured (xgboost regr, 1000 rows, 200 trees): numba beats shap's ``shap_interaction_values`` at
# every width from P=10 (~1.5x) upward, so the interaction crossover is far LOWER than the main-effect
# one (the shap interaction call is much heavier than its main-effect call). We keep a small floor so
# the numba JIT warmup is not paid on trivial widths. Tunable per-HW via kernel_tuning_cache.
_INTERACTION_NUMBA_MIN_FEATURES = 8

# Min problem size (rows * P^2 interaction-tensor cells) at which the cupy/CUDA interaction kernel is
# preferred over numba on a CUDA box. The GPU kernel carries large per-thread local-memory scratch (it
# spills on small GPUs) so it only pays off once the per-sample tree*cond_feat work amortises the
# upload + spill; below this we stay on numba (no JIT/launch overhead win). Tunable via
# kernel_tuning_cache key ``interaction_gpu_min_cells``.
_INTERACTION_GPU_MIN_CELLS = 3_000 * 40 * 40  # ~4.8M cells (e.g. 3000 rows x P=40)


def _interaction_numba_min_features() -> int:
    """Crossover width for routing interactions to the numba kernel, from kernel_tuning_cache if set."""
    try:
        from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("interaction_numba_min_features"):
                return int(entry["interaction_numba_min_features"])
    except Exception:
        pass
    return _INTERACTION_NUMBA_MIN_FEATURES


def _interaction_gpu_min_cells() -> int:
    """Min ``n * P^2`` to route the interaction tensor to the GPU kernel, from kernel_tuning_cache if set."""
    try:
        from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("interaction_gpu_min_cells"):
                return int(entry["interaction_gpu_min_cells"])
    except Exception:
        pass
    return _INTERACTION_GPU_MIN_CELLS


def _interaction_tensor_numba(est, X, *, classification):
    """Custom numba TreeSHAP interaction tensor for a supported xgboost estimator, else ``None``.

    Returns ``(Phi (n,P,P) float64, base (n,) float64)`` in margin / log-odds space, matching the shap
    library path's contract; ``None`` if the model is unsupported (caller falls back to ``shap``)."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection._shap_proxy_treeshap_interactions import interaction_tensor_numba

    base_est = _unwrap_estimator(est)
    ensemble = extract_ensemble(base_est)
    if ensemble is None:
        return None
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    Phi, _phi, base = interaction_tensor_numba(ensemble, Xv)
    n = Phi.shape[0]
    return Phi, np.full(n, float(base), dtype=np.float64)


def _interaction_tensor_gpu(est, X, *, classification):
    """cupy/CUDA TreeSHAP interaction tensor for a supported xgboost estimator, else ``None``.

    Same ``(Phi (n,P,P), base (n,))`` contract as ``_interaction_tensor_numba``. Returns ``None`` (so the
    caller can fall back to numba then shap) if the model is unsupported or the ensemble exceeds the GPU
    kernel's depth/width scratch caps."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    base_est = _unwrap_estimator(est)
    ensemble = extract_ensemble(base_est)
    if ensemble is None:
        return None
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    Phi, _phi, base = interaction_tensor_gpu(ensemble, Xv)
    n = Phi.shape[0]
    return Phi, np.full(n, float(base), dtype=np.float64)


def compute_interaction_tensor(model_template, X, y, *, classification, rng=None, backend="auto"):
    """Fit one model on X and return ``(Phi, base)`` where Phi is (n, P, P) SHAP interaction values
    (positive class for binary) and base is the scalar expected value broadcast to (n,).

    ``backend`` ("auto" default) routes between the cupy/CUDA TreeSHAP interaction kernel (on a CUDA box
    once the problem is large enough), the custom numba TreeSHAP interaction kernel (for supported
    xgboost / lightgbm models on wider proxy widths) and the ``shap`` library (always-correct fallback).
    The GPU and
    numba kernels match each other to ~machine eps and ``shap.shap_interaction_values`` to ~1e-4; both
    are far faster on wide proxies, where the interaction tensor is the search hotspot.

    ``"shap"`` / ``"treeshap_numba"`` / ``"treeshap_gpu"`` force a path. The GPU path falls back to numba
    (depth/width cap, missing cupy or device hiccup); the numba path falls back to shap for unsupported
    models."""
    rng = np.random.default_rng(0) if rng is None else rng
    est = _fit_one(model_template, X, y, classification, int(rng.integers(0, 2**31 - 1)))

    use_gpu = False
    use_numba = False
    if backend == "treeshap_gpu":
        use_gpu = True
    elif backend == "treeshap_numba":
        use_numba = True
    elif backend == "auto":
        from mlframe.feature_selection._shap_proxy_treeshap import (
            is_supported_lightgbm, is_supported_xgboost)

        base_est = _unwrap_estimator(est)
        supported = is_supported_xgboost(base_est) or is_supported_lightgbm(base_est)
        P = X.shape[1]
        if supported and P >= _interaction_numba_min_features():
            use_numba = True
            # Prefer the GPU interaction kernel on a CUDA box once the tensor is large enough that the
            # per-sample tree*cond-feat work amortises the upload + local-mem spill.
            try:
                from mlframe.feature_selection._shap_proxy_treeshap_interactions_gpu import (
                    gpu_interactions_available)

                if gpu_interactions_available() and X.shape[0] * P * P >= _interaction_gpu_min_cells():
                    use_gpu = True
            except Exception:
                pass

    if use_gpu:
        try:
            out = _interaction_tensor_gpu(est, X, classification=classification)
            if out is not None:
                return out
        except Exception:
            pass  # device/cupy/cap hiccup -> numba then shap (never lose the result)
        out = _interaction_tensor_numba(est, X, classification=classification)
        if out is not None:
            return out
    elif use_numba:
        out = _interaction_tensor_numba(est, X, classification=classification)
        if out is not None:
            return out
        # Unsupported despite routing -> fall through to shap.

    import shap

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
