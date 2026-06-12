"""Scalable correlated-feature clustering for ShapProxiedFS (the tens-of-thousands-of-features path).

The exhaustive-approx proxy is only tractable for ~10-25 columns. To apply it to wide data we first
collapse correlated features into denoised "units": cluster by |correlation|, replace each cluster
with one denoised representative (PC1 score / reliability-weighted mean of the standardized, sign-
aligned members, noise variance ~ sigma^2/k), then run SHAP + the subset search on the UNITS
(hundreds, not 10k). The model and TreeExplainer therefore never see the 10k raw columns - only the
correlation matmul does, which is a single GPU/numba op.

Clustering is unsupervised (no y) - the leakage firewall is the honest re-validation downstream,
which retrains on the real MEMBER columns of the selected units. Aggregates are FIT-TIME ONLY (they
drive ranking); the selector's final output is real member feature names, so ``transform`` stays a
plain name-based subset and needs no recipe replay.

Correlation backend dispatch (fastest path by size + HW):
  - dense GPU (cupy): standardized ``Z``; ``C = Z.T @ Z / n``; threshold -> edges. Used when
    ``f <= max_dense_features`` and a CUDA device is present.
  - dense CPU (numpy BLAS): same, when no GPU / cupy.
  - blocked (GPU or CPU): for very wide ``f`` the full ``f x f`` matrix would OOM, so correlation is
    streamed in feature-row blocks and only above-threshold edges are kept (never materialise f x f).
Connected components via a numba union-find over the edge list.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


@njit(cache=True)
def _uf_labels(n: int, ei: np.ndarray, ej: np.ndarray) -> np.ndarray:
    """Union-find connected-component labels (0..K-1, contiguous) over edges (ei[k], ej[k])."""
    parent = np.arange(n)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    for k in range(ei.shape[0]):
        a = find(ei[k])
        b = find(ej[k])
        if a != b:
            parent[b] = a
    # Flatten + relabel to contiguous ids.
    labels = np.empty(n, dtype=np.int64)
    remap = -np.ones(n, dtype=np.int64)
    nxt_id = 0
    for i in range(n):
        r = find(i)
        if remap[r] == -1:
            remap[r] = nxt_id
            nxt_id += 1
        labels[i] = remap[r]
    return labels


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    """Z-score columns (float32); constant columns map to all-zero (excluded from any edge)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 0.0, std, 1.0).astype(np.float32)
    Z = (X - mean) / std_safe
    Z[:, std == 0.0] = 0.0  # constant columns: no correlation with anything
    return np.ascontiguousarray(Z, dtype=np.float32)


def _edges_dense_gpu(Z, threshold, edge_cap):
    import cupy as cp

    n, f = Z.shape
    Zg = cp.asarray(Z)
    C = (Zg.T @ Zg) / np.float32(n)  # (f, f) correlation (Z already standardized)
    absC = cp.abs(C)
    iu = cp.triu_indices(f, k=1)
    mask = absC[iu] > np.float32(threshold)
    ei = iu[0][mask]
    ej = iu[1][mask]
    if ei.size > edge_cap:
        return None
    return cp.asnumpy(ei).astype(np.int64), cp.asnumpy(ej).astype(np.int64)


def _edges_blocked(Z, threshold, edge_cap, block, use_gpu):
    """Stream correlation in feature-row blocks; collect only above-threshold upper-triangle edges."""
    xp = None
    if use_gpu:
        try:
            import cupy as cp

            xp = cp
            Zb = cp.asarray(Z)
        except Exception:
            use_gpu = False
    if not use_gpu:
        xp = np
        Zb = Z
    n, f = Z.shape
    ei_parts, ej_parts = [], []
    total = 0
    for b0 in range(0, f, block):
        b1 = min(b0 + block, f)
        Cb = (Zb[:, b0:b1].T @ Zb) / np.float32(n)  # (blk, f)
        absCb = xp.abs(Cb)
        for r in range(b1 - b0):
            i = b0 + r
            row = absCb[r]
            js = xp.where(row > np.float32(threshold))[0]
            js = js[js > i]  # upper triangle only
            if js.size:
                if use_gpu:
                    js = xp.asnumpy(js)
                ei_parts.append(np.full(js.shape[0], i, dtype=np.int64))
                ej_parts.append(js.astype(np.int64))
                total += int(js.shape[0])
                if total > edge_cap:
                    return None
    if not ei_parts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ei_parts), np.concatenate(ej_parts)


def _resolve_gpu_min_features(default: int = 2000) -> int:
    """Smallest feature count at which the GPU dense path is preferred over CPU.

    Below this width the cold cupy/CUDA load + NVRTC kernel compile (~17s on the dev box) dwarfs
    even a single-threaded CPU `Z.T @ Z` on the bench (~0.3s at f=704/n=10000, ~50ms multithreaded).
    The blocked CPU path picks up >`max_dense_features`. The threshold is dispatcher-tunable per
    HW via ``pyutilz.performance.kernel_tuning.cache`` (key:
    ``mlframe.shap_proxied_fs.cluster_corr.gpu_min_features``).
    """
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_corr.gpu_min_features", default=default)
        return int(value)
    except Exception:
        return default


def cluster_correlated_features(
    X: np.ndarray,
    *,
    threshold: float = 0.7,
    use_gpu: str | bool = "auto",
    max_dense_features: int = 16000,
    block: int = 2048,
    edge_cap: int = 20_000_000,
    gpu_min_features: int | None = None,
) -> np.ndarray:
    """Label each feature with a cluster id (0..K-1) by single-linkage on |correlation| >= threshold.

    Constant columns and features with no above-threshold partner become singleton clusters.

    ``gpu_min_features`` (iter46): below this feature count the dense correlation runs on CPU even
    when a GPU is available, because the cupy/CUDA cold-start cost (one-time per process, ~17s on
    the dev box) is far larger than the CPU GEMM at small ``f``. ``None`` consults
    ``pyutilz.performance.kernel_tuning.cache`` (key
    ``mlframe.shap_proxied_fs.cluster_corr.gpu_min_features``); the default (2000) was calibrated
    on C4 (f=704 / n=10000 / cold cupy) where CPU dense ran in 0.3s vs the GPU path's 26.7s
    cumulative. After warm cupy is loaded for the process, the GPU path runs in ~50ms; users who
    keep cupy hot in the calling process can lower the threshold via the tuning cache.
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n, f = X.shape
    if f == 0:
        return np.empty(0, dtype=np.int64)
    Z = _standardize_columns(X)

    gpu = False
    if use_gpu in ("auto", True):
        try:
            import cupy as cp

            gpu = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            gpu = False
    if use_gpu is False:
        gpu = False
    # Small-f opt-out: even when CUDA is present, route the dense path to CPU if the GEMM is
    # tiny relative to the GPU cold-init cost. Forced GPU (`use_gpu=True` explicit) honours the
    # caller and skips this check.
    if gpu and use_gpu == "auto":
        gmin = gpu_min_features if gpu_min_features is not None else _resolve_gpu_min_features()
        if f < gmin:
            gpu = False

    edges = None
    if f <= max_dense_features:
        if gpu:
            edges = _edges_dense_gpu(Z, threshold, edge_cap)
        if edges is None:  # CPU dense or GPU cap exceeded -> blocked
            if not gpu:
                C = (Z.T @ Z) / np.float32(n)
                iu = np.triu_indices(f, k=1)
                mask = np.abs(C[iu]) > np.float32(threshold)
                ei, ej = iu[0][mask].astype(np.int64), iu[1][mask].astype(np.int64)
                edges = None if ei.size > edge_cap else (ei, ej)
    if edges is None:
        edges = _edges_blocked(Z, threshold, edge_cap, block, gpu)
    if edges is None:
        raise MemoryError(
            f"ShapProxiedFS clustering: >{edge_cap} correlation edges at threshold={threshold}. "
            f"Raise cluster_corr_threshold to merge fewer features."
        )
    ei, ej = edges
    return _uf_labels(f, ei, ej)


def build_unit_matrix(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    weighting: str = "pca_pc1",
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Collapse each cluster into one denoised representative column.

    Returns
    -------
    units : (n, n_units) float64
        Singleton clusters pass through their raw column; multi-member clusters become a denoised
        aggregate (standardized, sign-aligned members combined by ``weighting``).
    unit_to_members : list of int arrays
        Original feature column-indices behind each unit (for honest re-validation on real columns).
    unit_kind : list[str]
        "single" or "cluster:k" tag per unit (diagnostics).
    """
    from mlframe.feature_selection.filters import (
        derive_cluster_weights as _derive_weights,
        standardize_align_cluster as _standardize_align,
        apply_cluster_aggregate_nonlinear as _apply_nonlinear,
    )

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[0]
    n_clusters = int(labels.max()) + 1 if labels.size else 0

    members_by_cluster: list[list[int]] = [[] for _ in range(n_clusters)]
    for col, lab in enumerate(labels):
        members_by_cluster[int(lab)].append(col)

    unit_cols, unit_to_members, unit_kind = [], [], []
    for members in members_by_cluster:
        if len(members) == 1:
            unit_cols.append(X[:, members[0]])
            unit_to_members.append(np.array(members, dtype=np.int64))
            unit_kind.append("single")
            continue
        M = X[:, members]
        Z, _mean, _std, _signs = _standardize_align(M, 0)  # ref = first member
        w = _derive_weights(Z, weighting)
        # 2026-06-03 (audit gap-04): when the combiner is non-linear,
        # _derive_weights returns None for ALL of median/median_z/signed_max_abs/
        # signed_l2_sum. The old ``np.median`` fallback silently collapsed
        # signed_max_abs and signed_l2_sum to a plain median, diverging from the
        # canonical _cluster_aggregate dispatch. Route them through the shared
        # row-reducer so the requested non-linear aggregator is actually applied.
        agg = _apply_nonlinear(Z, weighting) if w is None else (Z @ w)
        unit_cols.append(np.ascontiguousarray(agg, dtype=np.float64))
        unit_to_members.append(np.array(members, dtype=np.int64))
        unit_kind.append(f"cluster:{len(members)}")

    units = np.column_stack(unit_cols) if unit_cols else np.empty((n, 0), dtype=np.float64)
    return units, unit_to_members, unit_kind


def cluster_summary(unit_to_members: list[np.ndarray]) -> dict:
    sizes = np.array([len(m) for m in unit_to_members], dtype=np.int64)
    return dict(
        n_units=int(len(unit_to_members)),
        n_singletons=int((sizes == 1).sum()),
        n_multi_clusters=int((sizes > 1).sum()),
        largest_cluster=int(sizes.max()) if sizes.size else 0,
        n_features=int(sizes.sum()),
    )
