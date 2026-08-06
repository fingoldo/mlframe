"""Shared SMOTE-interpolation kernel for the clustered-SMOTE transformer family
(``cluster_smote.py`` / ``bgm_clustered_smote.py``): independently duplicated across those
modules, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def smote_within_cluster(X_cluster: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """SMOTE-interpolate within a single cluster of positives."""
    n_cluster = X_cluster.shape[0]
    if n_cluster < 2:
        return np.tile(X_cluster, (n_synthetic // max(1, n_cluster) + 1, 1))[:n_synthetic].astype(np.float32)
    from sklearn.neighbors import NearestNeighbors

    k_used = min(k_neighbors + 1, n_cluster)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_cluster)
    _dists, ids = nn.kneighbors(X_cluster)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_synthetic, X_cluster.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = rng.integers(0, n_cluster)
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X_cluster[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X_cluster[src_idx] + alpha * (X_cluster[nbr_idx] - X_cluster[src_idx])
    return out


def smote_synthesize_minority_rowloop(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Row-loop SMOTE intra-minority interpolation with an explicit ``n_min < 2`` fallback (returns a copy,
    or an empty ``(0, d)`` array): the variant shared by ``multiscale_smote.py`` / ``active_virtual.py``."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1] if n_min > 0 else 1), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors

    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_synthetic, X_minority.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = rng.integers(0, n_min)
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X_minority[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X_minority[src_idx] + alpha * (X_minority[nbr_idx] - X_minority[src_idx])
    return out.astype(np.float32)


def smote_synthesize_minority_vectorized(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Draw-then-vectorise SMOTE intra-minority interpolation with an explicit ``n_min < 2`` fallback
    (returns a copy, or an empty ``(0, 0)`` array): the variant shared by ``pseudo_smote.py`` / ``borderline_smote.py``."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, 0), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors

    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    # Draw indices/alphas in the exact per-iteration order (interleaved src/nbr/alpha) the PCG64 stream
    # produced before, then do the gather+lerp as one vectorised pass -- bit-identical to the row loop.
    src = np.empty(n_synthetic, dtype=np.int64)
    nbr = np.empty(n_synthetic, dtype=np.int64)
    alpha = np.empty(n_synthetic, dtype=np.float32)
    for i in range(n_synthetic):
        s = rng.integers(0, n_min)
        candidates = ids[s, 1:k_used]
        src[i] = s
        if candidates.size == 0:
            nbr[i] = s
            alpha[i] = np.float32(0.0)
            continue
        nbr[i] = candidates[rng.integers(0, candidates.size)]
        alpha[i] = rng.random()
    x_src = X_minority[src]
    return np.asarray((x_src + alpha[:, None] * (X_minority[nbr] - x_src)).astype(np.float32))


def smote_synthesize_rowloop(X: np.ndarray, n_syn: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Row-loop SMOTE interpolation (the pre-vectorization reference): draws + writes one output row per
    iteration. See ``smote_synthesize_vectorized`` for the bit-identical, faster gather+lerp variant.

    NOT a production entry point -- no transformer calls this directly (use
    ``smote_synthesize_minority_rowloop``/``smote_synthesize_minority_vectorized`` for that). This pair
    exists solely as the ground-truth reference ``bench_borderline_smote_synthesize_vectorized.py`` and
    ``bench_pseudo_smote_synthesize_vectorized.py`` diff their optimized kernels against.
    """
    n_min = X.shape[0]
    from sklearn.neighbors import NearestNeighbors

    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X)
    _d, ids = nn.kneighbors(X)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_syn, X.shape[1]), dtype=np.float32)
    for i in range(n_syn):
        s = rng.integers(0, n_min)
        cand = ids[s, 1:k_used]
        if cand.size == 0:
            out[i] = X[s]
            continue
        nbr = cand[rng.integers(0, cand.size)]
        a = rng.random()
        out[i] = X[s] + a * (X[nbr] - X[s])
    return out.astype(np.float32)


def smote_synthesize_vectorized(X: np.ndarray, n_syn: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Draw-then-vectorise SMOTE interpolation: identical (src, nbr, alpha) RNG draw order to
    ``smote_within_cluster``, but the gather+lerp is hoisted out of the per-iteration loop -- bit-identical,
    ~35% faster (see ``bench_pseudo_smote_synthesize_vectorized.py`` / ``bench_borderline_smote_synthesize_vectorized.py``).

    NOT a production entry point (see ``smote_synthesize_rowloop``'s docstring)."""
    n_min = X.shape[0]
    from sklearn.neighbors import NearestNeighbors

    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X)
    _d, ids = nn.kneighbors(X)
    rng = np.random.default_rng(seed)
    src = np.empty(n_syn, np.int64)
    nbr = np.empty(n_syn, np.int64)
    alpha = np.empty(n_syn, np.float32)
    for i in range(n_syn):
        s = rng.integers(0, n_min)
        cand = ids[s, 1:k_used]
        src[i] = s
        if cand.size == 0:
            nbr[i] = s
            alpha[i] = np.float32(0.0)
            continue
        nbr[i] = cand[rng.integers(0, cand.size)]
        alpha[i] = rng.random()
    x_src = X[src]
    return np.asarray((x_src + alpha[:, None] * (X[nbr] - x_src)).astype(np.float32))
