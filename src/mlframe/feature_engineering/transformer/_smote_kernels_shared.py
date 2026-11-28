"""Shared SMOTE-interpolation kernel for the clustered-SMOTE transformer family
(``cluster_smote.py`` / ``bgm_clustered_smote.py``): independently duplicated across those
modules, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def smote_within_cluster(X_cluster: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """SMOTE-interpolate within a single cluster of positives; a cluster of fewer than 2 rows is tiled instead of interpolated."""
    n_cluster = X_cluster.shape[0]
    if n_cluster < 2:
        return np.tile(X_cluster, (n_synthetic // max(1, n_cluster) + 1, 1))[:n_synthetic].astype(np.float32)
    return smote_synthesize_vectorized(X_cluster, n_synthetic, k_neighbors, seed)


def smote_synthesize_minority_vectorized(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Draw-then-vectorise SMOTE intra-minority interpolation with an explicit ``n_min < 2`` fallback
    (returns a copy, or an empty ``(0, 0)`` array): the variant shared by ``pseudo_smote.py`` / ``borderline_smote.py``.

    The gather+lerp itself lives in ``smote_synthesize_vectorized``, which this delegates to after the fallback."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1]), dtype=np.float32)  # same empty shape as the row-loop sibling (audit 08-F5)
    return smote_synthesize_vectorized(X_minority, n_synthetic, k_neighbors, seed)


def smote_synthesize_rowloop(X: np.ndarray, n_syn: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Row-loop SMOTE interpolation (the pre-vectorization reference): draws + writes one output row per
    iteration. See ``smote_synthesize_vectorized`` for the bit-identical, faster gather+lerp variant.

    NOT a production entry point -- no transformer calls this directly (use
    ``smote_synthesize_minority_vectorized`` for that). This pair
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
    """Draw-then-vectorise SMOTE interpolation of ``X`` onto itself: the one production kernel behind ``smote_within_cluster`` and
    ``smote_synthesize_minority_vectorized`` (audit 08-F15 removed the second row-loop copy).

    Bit-identical to ``smote_synthesize_rowloop`` for float32 data without duplicate rows, ~35% faster (``bench_pseudo_smote_synthesize_vectorized.py``). Self is
    excluded by index, so exact duplicates in ``X`` no longer make a copy of the source its own neighbour (audit 09-F40)."""
    from sklearn.neighbors import NearestNeighbors

    n_min = X.shape[0]
    k_used = min(k_neighbors + 1, n_min)
    ids = NearestNeighbors(n_neighbors=k_used).fit(X).kneighbors(X)[1]
    return smote_interpolate_from_neighbour_table(X, ids, k_used, n_syn, np.random.default_rng(seed))


def smote_interpolate_from_neighbour_table(X_minority: np.ndarray, ids: np.ndarray, k_used: int, n_synthetic: int, rng: np.random.Generator, src: np.ndarray | None = None) -> np.ndarray:
    """Draw-then-vectorise SMOTE interpolation over a precomputed neighbour table; self is excluded by index, not by column (audit 09-F40).

    ``ids`` is the ``(n_min, k_used)`` kNN table of ``X_minority``. Per synthetic row the generator is consumed in the order the former row loops did:
    the source row (drawn here when ``src`` is None, else given), then the neighbour pick, then alpha, so the output is bit-identical to them for float32
    input while the gather and the convex interpolation run as one vectorised pass (09-F28).
    """
    n_min = X_minority.shape[0]
    nbr = np.empty(n_synthetic, dtype=np.int64)
    alpha_dtype = X_minority.dtype if X_minority.dtype.kind == "f" else np.float64
    alpha = np.zeros(n_synthetic, dtype=alpha_dtype)
    src_out = np.empty(n_synthetic, dtype=np.int64)
    for i in range(n_synthetic):
        s = int(rng.integers(0, n_min)) if src is None else int(src[i])
        src_out[i] = s
        row = ids[s]
        candidates = row[row != s][: k_used - 1]
        if candidates.size == 0:
            nbr[i] = s
            continue
        nbr[i] = candidates[rng.integers(0, candidates.size)]
        alpha[i] = rng.random()
    x_src = X_minority[src_out]
    return np.asarray((x_src + alpha[:, None] * (X_minority[nbr] - x_src)).astype(np.float32))
