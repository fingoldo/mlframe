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
