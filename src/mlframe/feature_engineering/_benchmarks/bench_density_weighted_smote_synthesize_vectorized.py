"""Bench: density_weighted_smote._density_weighted_smote_synthesize row-loop vs draw-then-vectorise gather+lerp.

``_density_weighted_smote_synthesize`` pre-draws the inverse-density-weighted ``src_indices`` (one ``rng.choice``
before the loop), then per iteration draws the in-cluster neighbour (``rng.integers``) and the convex weight
(``rng.random``) AND writes one interpolated output row. The per-row numpy fancy-index + lerp dominates the loop.
The (nbr, alpha) draws must stay in the exact interleaved per-iteration order to keep the PCG64 stream (hence the
synthetic cloud) bit-identical, so only the gather + convex interpolation is hoisted out of the loop.

Measured (py3.14.3, n_min=500, d=30, k_neighbors=5, n_syn=5000, best-of-15):
  OLD (row loop)            ~98.2 ms
  NEW (draws + vec lerp)    ~69.8 ms   => ~1.41x faster, BIT-IDENTICAL across seeds 1/2/7.

Run: python -m mlframe.feature_engineering._benchmarks.bench_density_weighted_smote_synthesize_vectorized
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _weights(X, k_neighbors):
    n_min = X.shape[0]
    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X)
    dists, ids = nn.kneighbors(X)
    mean_knn_dist = dists[:, 1:].mean(axis=1) + 1e-9
    w = mean_knn_dist / mean_knn_dist.sum()
    return w, ids, k_used


def _old(X, n_syn, k_neighbors, seed):
    w, ids, k_used = _weights(X, k_neighbors)
    rng = np.random.default_rng(seed)
    src_indices = rng.choice(X.shape[0], size=n_syn, p=w)
    out = np.zeros((n_syn, X.shape[1]), dtype=np.float32)
    for i in range(n_syn):
        src_idx = src_indices[i]
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X[src_idx] + alpha * (X[nbr_idx] - X[src_idx])
    return out


def _new(X, n_syn, k_neighbors, seed):
    w, ids, k_used = _weights(X, k_neighbors)
    rng = np.random.default_rng(seed)
    src = rng.choice(X.shape[0], size=n_syn, p=w)
    nbr = np.empty(n_syn, np.int64)
    alpha = np.empty(n_syn, np.float32)
    for i in range(n_syn):
        candidates = ids[src[i], 1:k_used]
        if candidates.size == 0:
            nbr[i] = src[i]
            alpha[i] = np.float32(0.0)
            continue
        nbr[i] = candidates[rng.integers(0, candidates.size)]
        alpha[i] = rng.random()
    x_src = X[src]
    return (x_src + alpha[:, None] * (X[nbr] - x_src)).astype(np.float32)


def _best(fn, *a, n=15):
    fn(*a)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 30)).astype(np.float32)
    for seed in (1, 2, 7):
        assert np.array_equal(_old(X, 5000, 5, seed), _new(X, 5000, 5, seed)), f"not bit-identical seed={seed}"
    old = _best(_old, X, 5000, 5, 1)
    new = _best(_new, X, 5000, 5, 1)
    print(f"OLD={old*1000:.2f}ms  NEW={new*1000:.2f}ms  speedup={old/new:.2f}x  bit-identical=True")


if __name__ == "__main__":
    main()
