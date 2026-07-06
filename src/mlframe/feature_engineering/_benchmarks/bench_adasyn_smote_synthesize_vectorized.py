"""Bench: adasyn_smote._adasyn_synthesize row-loop vs draw-then-vectorise gather+lerp.

``_adasyn_synthesize`` pre-draws the weighted ``src_indices`` (one ``rng.choice`` before the loop), then per
iteration draws the in-cluster neighbour (``rng.integers``) and the convex weight (``rng.random``) AND writes one
interpolated output row. The per-row numpy fancy-index + lerp dominates the loop. The (nbr, alpha) draws must stay
in the exact interleaved per-iteration order to keep the PCG64 stream (hence the synthetic cloud) bit-identical, so
only the gather + convex interpolation is hoisted out of the loop.

Measured (py3.14.3, n_min=500, d=30, k_smote=5, n_syn=5000, best-of-15):
  OLD (row loop)            ~100.1 ms
  NEW (draws + vec lerp)    ~68.8 ms   => ~1.46x faster, BIT-IDENTICAL.

Run: python -m mlframe.feature_engineering._benchmarks.bench_adasyn_smote_synthesize_vectorized
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _old(X, n_syn, k_smote, seed):
    n_min = X.shape[0]
    nn_pos = NearestNeighbors(n_neighbors=min(k_smote + 1, n_min)).fit(X)
    _d, ids_pos = nn_pos.kneighbors(X)
    rng = np.random.default_rng(seed)
    weights = np.full(n_min, 1.0 / n_min)
    src_indices = rng.choice(n_min, size=n_syn, p=weights)
    out = np.zeros((n_syn, X.shape[1]), dtype=np.float32)
    for i in range(n_syn):
        src_idx = src_indices[i]
        candidates = ids_pos[src_idx, 1 : min(k_smote + 1, n_min)]
        if candidates.size == 0:
            out[i] = X[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X[src_idx] + alpha * (X[nbr_idx] - X[src_idx])
    return out


def _new(X, n_syn, k_smote, seed):
    n_min = X.shape[0]
    nn_pos = NearestNeighbors(n_neighbors=min(k_smote + 1, n_min)).fit(X)
    _d, ids_pos = nn_pos.kneighbors(X)
    rng = np.random.default_rng(seed)
    weights = np.full(n_min, 1.0 / n_min)
    src_indices = rng.choice(n_min, size=n_syn, p=weights)
    k_hi = min(k_smote + 1, n_min)
    src = src_indices
    nbr = np.empty(n_syn, np.int64)
    alpha = np.empty(n_syn, np.float32)
    for i in range(n_syn):
        candidates = ids_pos[src[i], 1:k_hi]
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
    a = _old(X, 5000, 5, 1)
    b = _new(X, 5000, 5, 1)
    assert np.array_equal(a, b), "not bit-identical"  # nosec B101 - internal invariant check in src/mlframe/feature_engineering/_benchmarks, not reachable with untrusted input
    old = _best(_old, X, 5000, 5, 1)
    new = _best(_new, X, 5000, 5, 1)
    print(f"OLD={old*1000:.2f}ms  NEW={new*1000:.2f}ms  speedup={old/new:.2f}x  bit-identical=True")


if __name__ == "__main__":
    main()
