"""Bench: borderline_smote._smote_synthesize_from row-loop vs draw-then-vectorise gather+lerp.

The SMOTE interpolation drew (src, nbr, alpha) per iteration AND wrote one output row per iteration.
The per-row numpy fancy-index + lerp dominated (~81% tottime at n_synthetic=5000, vs ~18% for the
NearestNeighbors fit). The RNG draws must stay in the exact interleaved per-iteration order to keep the
PCG64 stream (hence the synthetic cloud) bit-identical, so only the gather+lerp is hoisted out of the loop.

Measured (py3.14.3, n_min=500, d=30, k_smote=5, best-of-15):
  OLD (row loop)            ~48.7 ms
  NEW (draws + vec lerp)    ~31.7 ms   => ~35% faster, BIT-IDENTICAL.

Run: python -m mlframe.feature_engineering._benchmarks.bench_borderline_smote_synthesize_vectorized
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _old(X, n_syn, k_neighbors, seed):
    n_min = X.shape[0]
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


def _new(X, n_syn, k_neighbors, seed):
    n_min = X.shape[0]
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
    assert np.array_equal(a, b), "not bit-identical"
    old = _best(_old, X, 5000, 5, 1)
    new = _best(_new, X, 5000, 5, 1)
    print(f"OLD={old*1000:.2f}ms  NEW={new*1000:.2f}ms  speedup={old/new:.2f}x  bit-identical=True")


if __name__ == "__main__":
    main()
