"""Bench the (M, N, d) broadcast-cube squared-distance in inducing_attention._stage_a_anchor_to_train.

Stage A computes, for each of M anchors, the squared distance to all N train rows:
``sq = ((anchors[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=-1)`` — an (M, N, d) cube
materialised then reduced over d. The GEMM decomposition ``||a||^2 - 2 a.x + ||x||^2`` allocates
only the (M, N) result. N is the FULL train-fold row count (thousands), so the cube is large.

Run: python bench_inducing_attention_stage_a_gemm.py
"""
import time

import numpy as np


def _old(anchors, X):
    diffs = anchors[:, None, :] - X[None, :, :]
    return (diffs ** 2).sum(axis=-1)


def _new(anchors, X):
    a_sq = np.einsum("ij,ij->i", anchors, anchors)[:, None]
    x_sq = np.einsum("ij,ij->i", X, X)[None, :]
    d = a_sq - 2.0 * (anchors @ X.T) + x_sq
    np.maximum(d, 0.0, out=d)
    return d


def _bestof(fn, anchors, X, n=7):
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(anchors, X)
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for (M, N, d) in [(16, 5000, 30), (16, 20000, 30), (16, 50000, 50), (32, 20000, 100)]:
        anchors = rng.standard_normal((M, d)).astype(np.float32)
        X = rng.standard_normal((N, d)).astype(np.float32)
        _old(anchors, X); _new(anchors, X)  # warm
        to = _bestof(_old, anchors, X)
        tn = _bestof(_new, anchors, X)
        ref = _old(anchors, X)
        got = _new(anchors, X)
        max_abs = float(np.max(np.abs(ref - got)))
        rel = max_abs / (float(np.max(np.abs(ref))) + 1e-12)
        print(f"M={M} N={N} d={d}: old={to*1e3:.2f}ms new={tn*1e3:.2f}ms speedup={to/tn:.2f}x max_abs={max_abs:.3e} rel={rel:.2e}")
