"""Shared distance/softmax/timing helpers for the anchor-attention GEMM benchmark family
(bench_anchor_attention_softmax_gemm.py, bench_class_conditional_anchor_softmax_gemm.py):
independently duplicated across those scripts, consolidated here so a fix can't silently drift
out of sync across copies.
"""
from __future__ import annotations

import time

import numpy as np


def dists_broadcast(X, anchors):
    """Squared Euclidean distances via broadcast subtraction (the naive O(n*k*d) reference path)."""
    return np.sum((X[:, None, :] - anchors[None, :, :]) ** 2, axis=2)


def dists_gemm(X, anchors):
    """Squared Euclidean distances via the ||x||^2 - 2*x.a + ||a||^2 GEMM identity, bit-identical (clamped >= 0)."""
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    a_sq = np.einsum("ij,ij->i", anchors, anchors)[None, :]
    d = x_sq - 2.0 * (X @ anchors.T) + a_sq
    np.maximum(d, 0.0, out=d)
    return d


def softmax_from_dists(dists, temp):
    """Row-softmax of ``-dists / temp``, numerically stabilized (max-subtraction), cast to float32."""
    logits = -dists / (temp + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def best_of(fn, *args, n: int = 30) -> float:
    """Run ``fn(*args)`` ``n`` times and return the best (minimum) wall-clock time."""
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def best_of_args(fn, args: tuple, n: int = 7) -> float:
    """Run ``fn(*args)`` ``n`` times and return the best (minimum) wall-clock time (args passed as an explicit
    tuple rather than varargs, matching this call convention's existing call sites)."""
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best
