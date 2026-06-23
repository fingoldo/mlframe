"""Bench: class_mahalanobis._mahalanobis quadratic form.

OLD: np.einsum("ij,jk,ik->i", diff, inv_cov, diff)  -- no optimize, naive O(n*d^2)
NEW: ((diff @ inv_cov) * diff).sum(axis=1)           -- BLAS GEMM + elementwise

Identity: ~1e-9 FP reduction-order delta (different summation order); selection-safe.
Run: CUDA_VISIBLE_DEVICES="" python bench_mahalanobis_quadform.py
"""
from __future__ import annotations
import time
import numpy as np


def old_mahalanobis(X, mean, inv_cov):
    diff = X - mean
    return np.einsum("ij,jk,ik->i", diff, inv_cov, diff).astype(np.float32)


def new_mahalanobis(X, mean, inv_cov):
    diff = X - mean
    return (np.matmul(diff, inv_cov) * diff).sum(axis=1).astype(np.float32)


def _make(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    mean = rng.standard_normal(d).astype(np.float32)
    A = rng.standard_normal((d, d)).astype(np.float32)
    inv_cov = (A @ A.T + np.eye(d)).astype(np.float32)  # SPD, symmetric
    return X, mean, inv_cov


def bench(fn, X, mean, inv_cov, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(X, mean, inv_cov)
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    print(f"{'n':>8} {'d':>4} {'old_ms':>10} {'new_ms':>10} {'speedup':>8} {'max_abs_diff':>14} {'max_rel':>10}")
    for n in (4000, 20000, 100000):
        for d in (6, 15, 30):
            X, mean, inv_cov = _make(n, d)
            # warm
            old_mahalanobis(X, mean, inv_cov); new_mahalanobis(X, mean, inv_cov)
            reps = 30 if n <= 20000 else 15
            t_old = bench(old_mahalanobis, X, mean, inv_cov, reps)
            t_new = bench(new_mahalanobis, X, mean, inv_cov, reps)
            ro = old_mahalanobis(X, mean, inv_cov)
            rn = new_mahalanobis(X, mean, inv_cov)
            mad = float(np.max(np.abs(ro - rn)))
            mrel = float(np.max(np.abs(ro - rn) / (np.abs(ro) + 1e-12)))
            print(f"{n:>8} {d:>4} {t_old*1e3:>10.3f} {t_new*1e3:>10.3f} {t_old/t_new:>8.2f} {mad:>14.3e} {mrel:>10.3e}")
