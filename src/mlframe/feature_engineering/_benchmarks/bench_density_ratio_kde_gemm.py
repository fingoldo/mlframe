"""Bench: _gaussian_kde_log broadcast (Xq[:,None,:]-Xt[None,:,:])**2 vs gemm-expansion ||x-y||^2 = |x|^2+|y|^2-2 x.y^T.

The broadcast path materialises a (chunk, n_t, d) float32 temporary per chunk and reduces it; the gemm path
computes the same squared-distance matrix (chunk, n_t) via a single BLAS sgemm plus two squared-norm vectors,
never allocating the 3-D tensor. Mathematically identical up to fp reduction order; the LSE downstream is unchanged.

Run: CUDA_VISIBLE_DEVICES="" D:/ProgramData/anaconda3/python.exe bench_density_ratio_kde_gemm.py
"""
from __future__ import annotations

import time
import numpy as np


def _kde_broadcast(X_query, X_train_subset, h, chunk=1000):
    n_q = X_query.shape[0]
    n_t = X_train_subset.shape[0]
    if n_t < 1:
        return np.full(n_q, -30.0, dtype=np.float32)
    out = np.zeros(n_q, dtype=np.float32)
    h_sq = max(h * h, 1e-9)
    log_N = np.log(n_t)
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        Xq = X_query[start:end]
        d2 = ((Xq[:, None, :] - X_train_subset[None, :, :]) ** 2).sum(axis=2)
        logits = -d2 / (2.0 * h_sq)
        m = logits.max(axis=1, keepdims=True)
        lse = m.ravel() + np.log(np.exp(logits - m).sum(axis=1) + 1e-30)
        out[start:end] = (lse - log_N).astype(np.float32)
    return out


def _kde_gemm(X_query, X_train_subset, h, chunk=1000):
    n_q = X_query.shape[0]
    n_t = X_train_subset.shape[0]
    if n_t < 1:
        return np.full(n_q, -30.0, dtype=np.float32)
    out = np.zeros(n_q, dtype=np.float32)
    h_sq = max(h * h, 1e-9)
    log_N = np.log(n_t)
    t_sq = np.einsum("ij,ij->i", X_train_subset, X_train_subset)  # |y|^2, (n_t,)
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        Xq = X_query[start:end]
        q_sq = np.einsum("ij,ij->i", Xq, Xq)  # |x|^2, (chunk,)
        d2 = q_sq[:, None] + t_sq[None, :] - 2.0 * (Xq @ X_train_subset.T)
        np.maximum(d2, 0.0, out=d2)  # guard tiny negatives from cancellation
        logits = -d2 / (2.0 * h_sq)
        m = logits.max(axis=1, keepdims=True)
        lse = m.ravel() + np.log(np.exp(logits - m).sum(axis=1) + 1e-30)
        out[start:end] = (lse - log_N).astype(np.float32)
    return out


def bench(n_q, n_t, d, h=1.5, reps=5):
    rng = np.random.default_rng(0)
    Xq = rng.standard_normal((n_q, d)).astype(np.float32)
    Xt = rng.standard_normal((n_t, d)).astype(np.float32)
    a = _kde_broadcast(Xq, Xt, h)
    b = _kde_gemm(Xq, Xt, h)
    max_abs = float(np.max(np.abs(a - b)))
    max_rel = float(np.max(np.abs(a - b) / (np.abs(a) + 1e-6)))

    def best(fn):
        t = []
        for _ in range(reps):
            s = time.perf_counter()
            fn(Xq, Xt, h)
            t.append(time.perf_counter() - s)
        return min(t)

    t_old = best(_kde_broadcast)
    t_new = best(_kde_gemm)
    print(f"n_q={n_q} n_t={n_t} d={d}: OLD={t_old*1e3:.2f}ms NEW={t_new*1e3:.2f}ms "
          f"speedup={t_old/t_new:.2f}x max_abs={max_abs:.2e} max_rel={max_rel:.2e}")


if __name__ == "__main__":
    for nq, nt, dd in [(2000, 500, 20), (3000, 1500, 30), (5000, 2500, 50), (2000, 1000, 100)]:
        bench(nq, nt, dd)
