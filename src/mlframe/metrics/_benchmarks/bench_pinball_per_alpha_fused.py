"""Bench: pinball_loss_per_alpha -- K separate JIT calls + K strided column
copies (P[:, j] is cache-unfriendly on a C-contiguous (N,K) matrix) vs a single
fused row-major kernel that scores all K alphas in one pass.

Run: python -m mlframe.metrics._benchmarks.bench_pinball_per_alpha_fused
"""
from __future__ import annotations

import time

import numpy as np
import numba

_NJIT_KW = dict(fastmath=False, cache=True, nogil=True)


@numba.njit(**_NJIT_KW)
def _fast_pinball(y, q, alpha):
    n = y.shape[0]
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        e = y[i] - q[i]
        if e > 0:
            s += alpha * e
        else:
            s += (alpha - 1.0) * e
    return s / n


def _per_alpha_old(y, P, alphas):
    return {float(a): float(_fast_pinball(y, np.ascontiguousarray(P[:, j]), float(a))) for j, a in enumerate(alphas)}


@numba.njit(**_NJIT_KW)
def _fast_pinball_per_alpha(y, P, alphas):
    """All K alphas in one row-major pass over C-contiguous (N,K) ``P``."""
    n = P.shape[0]
    k = P.shape[1]
    out = np.zeros(k, dtype=np.float64)
    if n == 0:
        return out
    for i in range(n):
        yi = y[i]
        for j in range(k):
            e = yi - P[i, j]
            a = alphas[j]
            if e > 0:
                out[j] += a * e
            else:
                out[j] += (a - 1.0) * e
    for j in range(k):
        out[j] /= n
    return out


def _per_alpha_new(y, P, alphas_arr):
    res = _fast_pinball_per_alpha(y, P, alphas_arr)
    return {float(alphas_arr[j]): float(res[j]) for j in range(len(alphas_arr))}


def _best_of(fn, *args, reps=20):
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(0)
    for n in (5_000, 50_000, 200_000):
        for k in (3, 9, 19):
            y = np.ascontiguousarray(rng.standard_normal(n))
            P = np.ascontiguousarray(rng.standard_normal((n, k)))
            alphas = np.linspace(0.05, 0.95, k)
            # warm both
            old = _per_alpha_old(y, P, alphas)
            new = _per_alpha_new(y, P, alphas)
            # identity
            maxdiff = max(abs(old[float(a)] - new[float(a)]) for a in alphas)
            t_old = _best_of(_per_alpha_old, y, P, alphas)
            t_new = _best_of(_per_alpha_new, y, P, alphas)
            print(f"n={n:>7} k={k:>2}  old={t_old*1e3:8.3f}ms  new={t_new*1e3:8.3f}ms  " f"speedup={t_old/t_new:5.2f}x  maxdiff={maxdiff:.2e}")


if __name__ == "__main__":
    main()
