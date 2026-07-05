"""Bench: single-thread polynomial-eval register recurrence vs. legacy per-degree-allocation form.

The single-thread njit evaluators (_legval_njit / _chebval_njit / _lagval_njit / _hermeval_njit)
in hermite_fe/__init__.py are the n<50k dispatch path -- i.e. the dominant CMA-ES inner-search
path for the default max_degree<=4. The LEGACY form allocates a fresh ``np.empty(n)`` per degree
step (k=2..nc-1) plus an initial ``x.copy()``, and makes multiple passes over the array.

The NEW form mirrors the already-shipped *_parallel variants: a single pass over i with the
Horner-style recurrence carried in scalar registers (p_prev/p_curr), no per-degree heap
allocation. Bit-identical by construction (same recurrence, same per-element op order).

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters.hermite_fe._benchmarks.bench_polyeval_single_thread_register
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit


# -------- LEGACY (current source) single-thread forms --------
@njit(cache=True, fastmath=True)
def _legval_old(x, c):
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x.copy()
    for i in range(n):
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        inv_k = 1.0 / k
        two_km1 = 2 * k - 1
        km1 = k - 1
        for i in range(n):
            p_next[i] = (two_km1 * x[i] * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


# -------- NEW register single-pass form --------
@njit(cache=True, fastmath=True)
def _legval_new(x, c):
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in range(n):
            out[i] = c0
        return out
    for i in range(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = (two_km1 * xi * p_curr - km1 * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


# -------- FUSED-PROLOGUE form: keep per-degree array loops (SIMD-friendly) but fuse the
# out[i]=c[0] / out[i]+=c[1]*p_curr / x.copy() prologue passes into one. --------
@njit(cache=True, fastmath=True)
def _legval_fused(x, c):
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    c0 = c[0]
    if nc == 1:
        for i in range(n):
            out[i] = c0
        return out
    c1 = c[1]
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x  # P_1 == x; no copy needed, x is not mutated below
    for i in range(n):
        out[i] = c0 + c1 * x[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        inv_k = 1.0 / k
        two_km1 = 2 * k - 1
        km1 = k - 1
        for i in range(n):
            p_next[i] = (two_km1 * x[i] * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


def _best_of(fn, x, c, reps=200):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn(x, c)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(0)
    c = rng.standard_normal(5)  # degree 4 -> nc=5 (default max_degree)
    print(f"{'n':>8} {'old_ms':>10} {'new_ms':>10} {'speedup':>8}  identical")
    for n in (500, 2000, 10000, 50000):
        x = rng.standard_normal(n)
        # warm
        a = _legval_old(x, c); b = _legval_new(x, c); f = _legval_fused(x, c)
        ident = np.array_equal(a, b)
        ident_f = np.array_equal(a, f)
        reps = 2000 if n <= 2000 else (500 if n <= 10000 else 100)
        old = _best_of(_legval_old, x, c, reps)
        new = _best_of(_legval_new, x, c, reps)
        fus = _best_of(_legval_fused, x, c, reps)
        print(f"{n:>8} old={old*1e3:>9.4f} reg={new*1e3:>9.4f}({old/new:>5.2f}x) " f"fused={fus*1e3:>9.4f}({old/fus:>5.2f}x) reg_id={ident} fused_id={ident_f}")


if __name__ == "__main__":
    main()
