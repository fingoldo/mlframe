"""Bench: single-thread polynomial-eval ping-pong scratch reuse vs. legacy per-degree-allocation.

The shipped single-thread njit evaluators (_legval_njit / _chebval_njit / _lagval_njit / _hermeval_njit)
in hermite_fe/__init__.py are the n<50k dispatch path -- the dominant CMA-ES inner-search path for the
default max_degree<=4. The shipped (fused-prologue) form STILL allocates a fresh ``np.empty(n)`` per
degree step (k=2..nc-1): for nc=5 that's 3 heap allocs per call, and the kernel is called thousands of
times across a CMA-ES search.

The register single-pass form (bench_polyeval_single_thread_register) was REJECTED single-thread because
the scalar p_prev/p_curr recurrence blocks SIMD vectorization of the inner i-loop.

NEW idea (fresh): keep the SIMD-friendly per-degree ARRAY loop, but PING-PONG between two pre-allocated
scratch buffers instead of allocating ``np.empty(n)`` every degree. Bit-identical by construction (same
recurrence, same per-element op order, same array-loop SIMD shape) -- only the allocation is removed.

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters.hermite_fe._benchmarks.bench_polyeval_pingpong_scratch

bench-attempt-rejected (2026-06-24): ping-pong scratch reuse is 0.47-0.64x (SLOWER) than the
shipped per-degree ``np.empty(n)`` form across n=500..50k, nc=5 and nc=9 -- bit-identical but a
regression. Reason: reusing two buffers (p_prev / p_curr / p_next overlap across iterations) defeats
LLVM no-alias analysis, so the inner i-loop loses SIMD vectorization. A fresh ``np.empty`` per degree
proves distinctness to the compiler and stays vectorized. SAME vectorization-blocking failure class as
the rejected register single-pass form. Do not re-try buffer reuse for the single-thread polyeval path.
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit


# -------- SHIPPED (current source) fused-prologue form, per-degree np.empty --------
@njit(cache=True, fastmath=True)
def _legval_shipped(x, c):
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
    p_curr = x
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


# -------- NEW ping-pong scratch form: two pre-allocated buffers, no per-degree alloc --------
@njit(cache=True, fastmath=True)
def _legval_pingpong(x, c):
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
    # Two scratch buffers. buf_prev starts as P_0 (ones), buf_curr as P_1 (x).
    buf_a = np.ones(n, dtype=np.float64)      # P_{k-2}
    buf_b = np.empty(n, dtype=np.float64)     # P_{k-1}
    for i in range(n):
        xi = x[i]
        buf_b[i] = xi
        out[i] = c0 + c1 * xi
    # ping-pong: at each step write into the buffer holding P_{k-2} (no longer needed).
    p_prev = buf_a
    p_curr = buf_b
    for k in range(2, nc):
        p_next = p_prev  # reuse the now-dead P_{k-2} buffer
        ck = c[k]
        inv_k = 1.0 / k
        two_km1 = 2 * k - 1
        km1 = k - 1
        for i in range(n):
            p_next[i] = (two_km1 * x[i] * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        # rotate: P_{k-1} -> prev, P_k (just written) -> curr
        p_prev = p_curr
        p_curr = p_next
    return out


def _best_of(fn, x, c, reps):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn(x, c)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(0)
    print("degree-4 (nc=5):")
    for nc in (5, 9):
        c = rng.standard_normal(nc)
        print(f"\n--- nc={nc} (max_degree={nc-1}) ---")
        print(f"{'n':>8} {'shipped_ms':>12} {'pingpong_ms':>12} {'speedup':>8}  identical")
        for n in (500, 2000, 10000, 50000):
            x = rng.standard_normal(n)
            a = _legval_shipped(x, c)
            b = _legval_pingpong(x, c)
            ident = np.array_equal(a, b)
            reps = 4000 if n <= 2000 else (1000 if n <= 10000 else 200)
            # interleaved paired best-of to cancel machine noise
            s = _best_of(_legval_shipped, x, c, reps)
            p = _best_of(_legval_pingpong, x, c, reps)
            print(f"{n:>8} {s*1e3:>12.5f} {p*1e3:>12.5f} {s/p:>7.2f}x  {ident}")


if __name__ == "__main__":
    main()
