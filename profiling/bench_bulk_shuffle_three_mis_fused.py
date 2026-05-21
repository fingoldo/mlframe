"""Bench fused shuffle+joint-accumulate in _bulk_shuffle_and_compute_three_mis.

bench-attempt-rejected (2026-05-21, c0109 / iter135): folding the joint
accumulation INTO the Fisher-Yates loop (position i becomes final after
the swap at iter i, so we can accumulate right there) eliminates one
full N-pass over local[] -- but runs 18-31% SLOWER:

    n=  50000  K=8 P= 8: 2pass=   1.53ms  fused=   1.49ms  (1.02x)
    n= 200000  K=8 P= 8: 2pass=   8.31ms  fused=  11.04ms  (0.75x)
    n= 500000  K=8 P= 8: 2pass=  31.57ms  fused=  38.36ms  (0.82x)
    n= 200000  K=16 P=16: 2pass=  16.74ms  fused=  21.24ms  (0.79x)
    n=1000000  K=8 P= 4: 2pass=  37.99ms  fused=  55.18ms  (0.69x)

Cause: the 2-pass version has clean streaming patterns (pass 1: write
local[] in random order, pass 2: read local[] sequentially while
reading classes_pair/x1/x2 sequentially). The fused version touches 6
arrays per iteration (local[i], local[k] random, classes_pair[i],
classes_x1[i], classes_x2[i] sequential, plus 3 random-write joint
matrices). The larger per-iteration working set blows L1; the saved
N-pass is more than paid back in cache misses.

Numerical equivalence verified: max abs diff == 0 (the fusion is
correct, just slower).

Bench preserved per ``feedback_document_failed_optimization_attempts``
so the next agent doesn't re-try this same path.

c0109 attributed 13.66s self-time across 268 calls of
_bulk_shuffle_and_compute_three_mis (~51 ms / call at n=200k); fusion
would have ADDED ~3.5s, not removed it.

Run: ``python profiling/bench_bulk_shuffle_three_mis_fused.py``
"""

import time
import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def _shuffle_then_accumulate(
    classes_pair, freqs_pair, classes_x1, freqs_x1, classes_x2, freqs_x2,
    classes_y, freqs_y, n_perms, base_seed, dtype,
):
    """Current production form: shuffle in one pass, accumulate in second."""
    n = len(classes_y)
    K_pair = len(freqs_pair); K_x1 = len(freqs_x1)
    K_x2 = len(freqs_x2); K_y = len(freqs_y)
    out_i_pair = np.zeros(n_perms, dtype=np.float64)
    out_i_x1 = np.zeros(n_perms, dtype=np.float64)
    out_i_x2 = np.zeros(n_perms, dtype=np.float64)

    for p in prange(n_perms):
        state = np.uint64(base_seed) + np.uint64(p) * np.uint64(2654435761)
        local = classes_y.copy()
        for i in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (i + 1)
            tmp = local[i]; local[i] = local[k]; local[k] = tmp

        joint_pair = np.zeros((K_pair, K_y), dtype=dtype)
        joint_x1 = np.zeros((K_x1, K_y), dtype=dtype)
        joint_x2 = np.zeros((K_x2, K_y), dtype=dtype)
        for k in range(n):
            cy = local[k]
            joint_pair[classes_pair[k], cy] += 1
            joint_x1[classes_x1[k], cy] += 1
            joint_x2[classes_x2[k], cy] += 1

        inv_n = 1.0 / n
        i_pair = 0.0
        for i in range(K_pair):
            px = freqs_pair[i]
            for j in range(K_y):
                jc = joint_pair[i, j]
                if jc:
                    jf = jc * inv_n
                    i_pair += jf * math.log(jf / (px * freqs_y[j]))
        i_x1 = 0.0
        for i in range(K_x1):
            px = freqs_x1[i]
            for j in range(K_y):
                jc = joint_x1[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x1 += jf * math.log(jf / (px * freqs_y[j]))
        i_x2 = 0.0
        for i in range(K_x2):
            px = freqs_x2[i]
            for j in range(K_y):
                jc = joint_x2[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x2 += jf * math.log(jf / (px * freqs_y[j]))
        out_i_pair[p] = i_pair
        out_i_x1[p] = i_x1
        out_i_x2[p] = i_x2
    return out_i_pair, out_i_x1, out_i_x2


@njit(parallel=True, nogil=True, cache=True)
def _shuffle_accumulate_fused(
    classes_pair, freqs_pair, classes_x1, freqs_x1, classes_x2, freqs_x2,
    classes_y, freqs_y, n_perms, base_seed, dtype,
):
    """Fused: at each shuffle step i, position i becomes final -- accumulate
    into joint matrices immediately. Saves one full N-pass over local."""
    n = len(classes_y)
    K_pair = len(freqs_pair); K_x1 = len(freqs_x1)
    K_x2 = len(freqs_x2); K_y = len(freqs_y)
    out_i_pair = np.zeros(n_perms, dtype=np.float64)
    out_i_x1 = np.zeros(n_perms, dtype=np.float64)
    out_i_x2 = np.zeros(n_perms, dtype=np.float64)

    for p in prange(n_perms):
        state = np.uint64(base_seed) + np.uint64(p) * np.uint64(2654435761)
        local = classes_y.copy()

        joint_pair = np.zeros((K_pair, K_y), dtype=dtype)
        joint_x1 = np.zeros((K_x1, K_y), dtype=dtype)
        joint_x2 = np.zeros((K_x2, K_y), dtype=dtype)

        for i in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (i + 1)
            tmp = local[i]; local[i] = local[k]; local[k] = tmp
            # local[i] is now FINAL (future iters touch only indices < i).
            cy = local[i]
            joint_pair[classes_pair[i], cy] += 1
            joint_x1[classes_x1[i], cy] += 1
            joint_x2[classes_x2[i], cy] += 1
        # Index 0 is never the loop target; handle here.
        cy0 = local[0]
        joint_pair[classes_pair[0], cy0] += 1
        joint_x1[classes_x1[0], cy0] += 1
        joint_x2[classes_x2[0], cy0] += 1

        inv_n = 1.0 / n
        i_pair = 0.0
        for i in range(K_pair):
            px = freqs_pair[i]
            for j in range(K_y):
                jc = joint_pair[i, j]
                if jc:
                    jf = jc * inv_n
                    i_pair += jf * math.log(jf / (px * freqs_y[j]))
        i_x1 = 0.0
        for i in range(K_x1):
            px = freqs_x1[i]
            for j in range(K_y):
                jc = joint_x1[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x1 += jf * math.log(jf / (px * freqs_y[j]))
        i_x2 = 0.0
        for i in range(K_x2):
            px = freqs_x2[i]
            for j in range(K_y):
                jc = joint_x2[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x2 += jf * math.log(jf / (px * freqs_y[j]))
        out_i_pair[p] = i_pair
        out_i_x1[p] = i_x1
        out_i_x2[p] = i_x2
    return out_i_pair, out_i_x1, out_i_x2


def bench(label, fn, args, n_iter=20):
    fn(*args); fn(*args)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(*args)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3, label


def make_args(n, K_pair, K_x1, K_x2, K_y, n_perms, seed=0):
    rng = np.random.default_rng(seed)
    classes_pair = rng.integers(0, K_pair, size=n).astype(np.int64)
    classes_x1 = rng.integers(0, K_x1, size=n).astype(np.int64)
    classes_x2 = rng.integers(0, K_x2, size=n).astype(np.int64)
    classes_y = rng.integers(0, K_y, size=n).astype(np.int64)
    freqs_pair = np.bincount(classes_pair, minlength=K_pair) / n
    freqs_x1 = np.bincount(classes_x1, minlength=K_x1) / n
    freqs_x2 = np.bincount(classes_x2, minlength=K_x2) / n
    freqs_y = np.bincount(classes_y, minlength=K_y) / n
    return (classes_pair, freqs_pair, classes_x1, freqs_x1, classes_x2, freqs_x2,
            classes_y, freqs_y, n_perms, np.uint64(12345), np.int64)


if __name__ == "__main__":
    for n, K, P in [(50_000, 8, 8), (200_000, 8, 8), (500_000, 8, 8),
                    (200_000, 16, 16), (1_000_000, 8, 4)]:
        args = make_args(n, K, K, K, 3, P)
        t_a, _ = bench("2-pass", _shuffle_then_accumulate, args)
        t_b, _ = bench("fused",  _shuffle_accumulate_fused, args)

        # Verify numerical equivalence (joint+MI values are exact; only RNG differs trivially)
        out_a = _shuffle_then_accumulate(*args)
        out_b = _shuffle_accumulate_fused(*args)
        diff = max(abs(a - b).max() for a, b in zip(out_a, out_b))

        speedup = t_a / t_b
        print(f"n={n:>7}  K={K} P={P:>2}: 2pass={t_a:7.2f}ms  fused={t_b:7.2f}ms  ({speedup:.2f}x)  max_diff={diff:.2e}")
