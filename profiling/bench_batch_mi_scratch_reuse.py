"""Q5 PROTOTYPE: reuse joint_counts scratch in the CPU batch_mi noise gate.

_relevance_from_dense allocates np.zeros((K_x, K_y)) on EVERY (perm, col) call
-> K*(nperm+1) allocations. Prototype a thread-local scratch (num_threads, max_Kx*K_y)
indexed by numba.get_thread_id(), zeroed per use, vs the current per-call alloc.
Bit-identity: same counts, same accumulation order; only the buffer's lifetime changes.
"""
import math, time
import numpy as np
import numba
from numba import njit, prange, get_thread_id, get_num_threads

N_BINS = 10
K_Y = 2


@njit(nogil=True, cache=False)
def _rel(use_su, classes_dense, k, freqs_dense, K_x, classes_y, freqs_y, dtype):
    n = classes_dense.shape[0]; K_y = len(freqs_y)
    joint_counts = np.zeros((K_x, K_y), dtype=dtype)
    for r in range(n):
        joint_counts[classes_dense[r, k], classes_y[r]] += 1
    inv_n = 1.0 / n
    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_dense[k, i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * freqs_y[j]))
    return mi_xy


@njit(nogil=True, cache=False)
def _rel_scratch(use_su, classes_dense, k, freqs_dense, K_x, classes_y, freqs_y, dtype, scratch_row, K_y):
    n = classes_dense.shape[0]
    for t in range(K_x * K_y):
        scratch_row[t] = 0
    for r in range(n):
        scratch_row[classes_dense[r, k] * K_y + classes_y[r]] += 1
    inv_n = 1.0 / n
    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_dense[k, i]
        base = i * K_y
        for j in range(K_y):
            jc = scratch_row[base + j]
            if jc != 0:
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * freqs_y[j]))
    return mi_xy


@njit(parallel=True, nogil=True, cache=False)
def gate_alloc(classes_dense, freqs_dense, kx, original_mi, classes_y_safe, freqs_y, npermutations, K, dtype):
    nfailed = np.zeros(K, dtype=np.int64)
    ny = classes_y_safe.shape[0]
    for i in range(npermutations):
        state = np.uint64(0) * np.uint64(2654435761) + np.uint64(i + 1)
        local = classes_y_safe.copy()
        for j in range(ny - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            kk = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]; local[j] = local[kk]; local[kk] = tmp
        for k in prange(K):
            if original_mi[k] <= 0.0:
                continue
            mp = _rel(False, classes_dense, k, freqs_dense, int(kx[k]), local, freqs_y, dtype)
            if mp >= original_mi[k]:
                nfailed[k] += 1
    return nfailed


@njit(parallel=True, nogil=True, cache=False)
def gate_scratch(classes_dense, freqs_dense, kx, original_mi, classes_y_safe, freqs_y, npermutations, K, dtype, max_Kx, K_y):
    nfailed = np.zeros(K, dtype=np.int64)
    ny = classes_y_safe.shape[0]
    nthreads = get_num_threads()
    scratch = np.zeros((nthreads, max_Kx * K_y), dtype=dtype)
    for i in range(npermutations):
        state = np.uint64(0) * np.uint64(2654435761) + np.uint64(i + 1)
        local = classes_y_safe.copy()
        for j in range(ny - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            kk = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]; local[j] = local[kk]; local[kk] = tmp
        for k in prange(K):
            if original_mi[k] <= 0.0:
                continue
            tid = get_thread_id()
            mp = _rel_scratch(False, classes_dense, k, freqs_dense, int(kx[k]), local, freqs_y, dtype, scratch[tid], K_y)
            if mp >= original_mi[k]:
                nfailed[k] += 1
    return nfailed


def _time(fn, *a, repeats=5):
    fn(*a); best = 1e30
    for _ in range(repeats):
        t = time.perf_counter(); fn(*a); best = min(best, time.perf_counter() - t)
    return best


def run(n, K, nperm):
    rng = np.random.default_rng(0)
    classes_dense = (rng.integers(0, N_BINS, size=(n, K))).astype(np.int8)
    kx = np.full(K, N_BINS, dtype=np.int64)
    freqs_dense = np.zeros((K, N_BINS), dtype=np.float64)
    for k in range(K):
        c = np.bincount(classes_dense[:, k], minlength=N_BINS)
        freqs_dense[k] = c / n
    original_mi = rng.random(K) * 0.5 + 0.01  # all > 0 (worst case: no skips)
    classes_y_safe = rng.integers(0, K_Y, size=n).astype(np.int32)
    freqs_y = np.bincount(classes_y_safe, minlength=K_Y).astype(np.float64) / n
    dtype = np.int32

    na = gate_alloc(classes_dense, freqs_dense, kx, original_mi, classes_y_safe, freqs_y, nperm, K, dtype)
    ns = gate_scratch(classes_dense, freqs_dense, kx, original_mi, classes_y_safe, freqs_y, nperm, K, dtype, N_BINS, K_Y)
    ok = np.array_equal(na, ns)

    ta = _time(gate_alloc, classes_dense, freqs_dense, kx, original_mi, classes_y_safe, freqs_y, nperm, K, dtype)
    ts = _time(gate_scratch, classes_dense, freqs_dense, kx, original_mi, classes_y_safe, freqs_y, nperm, K, dtype, N_BINS, K_Y)
    print(f"n={n} K={K} nperm={nperm}: alloc {ta*1e3:8.2f}ms  scratch {ts*1e3:8.2f}ms  ({ta/ts:.2f}x)  nfailed_identical={ok}")


if __name__ == "__main__":
    for K in (300, 1000, 4000):
        run(2407, K, 10)
    run(2407, 1000, 30)
