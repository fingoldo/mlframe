"""A/B: hoist the per-shuffle ``joint_counts`` allocation out of the K-loop in the
maxT pair-MI kernel (``batch_pair_mi_perm_batched``).

Current: allocates a (joint_card x k_y) int64 histogram K times PER PAIR (K*n_pairs
allocations). Variant: allocate once per pair, zero-and-reuse across the K shuffles.
Measures whether removing the allocator churn eases the memory-bound plateau.

Run: python -m mlframe.feature_selection._benchmarks.bench_pair_maxt_kernel_hoist
"""

import math
import time

import numpy as np
from numba import njit, prange

from mlframe.feature_selection._benchmarks.bench_pair_maxt_floor_subsample import (
    all_pairs,
    make_screen_data,
)
from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    MAX_JOINT_CARDINALITY,
    batch_pair_mi_perm_batched as K_CUR,
)


@njit(parallel=True, nogil=True, cache=True)
def K_HOIST(factors_data, pair_a, pair_b, nbins, y_perms, freqs_y):
    n = factors_data.shape[0]
    n_pairs = pair_a.shape[0]
    K = y_perms.shape[0]
    n_classes_y = freqs_y.shape[0]
    out = np.empty((K, n_pairs), dtype=np.float64)
    if n == 0:
        out[:, :] = 0.0
        return out
    inv_n = 1.0 / n
    for p in prange(n_pairs):
        a = pair_a[p]
        b = pair_b[p]
        nb_a = int(nbins[a])
        nb_b = int(nbins[b])
        if nb_a <= 0 or nb_b <= 0 or nb_a > MAX_JOINT_CARDINALITY // nb_b:
            for k in range(K):
                out[k, p] = 0.0
            continue
        joint_card = nb_a * nb_b
        cls_x = np.empty(n, dtype=np.int64)
        freqs_x_int = np.zeros(joint_card, dtype=np.int64)
        for i in range(n):
            c = int(factors_data[i, a]) * nb_b + int(factors_data[i, b])
            cls_x[i] = c
            freqs_x_int[c] += 1
        joint_counts = np.empty((joint_card, n_classes_y), dtype=np.int64)  # hoisted: one alloc per pair
        for k in range(K):
            joint_counts[:, :] = 0
            for i in range(n):
                joint_counts[cls_x[i], int(y_perms[k, i])] += 1
            total = 0.0
            for ci in range(joint_card):
                fx = freqs_x_int[ci]
                if fx == 0:
                    continue
                prob_x = fx * inv_n
                for cj in range(n_classes_y):
                    jc = joint_counts[ci, cj]
                    if jc == 0:
                        continue
                    jf = jc * inv_n
                    prob_y = freqs_y[cj]
                    if prob_y > 0.0:
                        total += jf * math.log(jf / (prob_x * prob_y))
            out[k, p] = total
    return out


def main():
    data, nbins, cy, fy = make_screen_data(30000, 60, 10, 3, 12345)
    pa, pb = all_pairs(60)
    rng = np.random.default_rng(0)
    yp = np.empty((25, 30000), dtype=np.int64)
    t = cy.copy()
    for k in range(25):
        rng.shuffle(t)
        yp[k] = t
    # warm both
    _ = K_CUR(data[:64], pa, pb, nbins, yp[:2, :64], fy)
    _ = K_HOIST(data[:64], pa, pb, nbins, yp[:2, :64], fy)
    # identity
    r_cur = K_CUR(data, pa, pb, nbins, yp, fy)
    r_h = K_HOIST(data, pa, pb, nbins, yp, fy)
    print("max|diff| =", float(np.max(np.abs(r_cur - r_h))))

    def best(fn, reps=6):
        return min(( (lambda: (time.perf_counter(), fn(), time.perf_counter()))() and 0 for _ in range(0))  or
                   [ (lambda s=time.perf_counter(): (fn(), time.perf_counter() - s)[1])() for _ in range(reps)])

    # interleaved paired A/B
    tc, th = [], []
    for _ in range(8):
        s = time.perf_counter(); K_CUR(data, pa, pb, nbins, yp, fy); tc.append(time.perf_counter() - s)
        s = time.perf_counter(); K_HOIST(data, pa, pb, nbins, yp, fy); th.append(time.perf_counter() - s)
    wins = sum(h < c for c, h in zip(tc, th))
    print(f"CUR   min={min(tc)*1000:.1f}ms median={sorted(tc)[len(tc)//2]*1000:.1f}ms")
    print(f"HOIST min={min(th)*1000:.1f}ms median={sorted(th)[len(th)//2]*1000:.1f}ms")
    print(f"HOIST faster in {wins}/8 trials  min-speedup={min(tc)/min(th):.3f}x")


if __name__ == "__main__":
    main()
