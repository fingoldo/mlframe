"""Bench: _joint_chao_shen_mi_njit -- avoid the redundant N/f_1 re-walk of the joint table.

``_joint_chao_shen_mi_njit`` (filters/_chao_shen.py) computes the shared Good-Turing coverage by
walking the flattened joint table to accumulate N and f_1, then calls
``chao_shen_entropy_from_counts(flat, C_hat)`` for H_xy -- which walks ``flat`` AGAIN to recompute
N and f_1 (both DISCARDED, since coverage is supplied). For a (K_x, K_y) joint the flat array has
K_x*K_y entries, so H_xy's redundant prologue pass is the single largest re-walk in the function.

NEW: pass the already-known N (and the local-fallback flag) into a pruned entropy-from-counts
variant for the H_xy term so the joint table is walked ONCE for (N, f_1) and ONCE for the entropy
sum, instead of twice. Bit-identical by construction: same C_hat, same per-cell entropy formula,
same N. row_sums / col_sums terms (length K_x, K_y -- tiny) keep the original kernel.

Run:
  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_chao_shen_joint_coverage_hoist

bench-attempt-rejected (2026-06-24): hoisting the known N into a pruned H_xy entropy kernel is
0.90-1.00x (NO win, slight regression) across joint shapes (10,2)..(100,100) at n=1500..10000,
bit-identical on non-all-singleton joints. Reason: the redundant N/f_1 re-walk is a single integer
pass over the flat table -- negligible against the per-cell ``log(p_adj)`` + ``(1-p_adj)**N`` pow that
dominates the entropy sum (both UNCHANGED). The extra C_hat<0 branch + lost inlining of the pruned
variant slightly outweighs the saved pass. The kernel is at its transcendental floor; the only real
lever would be approximating pow/log, which would change MI (selection-altering). Do not re-try the
coverage/N hoist. NOTE: this pruned variant would also DIVERGE on an all-singleton joint (C_hat=-1.0):
OLD recomputes a per-table local C_hat for row/col sums, NEW forces local plug-in for all three terms.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit


# ---------- OLD (current source) ----------
@njit(nogil=True, cache=True)
def _cs_entropy_old(counts, coverage=-1.0):
    N = 0
    f_1 = 0
    K = counts.shape[0]
    for k in range(K):
        c = counts[k]
        if c <= 0:
            continue
        N += int(c)
        if c == 1:
            f_1 += 1
    if N <= 0:
        return 0.0
    if coverage >= 0.0:
        C_hat = coverage
    else:
        C_hat = 1.0 - float(f_1) / float(N)
    if C_hat <= 1e-12:
        N_f = float(N)
        h = 0.0
        for k in range(K):
            c = counts[k]
            if c > 0:
                p = float(c) / N_f
                h -= p * math.log(p)
        return max(0.0, h)
    N_f = float(N)
    h_cs = 0.0
    for k in range(K):
        c = counts[k]
        if c <= 0:
            continue
        p_emp = float(c) / N_f
        p_adj = C_hat * p_emp
        if p_adj <= 0.0:
            continue
        denom = 1.0 - (1.0 - p_adj) ** N
        if denom <= 1e-12:
            continue
        h_cs -= p_adj * math.log(p_adj) / denom
    return max(0.0, h_cs)


@njit(nogil=True, cache=True)
def _joint_old(joint):
    K_x, K_y = joint.shape
    if K_x < 1 or K_y < 1:
        return 0.0
    row_sums = np.zeros(K_x, dtype=np.int64)
    col_sums = np.zeros(K_y, dtype=np.int64)
    for i in range(K_x):
        for j in range(K_y):
            v = int(joint[i, j])
            if v > 0:
                row_sums[i] += v
                col_sums[j] += v
    flat = joint.ravel().astype(np.int64)
    N = 0
    f_1 = 0
    for t in range(flat.shape[0]):
        c = int(flat[t])
        if c > 0:
            N += c
            if c == 1:
                f_1 += 1
    if N <= 0:
        return 0.0
    C_hat = 1.0 - float(f_1) / float(N)
    if C_hat <= 1e-12:
        C_hat = -1.0
    H_xy = _cs_entropy_old(flat, C_hat)
    H_x = _cs_entropy_old(row_sums, C_hat)
    H_y = _cs_entropy_old(col_sums, C_hat)
    return max(0.0, H_x + H_y - H_xy)


# ---------- NEW: pruned H_xy entropy that takes (N, use_local_fallback) ----------
@njit(nogil=True, cache=True)
def _cs_entropy_known_N(counts, N, C_hat):
    """Entropy-from-counts with N and coverage already known (no N/f_1 re-walk).
    C_hat < 0 => local plug-in fallback (the all-singleton joint case)."""
    if N <= 0:
        return 0.0
    N_f = float(N)
    if C_hat < 0.0 or C_hat <= 1e-12:
        h = 0.0
        K = counts.shape[0]
        for k in range(K):
            c = counts[k]
            if c > 0:
                p = float(c) / N_f
                h -= p * math.log(p)
        return max(0.0, h)
    h_cs = 0.0
    K = counts.shape[0]
    for k in range(K):
        c = counts[k]
        if c <= 0:
            continue
        p_emp = float(c) / N_f
        p_adj = C_hat * p_emp
        if p_adj <= 0.0:
            continue
        denom = 1.0 - (1.0 - p_adj) ** N
        if denom <= 1e-12:
            continue
        h_cs -= p_adj * math.log(p_adj) / denom
    return max(0.0, h_cs)


@njit(nogil=True, cache=True)
def _joint_new(joint):
    K_x, K_y = joint.shape
    if K_x < 1 or K_y < 1:
        return 0.0
    row_sums = np.zeros(K_x, dtype=np.int64)
    col_sums = np.zeros(K_y, dtype=np.int64)
    for i in range(K_x):
        for j in range(K_y):
            v = int(joint[i, j])
            if v > 0:
                row_sums[i] += v
                col_sums[j] += v
    flat = joint.ravel().astype(np.int64)
    N = 0
    f_1 = 0
    for t in range(flat.shape[0]):
        c = int(flat[t])
        if c > 0:
            N += c
            if c == 1:
                f_1 += 1
    if N <= 0:
        return 0.0
    C_hat = 1.0 - float(f_1) / float(N)
    if C_hat <= 1e-12:
        C_hat = -1.0
    # H_xy: N already known; skip the re-walk. row/col terms keep the std kernel
    # (tiny K, and they need their OWN N which equals this N anyway -> reuse it too).
    H_xy = _cs_entropy_known_N(flat, N, C_hat)
    H_x = _cs_entropy_known_N(row_sums, N, C_hat)
    H_y = _cs_entropy_known_N(col_sums, N, C_hat)
    return max(0.0, H_x + H_y - H_xy)


def _make_joint(rng, n, K_x, K_y):
    x = rng.integers(0, K_x, n)
    y = rng.integers(0, K_y, n)
    j = np.zeros((K_x, K_y), dtype=np.int64)
    for a, b in zip(x, y):
        j[a, b] += 1
    return j


def _best_of(fn, j, reps):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn(j)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(0)
    print(f"{'shape':>14} {'n':>8} {'old_us':>10} {'new_us':>10} {'speedup':>8}  identical")
    # Realistic MRMR sparse-contingency shapes: feature card x target card.
    for K_x, K_y, n in [(10, 2, 1500), (14, 10, 1500), (39, 39, 1500), (50, 50, 5000), (100, 100, 10000)]:
        j = _make_joint(rng, n, K_x, K_y)
        a = _joint_old(j); b = _joint_new(j)
        ident = (a == b)
        reps = 5000
        o = _best_of(_joint_old, j, reps)
        nw = _best_of(_joint_new, j, reps)
        print(f"{str((K_x,K_y)):>14} {n:>8} {o*1e6:>10.3f} {nw*1e6:>10.3f} {o/nw:>7.2f}x  {ident}  (old={a:.6f} new={b:.6f})")


if __name__ == "__main__":
    main()
