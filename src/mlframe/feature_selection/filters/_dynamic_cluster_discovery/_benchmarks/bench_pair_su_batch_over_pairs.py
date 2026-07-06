"""A/B: prange-OVER-PAIRS batched joint-entropy for the DCD ``discover_cluster_members`` hot loop (2026-07).

Distinct from the REJECTED ``bench_pair_su_joint_entropy_prange.py`` (which parallelised the sample loop of a
SINGLE pair -> 0.46x, spawn dominates). The real DCD hot site is ``discover_cluster_members``: for one freshly
selected ANCHOR it computes ``pair_su(c, anchor)`` for EVERY candidate ``c`` in the pool -- K joint histograms
that all share the anchor column. Parallelising OVER THE K PAIRS (each thread runs the existing serial joint
histogram for a different candidate) amortises the one thread-spawn across K pairs and keeps the shared anchor
column hot in cache. This is what a batched ``discover`` path would call.

BIT-IDENTICAL by construction: each pair's joint histogram is the identical single-thread reduction in the same
ascending class-id order as ``joint_entropy_2var``; only the OUTER pair loop is parallel (independent outputs).

Run:  python -m mlframe.feature_selection.filters._dynamic_cluster_discovery._benchmarks.bench_pair_su_batch_over_pairs
"""
from __future__ import annotations

import time

import numpy as np
import numba
from numba import njit, prange

from mlframe.feature_selection.filters.info_theory._class_encoding import joint_entropy_2var


@njit(nogil=True, cache=True, fastmath=False)
def _joint_entropy_one(fd, ia, ib, nb_a, nb_b, hist):
    n = fd.shape[0]
    if n == 0:
        return 0.0
    size = nb_a * nb_b
    for c in range(size):
        hist[c] = 0
    for r in range(n):
        hist[fd[r, ia] + fd[r, ib] * nb_a] += 1
    h = 0.0
    for c in range(size):
        cnt = hist[c]
        if cnt != 0:
            p = cnt / n
            h += np.log(p) * p
    return -h


@njit(nogil=True, cache=True, parallel=True)
def _batch_joint_entropy_over_pairs(fd, anchor, cands, nb_arr):
    """H(anchor, c) for every c in ``cands`` -- prange over the OUTER pair index."""
    k = cands.shape[0]
    out = np.empty(k, dtype=np.float64)
    nb_a = nb_arr[anchor]
    for i in prange(k):
        cb = cands[i]
        nb_b = nb_arr[cb]
        size = nb_a * nb_b
        hist = np.zeros(size, dtype=np.int64)
        n = fd.shape[0]
        for r in range(n):
            hist[fd[r, anchor] + fd[r, cb] * nb_a] += 1
        h = 0.0
        for c in range(size):
            cnt = hist[c]
            if cnt != 0:
                p = cnt / n
                h += np.log(p) * p
        out[i] = -h
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    print("threads", numba.get_num_threads())
    K = 200
    P = K + 1
    for n in (30000, 300000, 1000000):
        fd = rng.integers(0, 10, size=(n, P)).astype(np.int32)
        nb_arr = np.full(P, 10, dtype=np.int64)
        anchor = 0
        cands = np.arange(1, P, dtype=np.int64)
        # warm
        a = np.array([joint_entropy_2var(fd, anchor, int(c), 10, 10) for c in cands])
        b = _batch_joint_entropy_over_pairs(fd, anchor, cands, nb_arr)
        diff = float(np.max(np.abs(a - b)))
        R = 5
        # OLD: serial per-pair loop (mirrors discover_cluster_members)
        t = time.perf_counter()
        for _ in range(R):
            _ = np.array([joint_entropy_2var(fd, anchor, int(c), 10, 10) for c in cands])
        s = (time.perf_counter() - t) / R * 1e3
        # NEW: prange over pairs
        t = time.perf_counter()
        for _ in range(R):
            _ = _batch_joint_entropy_over_pairs(fd, anchor, cands, nb_arr)
        p = (time.perf_counter() - t) / R * 1e3
        print(f"n={n:>8} K={K} serial={s:9.2f}ms batch={p:9.2f}ms speedup={s / p:5.2f}x maxdiff={diff:.2e}")


if __name__ == "__main__":
    main()
