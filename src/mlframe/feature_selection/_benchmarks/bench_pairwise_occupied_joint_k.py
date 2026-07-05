"""Bench: ``_pairwise_occupied_joint_k`` pure-Python set-per-pair loop vs njit boolean-seen.

The order-2 maxT permutation-null + auto-prevalence-debias paths call
``pairwise_mm_joint_bias`` -> ``_pairwise_occupied_joint_k`` on the FULL C(p,2) pair pool
of numeric vars once per FE step. The OLD body builds a Python ``set`` per pair and walks
all n rows in interpreter (O(n_pairs * n) with per-row hashing + boxing). The result is just
the count of DISTINCT joint codes ``a*nbins_b + b`` per pair -- exactly reproducible with a
flat boolean "seen" buffer of length nbins_a*nbins_b under numba, so the NEW kernel is
BIT-IDENTICAL by construction (it counts the same distinct codes).

Run:  CUDA_VISIBLE_DEVICES="" python bench_pairwise_occupied_joint_k.py
"""
from __future__ import annotations

import time
from itertools import combinations

import numba
import numpy as np


# ---- OLD: verbatim pure-Python reference (the prior _pairwise_occupied_joint_k body) ----
def _old_pairwise_occupied_joint_k(factors_data, pair_a, pair_b, nbins):
    n = int(factors_data.shape[0])
    n_pairs = int(pair_a.shape[0])
    out = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        a = int(pair_a[p]); b = int(pair_b[p])
        nb_b = int(nbins[b])
        seen = set()
        for i in range(n):
            seen.add(int(factors_data[i, a]) * nb_b + int(factors_data[i, b]))
        out[p] = len(seen)
    return out


# ---- NEW: njit boolean-seen kernel (bit-identical distinct-count) ----
@numba.njit(cache=True)
def _new_pairwise_occupied_joint_k(factors_data, pair_a, pair_b, nbins):
    n = factors_data.shape[0]
    n_pairs = pair_a.shape[0]
    out = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        a = pair_a[p]; b = pair_b[p]
        nb_a = nbins[a]; nb_b = nbins[b]
        seen = np.zeros(nb_a * nb_b, dtype=np.uint8)
        cnt = 0
        for i in range(n):
            code = factors_data[i, a] * nb_b + factors_data[i, b]
            if seen[code] == 0:
                seen[code] = 1
                cnt += 1
        out[p] = cnt
    return out


def _make(n, p, card, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, card, size=(n, p)).astype(np.int32)
    nbins = np.full(p, card, dtype=np.int64)
    pairs = list(combinations(range(p), 2))
    pa = np.fromiter((x[0] for x in pairs), dtype=np.int64, count=len(pairs))
    pb = np.fromiter((x[1] for x in pairs), dtype=np.int64, count=len(pairs))
    return data, pa, pb, nbins


def best_of(fn, *a, reps=5):
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        t.append(time.perf_counter() - s)
    return min(t)


if __name__ == "__main__":
    shapes = [
        (2000, 30, 10),
        (10000, 50, 10),
        (50000, 40, 16),
    ]
    # warm njit
    d, pa, pb, nb = _make(64, 6, 8)
    _new_pairwise_occupied_joint_k(d, pa, pb, nb)

    print(f"{'shape (n,p,card)':>22} {'pairs':>7} {'OLD ms':>10} {'NEW ms':>10} {'speedup':>8} identical")
    for n, p, card in shapes:
        d, pa, pb, nb = _make(n, p, card)
        old = _old_pairwise_occupied_joint_k(d, pa, pb, nb)
        new = _new_pairwise_occupied_joint_k(d, pa, pb, nb)
        ident = bool(np.array_equal(old, new))
        t_old = best_of(_old_pairwise_occupied_joint_k, d, pa, pb, nb, reps=3)
        t_new = best_of(_new_pairwise_occupied_joint_k, d, pa, pb, nb, reps=5)
        print(f"{str((n,p,card)):>22} {len(pa):>7} {t_old*1e3:>10.2f} {t_new*1e3:>10.2f} " f"{t_old/t_new:>7.2f}x {ident}")
