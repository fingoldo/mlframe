"""Bench for cluster_stability_selection single-linkage step (CPX13).

The pre-fix step-1 ran an O(p^2) pure-Python double loop over an
already-vectorised |Pearson| matrix; replaced by np.triu adjacency + union-find
over the actual edges only. Same clusters. Run: python this.py
"""
from __future__ import annotations

import time

import numpy as np


def _find_factory(parent):
    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    return _find


def _old(C, corr_threshold, num_ok, p):
    parent = np.arange(p, dtype=np.int64)
    _find = _find_factory(parent)
    for i in range(p):
        for j in range(i + 1, p):
            if num_ok[i] and num_ok[j] and C[i, j] >= corr_threshold:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj
    return np.array([_find(i) for i in range(p)])


def _new(C, corr_threshold, num_ok, p):
    parent = np.arange(p, dtype=np.int64)
    _find = _find_factory(parent)
    adj = (C >= corr_threshold) & num_ok[:, None] & num_ok[None, :]
    ii, jj = np.where(np.triu(adj, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([_find(i) for i in range(p)])


def _compact(raw):
    uniq = np.unique(raw)
    m = {int(v): k for k, v in enumerate(uniq)}
    return np.array([m[int(v)] for v in raw])


def _best(fn, *a, reps=5):
    t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(*a); t.append(time.perf_counter() - s)
    return min(t)


def main():
    rng = np.random.default_rng(0)
    for p in (200, 500):
        n = 400
        X = rng.standard_normal((n, p))
        # induce some correlated blocks
        for b in range(0, p, 10):
            X[:, b + 1 : b + 5] = X[:, b : b + 1] + 0.01 * rng.standard_normal((n, min(4, p - b - 1)))
        Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
        C = np.abs(Z.T @ Z / n)
        num_ok = np.ones(p, dtype=bool)
        thr = 0.8
        assert np.array_equal(_compact(_old(C, thr, num_ok, p)), _compact(_new(C, thr, num_ok, p)))
        to = _best(_old, C, thr, num_ok, p); tn = _best(_new, C, thr, num_ok, p)
        print(f"p={p}: OLD {to*1e3:.2f}ms -> NEW {tn*1e3:.3f}ms ({to/tn:.1f}x) clusters identical")


if __name__ == "__main__":
    main()
