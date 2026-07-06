"""Bench the per-bin mask construction in compute_fairness_metrics.

OLD: ``bins == bin_name`` (full O(N) equality) inside the per-bin loop -> O(N*B).
NEW-A: precompute all masks once via dict comprehension (still B full passes; only saves repeated np.asarray).
NEW-B: pd.factorize once + boolean masks from codes (single O(N) factorize + B==int passes on int codes).

Run: python -m mlframe.metrics._benchmarks.bench_fairness_bin_masks
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def old(bins, unique_bins):
    out = {}
    for bn in unique_bins:
        idx = np.asarray(bins == bn)
        out[bn] = idx.sum()
    return out


def new_a(bins, unique_bins):
    arr = np.asarray(bins)
    masks = {bn: (arr == bn) for bn in unique_bins}
    out = {}
    for bn in unique_bins:
        out[bn] = masks[bn].sum()
    return out


def new_b(bins, unique_bins):
    codes, uniq = pd.factorize(np.asarray(bins))
    pos = {u: i for i, u in enumerate(uniq)}
    masks = {bn: (codes == pos[bn]) for bn in unique_bins}
    out = {}
    for bn in unique_bins:
        out[bn] = masks[bn].sum()
    return out


def bestof(fn, *a, n=50):
    fn(*a)
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    for N, B in ((5000, 10), (50000, 20), (200000, 50), (200000, 200)):
        bins = pd.Series(rng.integers(0, B, size=N).astype(str))
        ub = bins.unique()
        o = bestof(old, bins, ub)
        a = bestof(new_a, bins, ub)
        b = bestof(new_b, bins, ub)
        # identity of counts
        assert old(bins, ub) == new_a(bins, ub) == new_b(bins, ub)  # nosec B101 - internal invariant check in src/mlframe/metrics/_benchmarks, not reachable with untrusted input
        print(f"N={N:7d} B={B:4d}  OLD {o*1e3:8.3f}ms  A {a*1e3:8.3f}ms ({o/a:.2f}x)  B {b*1e3:8.3f}ms ({o/b:.2f}x)")


if __name__ == "__main__":
    main()
