"""cProfile + wall A/B for the group-blocked relevance MI kernel (``info_theory/_group_mi.group_blocked_mi``).

Run: ``python -m mlframe.feature_selection._benchmarks.bench_group_blocked_mi``

The per-group tabulation loops G groups building a small (k_x,k_y) joint bincount each; the cost center is that loop
(G x candidates x rounds). This harness times it at a TVT-ish shape (773 groups, 5000 rows/group binned to 8x8) against
the equivalent single global ``compute_mi_from_classes`` so the overhead multiple is documented.

Conclusion (dev box, cache-warm njit): group-blocking a single candidate at 773 groups / ~3.86M rows measured 15.8 ms
vs 7.0 ms for a single global ``compute_mi_from_classes`` = 2.24x. The extra work is one O(n) segmented gather + G tiny
(8x8) joint tabulations with per-group Miller-Madow debias, all inside one njit body (cProfile shows 100% of time in the
single kernel frame, no Python-level hotspot). No actionable speedup: the kernel already reuses per-group scratch across
groups (no per-group allocation) and does one segment-sorted pass. GPU is deliberately off under group_aware_mi (the
batched GPU MI does not group-block); a GPU group-blocked kernel (block per (column,group)) is a documented later
optimization behind the same dispatcher. 2.24x on the relevance MI is acceptable given group_aware_mi is opt-in.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters.info_theory._group_mi import group_blocked_mi, prepare_group_segments
from mlframe.feature_selection.filters.info_theory._class_mi_kernels import compute_mi_from_classes


def _make(G: int = 773, per: int = 5000, k: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(G), per)
    n = groups.size
    cy = rng.integers(0, k, size=n).astype(np.int32)
    # within-group-correlated cx so the MI is non-trivial
    cx = ((cy + rng.integers(0, 3, size=n)) % k).astype(np.int32)
    return groups, cx, cy, k


def _freqs(codes, k):
    f = np.bincount(codes[codes >= 0], minlength=k).astype(np.float64)
    return f


def main():
    groups, cx, cy, k = _make()
    n = groups.size
    si, off = prepare_group_segments(groups)

    # warm both kernels
    group_blocked_mi(cx, cy, si, off, k, k, min_rows=20)
    fx, fy = _freqs(cx, k), _freqs(cy, k)
    compute_mi_from_classes(cx, fx, cy, fy)

    REP = 20
    t0 = time.perf_counter()
    for _ in range(REP):
        g = group_blocked_mi(cx, cy, si, off, k, k, min_rows=20)
    t_group = (time.perf_counter() - t0) / REP

    t0 = time.perf_counter()
    for _ in range(REP):
        m = compute_mi_from_classes(cx, fx, cy, fy)
    t_global = (time.perf_counter() - t0) / REP

    print(f"shape: n={n:,} groups={len(off) - 1} k={k}x{k}")
    print(f"group-blocked MI : {t_group * 1e3:8.3f} ms/call  (value {g:.4f})")
    print(f"global   MI      : {t_global * 1e3:8.3f} ms/call  (value {m:.4f})")
    print(f"overhead multiple: {t_group / t_global:6.2f}x")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(REP):
        group_blocked_mi(cx, cy, si, off, k, k, min_rows=20)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
