"""Bench: pair-index generation for the order-2 maxT permutation-null floor.

The old path built a full ``list(combinations(k, 2))`` Python tuple list before the kernel ran; at k~5000
that is ~12.5M tuple objects (~700MB) plus two int64 arrays. The new path emits the two int64 index arrays
directly via ``np.triu_indices`` in the identical lexicographic order (so the bias vector stays aligned and
the floor value is bit-identical). This bench reports wall + peak-RSS for both at a sweep of k, and asserts
the emitted (pa, pb) order is identical.

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_pair_maxt_index_gen
"""
from __future__ import annotations

import time
import tracemalloc
from itertools import combinations

import numpy as np


def _old(vars_list):
    pairs = list(combinations(vars_list, 2))
    pa = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    pb = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    return pa, pb


def _new(vars_list):
    # fromiter, not asarray: production passes a SET here (asarray cannot convert one), and fromiter
    # preserves the exact iteration order combinations() walks, keeping the pair order identical.
    kv = np.fromiter(vars_list, dtype=np.int64, count=len(vars_list))
    ia, ib = np.triu_indices(kv.shape[0], k=1)
    return kv[ia], kv[ib]


def _measure(fn, vars_list):
    tracemalloc.start()
    t0 = time.perf_counter()
    pa, pb = fn(vars_list)
    wall = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return pa, pb, wall, peak


def main() -> None:
    for k in (500, 2000, 5000):
        # A SET, matching what the FE step actually passes -- the first version of this bench used a list and
        # therefore never exercised the type that broke production.
        vars_list = set(range(k))
        pa_o, pb_o, w_o, p_o = _measure(_old, vars_list)
        pa_n, pb_n, w_n, p_n = _measure(_new, vars_list)
        assert np.array_equal(pa_o, pa_n) and np.array_equal(pb_o, pb_n), f"order diverged at k={k}"
        print(
            f"k={k:5d} n_pairs={pa_o.size:>10d} | old {w_o*1e3:8.2f}ms {p_o/1e6:8.1f}MB "
            f"| new {w_n*1e3:8.2f}ms {p_n/1e6:8.1f}MB | mem x{p_o/max(1,p_n):.1f} wall x{w_o/max(1e-9,w_n):.1f}"
        )


if __name__ == "__main__":
    main()
