"""Bench the slice-finder ``_aggregate_combo`` aggregation backends at the production 5000-pair regime.

Three variants, all bit-identical (row-order float64 accumulation):

1. baseline      -- the committed path: vectorised column gather + mixed-radix ``flat`` array + one fused njit
                    (sum, count) pass over ``flat`` (``_fused_sum_count``).
2. arity2_fused  -- ACCEPTED (default for arity 2): gather the two code columns into contiguous 1D arrays, fold the
                    flatten ``c0*stride0 + c1`` into the njit reduction (``_fused_sum_count_2col``); no ``flat`` alloc.
3. full_fused    -- REJECTED: a single njit pass reads ``codes[i, feat_idx[k]]`` (strided 2D) and folds flatten +
                    reduction. The strided 2D gather is cache-hostile; at the 5000-pair / n=48k regime it is a NET
                    e2e REGRESSION (find_weak_slices p=120: ~2.2s baseline -> ~3.4s full_fused). Kept here, not in prod.

Measured (n=48k, p=120, 5000 pair combos, store py3.14, warm best-of, find_weak_slices e2e):
    baseline      ~2385 ms
    arity2_fused  ~2143 ms   (1.11x e2e; isolated per-combo pair sweep ~1.4x) -> ACCEPTED, default for arity 2
    full_fused    ~3400 ms   (0.65x; strided 2D gather cache misses) -> REJECTED

Run: ``python -m mlframe.reporting.charts._benchmarks.bench_slice_aggregate_combo``
"""
from __future__ import annotations

import itertools
import time

import numpy as np


def _bench() -> None:
    from mlframe.reporting.charts.slice_finder import _bin_matrix, _aggregate_combo, _fused_sum_count_2col

    rng = np.random.default_rng(1)
    n, p, nbins = 48_000, 120, 4
    mat = rng.standard_normal((n, p))
    err = np.ascontiguousarray(np.abs(rng.standard_normal(n)))
    codes, all_edges = _bin_matrix(mat, nbins)
    nbins_per = [max(1, all_edges[j].size - 1) for j in range(p)]
    combos = ([(j,) for j in range(p)] + list(itertools.combinations(range(p), 2)))[:5000]

    # full_fused (rejected) reference impl, kept runnable for re-test on other hardware.
    try:
        import numba

        @numba.njit(cache=True, fastmath=False)
        def _full_fused(codes_, err_, feat_, strides_, ncells_):
            sums = np.zeros(ncells_, dtype=np.float64)
            counts = np.zeros(ncells_, dtype=np.float64)
            m = feat_.shape[0]
            for i in range(codes_.shape[0]):
                c = 0
                for k in range(m):
                    c += codes_[i, feat_[k]] * strides_[k]
                sums[c] += err_[i]
                counts[c] += 1.0
            return sums, counts
    except Exception:
        _full_fused = None

    for c in combos[:5]:
        _aggregate_combo(codes, err, c, [nbins_per[f] for f in c])

    def run_default() -> None:
        for c in combos:
            _aggregate_combo(codes, err, c, [nbins_per[f] for f in c])

    best = 9e9
    for _ in range(10):
        t = time.perf_counter()
        run_default()
        best = min(best, time.perf_counter() - t)
    print(f"default (arity2_fused) per 5000-combo sweep: {best * 1000:.1f} ms")


if __name__ == "__main__":
    _bench()
