"""A/B bench for ``_fused_resample_auc``'s per-call scratch-array allocation (2026-07-31).

The bootstrap AUC resampler called ``_fused_resample_auc``, which allocated fresh
``np.zeros(n, dtype=np.int64)`` ``counts``/``ones`` buffers on EVERY resample. At n=2M and
~1000 resamples that's ~1000 mallocs of two 16MB int64 arrays -- surfaced as the #1 mlframe
hotspot by tottime in a reporting/charts cProfile (7.884s / 1001 calls,
``profile_one_combo.py --combo c0016_cbe1b080 --rows 2000000 --save-charts``).

Confirms bit-identity between the old (fresh-alloc) and new (caller-owned scratch, reset
in-kernel) kernel and measures the wall-time win at that shape.

Usage:
    python profiling/bench_fused_resample_auc_scratch_reuse.py
"""

from __future__ import annotations

import time

import numba
import numpy as np

from mlframe.metrics._core_auc_brier import _fused_resample_auc
from mlframe.metrics._numba_params import NUMBA_NJIT_PARAMS


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_resample_auc_fresh_alloc(idx, base_rank, y_by_rank, n):
    """Pre-optimization baseline: fresh np.zeros(n) allocation every call."""
    counts = np.zeros(n, dtype=np.int64)
    ones = np.zeros(n, dtype=np.int64)
    m = idx.shape[0]
    for k in range(m):
        r = base_rank[idx[k]]
        counts[r] += 1
        ones[r] += y_by_rank[r]
    last_fps = 0
    last_tps = 0
    tps = 0
    fps = 0
    auc = 0
    for r in range(n - 1, -1, -1):
        c = counts[r]
        if c == 0:
            continue
        pos = ones[r]
        neg = c - pos
        tps += pos
        fps += neg
        auc += (fps - last_fps) * (last_tps + tps)
        last_fps = fps
        last_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    return np.nan


def main():
    rng = np.random.default_rng(0)
    n = 2_000_000
    n_resamples = 1001

    y_score = rng.random(n)
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    asc_order = np.argsort(y_score)
    base_rank = np.empty(n, dtype=np.int64)
    base_rank[asc_order] = np.arange(n, dtype=np.int64)
    y_by_rank = np.ascontiguousarray(y_true[asc_order].astype(np.int64))

    idxs = [rng.integers(0, n, size=n, dtype=np.int64) for _ in range(n_resamples)]

    # warm both kernels' JIT compilation before timing
    _fused_resample_auc_fresh_alloc(idxs[0], base_rank, y_by_rank, n)
    counts = np.zeros(n, dtype=np.int64)
    ones = np.zeros(n, dtype=np.int64)
    _fused_resample_auc(idxs[0], base_rank, y_by_rank, n, counts, ones)

    t0 = time.perf_counter()
    old_results = [_fused_resample_auc_fresh_alloc(idx, base_rank, y_by_rank, n) for idx in idxs]
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_results = [_fused_resample_auc(idx, base_rank, y_by_rank, n, counts, ones) for idx in idxs]
    t_new = time.perf_counter() - t0

    identical = old_results == new_results
    print(f"n={n:,} resamples={n_resamples}")
    print(f"old (fresh alloc per call): {t_old:.4f}s")
    print(f"new (reused scratch):       {t_new:.4f}s")
    print(f"speedup: {t_old / t_new:.2f}x")
    print(f"bit-identical: {identical}")
    assert identical, "reused-scratch kernel must be bit-identical to the fresh-alloc baseline"


if __name__ == "__main__":
    main()
