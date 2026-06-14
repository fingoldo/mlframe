"""iter94 A/B: ``_mi_from_binned_pair`` strided-column hot loop in composite-discovery screening.

The screening per-feature MI loop (``_mi_per_feature_prebinned``) calls ``_mi_from_binned_pair``
once per feature column with ``feature_binned[:, j]`` -- a STRIDED slice of a C-contiguous (n, F)
int16 prebin matrix. At HEAD the wrapper ran ``np.ascontiguousarray`` on BOTH inputs, forcing a
full O(n) copy of every strided column on each of the ~2.4k hot-loop calls (profiled 0.244s tottime
/ 2406 calls at 1M, cumtime ~= tottime so the cost is the wrapper-local copies, not the njit kernel).

The njit kernel indexes element-by-element (``int(x_idx[i])``), so it consumes a strided slice / any
integer dtype directly; the copies were pure waste. This bench measures the per-column MI loop OLD
(2x ascontiguousarray) vs NEW (pass-through), bit-identity included.

Run:
    MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
        python -m mlframe.training.composite.discovery._benchmarks.bench_iter94_mi_binned_pair_strided
"""
from __future__ import annotations

import sys

sys.modules.setdefault("cupy", None)
import scipy.stats  # noqa: F401,E402
import numba  # noqa: F401,E402

import statistics  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from ..screening import (  # noqa: E402
    _mi_from_binned_pair,
    _mi_from_binned_pair_njit_kernel,
)


def _wrapper_old(x_idx, y_idx, *, nbins):
    """HEAD wrapper: ascontiguousarray on both inputs before the njit kernel."""
    xi = np.ascontiguousarray(x_idx)
    yi = np.ascontiguousarray(y_idx)
    return float(_mi_from_binned_pair_njit_kernel(xi, yi, int(nbins)))


def _bench_one(n, F, nbins, reps):
    rng = np.random.default_rng(0)
    binned = np.empty((n, F), dtype=np.int16)
    for j in range(F):
        binned[:, j] = rng.integers(0, nbins, n)
    t_idx = rng.integers(0, nbins, n).astype(np.int64)
    # warm both compile paths
    for j in range(F):
        _wrapper_old(binned[:, j], t_idx, nbins=nbins)
        _mi_from_binned_pair(binned[:, j], t_idx, nbins=nbins)

    def loop(fn):
        s = 0.0
        for j in range(F):
            s += fn(binned[:, j], t_idx, nbins=nbins)
        return s

    s_old = loop(_wrapper_old)
    s_new = loop(_mi_from_binned_pair)
    assert abs(s_old - s_new) == 0.0, (n, F, nbins, s_old, s_new)
    told, tnew = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); loop(_wrapper_old); told.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); loop(_mi_from_binned_pair); tnew.append(time.perf_counter() - t0)
    mo, mn = statistics.median(told) * 1000, statistics.median(tnew) * 1000
    print(f"n={n:>8} F={F} nbins={nbins}: OLD {mo:7.3f}ms -> NEW {mn:7.3f}ms  {mo/mn:.2f}x  (loop-MI bit-identical: diff={abs(s_old-s_new)})")


def main():
    for n in (20_000, 100_000, 500_000):
        _bench_one(n, F=30, nbins=50, reps=120)


if __name__ == "__main__":  # pragma: no cover
    main()
