"""Thread-scaling / memory-bandwidth probe for the pruned hoisted CPU-CMI loop.

Set NUMBA_NUM_THREADS before import and time _cpu_cmi_loop_hoisted_parallel at the
wellbore redundancy shape. If wall stops improving well below the core count the
per-candidate melt is memory-bandwidth-gated (documents the prange verdict).

Run:  NUMBA_NUM_THREADS=<k> python -m ...bench_cmi_thread_scaling
"""
from __future__ import annotations
import os, time
import numpy as np


def main():
    from mlframe.feature_selection.filters.info_theory._cmi_cuda import _cpu_cmi_loop_hoisted_parallel
    from numba import get_num_threads
    n, p, nbins = 30000, 1000, 16
    rng = np.random.default_rng(0)
    ncols = p + 2
    data = np.empty((n, ncols), dtype=np.int32)
    for c in range(ncols):
        data[:, c] = rng.integers(0, nbins, n)
    fnb = np.full(ncols, nbins, dtype=np.int64)
    cand = np.arange(p, dtype=np.int64)
    y = np.array([p], dtype=np.int64); z = np.array([p + 1], dtype=np.int64)
    _cpu_cmi_loop_hoisted_parallel(data, cand, y, z, fnb)  # warm
    best = 1e30
    for _ in range(8):
        t0 = time.perf_counter()
        _cpu_cmi_loop_hoisted_parallel(data, cand, y, z, fnb)
        best = min(best, time.perf_counter() - t0)
    print(f"threads={get_num_threads()} n={n} p={p} nbins={nbins}: {best*1000:.1f}ms")


if __name__ == "__main__":
    main()
