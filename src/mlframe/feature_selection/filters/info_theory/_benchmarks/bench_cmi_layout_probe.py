"""Probe: is the per-candidate melt gated by STRIDED column access?

factors_data is (n, nfeat) C-contiguous, so reading column xi strides nfeat*4 bytes.
The batched-GPU coalesced idea (#1) only helps on CPU if converting those strided
column reads to contiguous reads wins. Time the pruned loop on the real C-order matrix
vs an F-order (column-contiguous) copy of the SAME data. If F-order is dramatically
faster, a batched/relayout kernel is worth building; if ~flat, the melt is not
column-stride-bound and the batched lever is REJECTED.
"""
from __future__ import annotations
import time
import numpy as np


def main():
    from mlframe.feature_selection.filters.info_theory._cmi_cuda import _cpu_cmi_loop_hoisted_parallel
    n, p, nbins = 30000, 1000, 16
    rng = np.random.default_rng(0)
    ncols = p + 2
    c_data = np.empty((n, ncols), dtype=np.int32, order="C")
    for c in range(ncols):
        c_data[:, c] = rng.integers(0, nbins, n)
    f_data = np.asfortranarray(c_data)
    fnb = np.full(ncols, nbins, dtype=np.int64)
    cand = np.arange(p, dtype=np.int64)
    y = np.array([p], dtype=np.int64); z = np.array([p + 1], dtype=np.int64)

    def best_of(data, k=8):
        _cpu_cmi_loop_hoisted_parallel(data, cand, y, z, fnb)
        b = 1e30
        for _ in range(k):
            t0 = time.perf_counter()
            _cpu_cmi_loop_hoisted_parallel(data, cand, y, z, fnb)
            b = min(b, time.perf_counter() - t0)
        return b

    # interleave
    tc = best_of(c_data); tf = best_of(f_data)
    tc = min(tc, best_of(c_data)); tf = min(tf, best_of(f_data))
    print(f"C-order={tc*1000:.1f}ms  F-order(col-contig)={tf*1000:.1f}ms  ratio C/F={tc/tf:.2f}x")


if __name__ == "__main__":
    main()
