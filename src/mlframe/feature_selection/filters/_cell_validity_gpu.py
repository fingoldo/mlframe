"""Fused validity statistics of a block of candidate cells on the device.

Win axis: the usable-cell mask of the basis-cell paths (finite and not constant) was ``isfinite().all()`` plus ``std()`` plus a boolean take, six generic cupy launches that read the block three times;
this kernel reads it once. It wins on any block of a few hundred thousand elements or more; below that the launch cost is the same as one cupy call.

One thread block is 32 columns x 8 row lanes and the grid is (column tiles, row chunks), so every warp reads consecutive columns of one row (coalesced for the row-major block). Each thread keeps a Welford
state (count, mean, M2) of its rows, the 8 lanes merge in a fixed order (Chan), and each (chunk, column) writes one partial; the partials of the chunks are merged on the host in chunk order. No floating-point
atomics, so the result does not depend on scheduling. Values are read as ``{REAL}``, the statistics are ``double``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._raw_module_cache import RawModuleCache
from ._kernel_types import render

_SRC = render(r"""
extern "C" __global__ void cell_moments(const {REAL}* __restrict__ x, const long long n, const int m, const long long rows_per_chunk,
                                        {ACC}* __restrict__ part, int* __restrict__ bad) {
    __shared__ {ACC} sh_n[8][32];
    __shared__ {ACC} sh_mean[8][32];
    __shared__ {ACC} sh_m2[8][32];
    __shared__ int sh_bad[8][32];
    __shared__ {ACC} rcp[1025];   // reciprocals of the per-thread counts: a {ACC} division is 1/32 rate on Pascal
    const int col = blockIdx.x * 32 + threadIdx.x;
    const int ly = threadIdx.y;
    for (int t = threadIdx.y * 32 + threadIdx.x; t < 1025; t += 256) rcp[t] = t == 0 ? 0.0 : 1.0 / ({ACC})t;
    __syncthreads();
    const long long start = (long long)blockIdx.y * rows_per_chunk;
    long long stop = start + rows_per_chunk;
    if (stop > n) stop = n;
    {ACC} cnt = 0.0, mean = 0.0, m2 = 0.0;
    int nonfinite = 0;
    if (col < m) {
        for (long long i = start + ly; i < stop; i += 8) {
            const {ACC} v = ({ACC})x[i * (long long)m + col];
            if (!isfinite(v)) { nonfinite = 1; continue; }
            cnt += 1.0;
            const {ACC} delta = v - mean;
            mean += delta * rcp[(int)cnt];
            m2 += delta * (v - mean);
        }
    }
    sh_n[ly][threadIdx.x] = cnt; sh_mean[ly][threadIdx.x] = mean; sh_m2[ly][threadIdx.x] = m2; sh_bad[ly][threadIdx.x] = nonfinite;
    __syncthreads();
    if (ly == 0 && col < m) {
        {ACC} tn = sh_n[0][threadIdx.x], tmean = sh_mean[0][threadIdx.x], tm2 = sh_m2[0][threadIdx.x];
        int tbad = sh_bad[0][threadIdx.x];
        for (int l = 1; l < 8; ++l) {   // fixed merge order
            const {ACC} bn = sh_n[l][threadIdx.x];
            tbad |= sh_bad[l][threadIdx.x];
            if (bn == 0.0) continue;
            const {ACC} bmean = sh_mean[l][threadIdx.x], bm2 = sh_m2[l][threadIdx.x];
            const {ACC} tot = tn + bn;
            const {ACC} d = bmean - tmean;
            tm2 = tm2 + bm2 + d * d * tn * bn / tot;
            tmean = tmean + d * bn / tot;
            tn = tot;
        }
        const long long o = ((long long)blockIdx.y * m + col) * 3;
        part[o] = tn; part[o + 1] = tmean; part[o + 2] = tm2;
        bad[(long long)blockIdx.y * m + col] = tbad;
    }
}
""")

_MODULES_BY_REAL = {"double": RawModuleCache(_SRC.replace("{REAL}", "double")), "float": RawModuleCache(_SRC.replace("{REAL}", "float"))}
_ROWS_PER_CHUNK = 8192  # at most 1024 rows per lane: the size of the reciprocal table


def cell_validity_gpu(cp: Any, cand: Any, min_std: float = 1e-12) -> "tuple[np.ndarray, np.ndarray]":
    """``(usable (m,) bool, mean (m,) double)`` of every column of the row-major ``(n, m)`` device block ``cand``.

    A column is usable when all its values are finite and its population standard deviation exceeds ``min_std`` (the host filter ``isfinite().all() and std() > 1e-12``).
    """
    n, m = int(cand.shape[0]), int(cand.shape[1])
    if n == 0 or m == 0:
        return np.zeros(m, dtype=bool), np.zeros(m, dtype=np.float64)  # Intentional float64: statistic vector
    block = cp.ascontiguousarray(cand)
    chunks = max(1, (n + _ROWS_PER_CHUNK - 1) // _ROWS_PER_CHUNK)
    # Intentional float64: device accumulator for the per-column moments
    part = cp.zeros((chunks, m, 3), dtype=cp.float64)
    bad = cp.zeros((chunks, m), dtype=cp.int32)
    module = _MODULES_BY_REAL["float" if block.dtype.itemsize == 4 else "double"].get(cp)
    module.get_function("cell_moments")(((m + 31) // 32, chunks), (32, 8), (block, np.int64(n), np.int32(m), np.int64(_ROWS_PER_CHUNK), part, bad))
    host_part = cp.asnumpy(part)
    host_bad = cp.asnumpy(bad).any(axis=0)
    cnt = host_part[0, :, 0].copy()
    mean = host_part[0, :, 1].copy()
    m2 = host_part[0, :, 2].copy()
    for c in range(1, chunks):  # chunk order: deterministic
        bn, bmean, bm2 = host_part[c, :, 0], host_part[c, :, 1], host_part[c, :, 2]
        tot = cnt + bn
        safe = np.where(tot > 0, tot, 1.0)
        d = bmean - mean
        m2 = np.where(bn > 0, m2 + bm2 + d * d * cnt * bn / safe, m2)
        mean = np.where(bn > 0, mean + d * bn / safe, mean)
        cnt = tot
    std = np.sqrt(np.where(cnt > 0, m2 / np.where(cnt > 0, cnt, 1.0), 0.0))
    usable = (~host_bad) & np.isfinite(std) & (std > min_std)
    return usable, mean


__all__ = ["cell_validity_gpu"]
