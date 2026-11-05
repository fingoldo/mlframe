"""Column-tiled |Pearson corr| of a row-major candidate block with a target (version 2 of the ``abs_corr`` kernel of ``_gpu_resident_basis``).

Win axis: v1 runs one block per column and every thread strides through the row-major block with step ``m``, so a warp touches 32 different 128-byte lines (measured global-load efficiency 22%).
This version has a block of 32 columns x 8 row lanes, so a warp reads 32 consecutive columns of one row. It wins for ``m >= 32`` columns; for fewer columns the blocks are mostly idle and v1 is kept.

Centred sums (``sum (v - mu)^2`` and ``sum (v - mu) * yc``) are accumulated in double per thread, merged over the 8 lanes in a fixed order and written per row chunk; the chunks are summed on the host in
chunk order, so the result does not depend on scheduling. ``{REAL}`` is the element type of the block.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._raw_module_cache import RawModuleCache
from ._kernel_types import render

_SRC = render(r"""
extern "C" __global__ void abs_corr_tiled(const {REAL}* __restrict__ cand, const {REAL}* __restrict__ yc, const {REAL}* __restrict__ mu, const long long n, const int m,
                                          const long long rows_per_chunk, {ACC}* __restrict__ part) {
    __shared__ {ACC} sh_cc[8][32];
    __shared__ {ACC} sh_cy[8][32];
    const int col = blockIdx.x * 32 + threadIdx.x;
    const int ly = threadIdx.y;
    const long long start = (long long)blockIdx.y * rows_per_chunk;
    long long stop = start + rows_per_chunk;
    if (stop > n) stop = n;
    {ACC} scc = 0.0, scy = 0.0;
    if (col < m) {
        const {ACC} muc = ({ACC})mu[col];
        for (long long i = start + ly; i < stop; i += 8) {
            const {ACC} d = ({ACC})cand[i * (long long)m + col] - muc;
            scc += d * d;
            scy += d * ({ACC})yc[i];
        }
    }
    sh_cc[ly][threadIdx.x] = scc; sh_cy[ly][threadIdx.x] = scy;
    __syncthreads();
    if (ly == 0 && col < m) {
        {ACC} a = sh_cc[0][threadIdx.x], b = sh_cy[0][threadIdx.x];
        for (int l = 1; l < 8; ++l) { a += sh_cc[l][threadIdx.x]; b += sh_cy[l][threadIdx.x]; }   // fixed merge order
        const long long o = ((long long)blockIdx.y * m + col) * 2;
        part[o] = a; part[o + 1] = b;
    }
}
""")

_MODULES_BY_REAL = {"double": RawModuleCache(_SRC.replace("{REAL}", "double")), "float": RawModuleCache(_SRC.replace("{REAL}", "float"))}
_ROWS_PER_CHUNK = 8192
MIN_COLUMNS = 32


def abs_corr_tiled(cp: Any, cand: Any, yc: Any, mu: Any, syy: float) -> Any:
    """``|corr|`` of each column of the row-major ``(n, m)`` block ``cand`` with the centred target ``yc``, or -1.0 for a degenerate column (the v1 convention); ``mu`` are the column means."""
    n, m = int(cand.shape[0]), int(cand.shape[1])
    chunks = max(1, (n + _ROWS_PER_CHUNK - 1) // _ROWS_PER_CHUNK)
    # Intentional float64: device accumulator for the tiled correlation sums
    part = cp.zeros((chunks, m, 2), dtype=cp.float64)
    module = _MODULES_BY_REAL["float" if cand.dtype.itemsize == 4 else "double"].get(cp)
    module.get_function("abs_corr_tiled")(((m + 31) // 32, chunks), (32, 8), (cand, yc, mu, np.int64(n), np.int32(m), np.int64(_ROWS_PER_CHUNK), part))
    host = cp.asnumpy(part)
    scc = host[0, :, 0].copy()
    scy = host[0, :, 1].copy()
    for c in range(1, chunks):  # chunk order: deterministic
        scc += host[c, :, 0]
        scy += host[c, :, 1]
    den = np.sqrt(scc * syy)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(den > 1e-300, np.abs(scy / den), 0.0)
    ok = np.isfinite(corr) & (np.sqrt(scc) > 1e-12)
    return cp.asarray(np.where(ok, corr, -1.0))
