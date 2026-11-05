"""Device twin of ``_orthogonal_triplet_fe._triple_mi_kernel``: the plug-in MI of every triple of joint quantile codes with the target.

Win axis: the host kernel makes one counting pass over the rows per triple (C(p, 3) passes, each into a ``base**3 * n_y`` table). Here one block owns one triple: its threads stride over the rows ONCE, reading the three
codes (one byte each, coalesced along the column) and the class, and count into a shared-memory table of ``base**3 * n_y`` integers; the MI of the table is finished in the same kernel with the host's formula and cell
order (cell-major, class-minor), so the value equals the host's up to the device ``log``. Triples are enumerated in lexicographic order, as the host writes them.

Handled when the table fits the shared budget (``base**3 * n_y * 4`` bytes) and the codes fit a byte; otherwise the host kernel runs.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ._raw_module_cache import RawModuleCache
from ._kernel_types import render

logger = logging.getLogger(__name__)

THREADS = 256
SHARED_BUDGET_BYTES = 40 * 1024
#: Rows x triples from which the device is the default choice (the host kernel is memory-latency bound well below this).
DEVICE_MIN_WORK = 5_000_000
TRIPLES_PER_LAUNCH = 1 << 20

_SRC = render(r"""
extern "C" __global__ void triple_mi(const signed char* __restrict__ codes, const unsigned char* __restrict__ Y, const int* __restrict__ TI, const int* __restrict__ TJ,
                                     const int* __restrict__ TK, const long long n, const int base, const int n_y, const int n_cells, {ACC}* __restrict__ out) {
    extern __shared__ int sh[];                       // (n_cells, n_y) joint table, cell-major
    const int t = blockIdx.x;
    const signed char* ci = codes + (long long)TI[t] * n;
    const signed char* cj = codes + (long long)TJ[t] * n;
    const signed char* ck = codes + (long long)TK[t] * n;
    const int size = n_cells * n_y;
    for (int s = threadIdx.x; s < size; s += blockDim.x) sh[s] = 0;
    __syncthreads();
    for (long long r = threadIdx.x; r < n; r += blockDim.x) {
        const int cell = (((int)ci[r] * base + (int)cj[r]) * base + (int)ck[r]) * n_y + (int)Y[r];
        atomicAdd(&sh[cell], 1);
    }
    __syncthreads();
    // The MI tail is spread over the block: class totals, then one term per cell, then a short ordered sum.
    __shared__ long long pys[16];
    __shared__ {ACC} term[1024];
    if (threadIdx.x < n_y) {
        long long tot = 0;
        for (int c = 0; c < n_cells; ++c) tot += sh[c * n_y + threadIdx.x];
        pys[threadIdx.x] = tot;
    }
    __syncthreads();
    const {ACC} nn = ({ACC})n;
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x) {
        long long px = 0;
        for (int v = 0; v < n_y; ++v) px += sh[c * n_y + v];
        {ACC} tc = 0.0;
        if (px != 0) {
            for (int v = 0; v < n_y; ++v) {
                const long long cnt = sh[c * n_y + v];
                if (cnt == 0) continue;
                tc += (({ACC})cnt / nn) * log((({ACC})cnt * nn) / (({ACC})px * ({ACC})pys[v]));
            }
        }
        term[c] = tc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} mi = 0.0;
        for (int c = 0; c < n_cells; ++c) mi += term[c];
        out[t] = mi;
    }
}
""")

_MODULE = RawModuleCache(_SRC)


def _triples(p: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Every ``i < j < k`` of ``range(p)`` in lexicographic order (the order the host kernel writes them)."""
    i, j, k = np.nonzero(_strict_order_mask(p))
    return i.astype(np.int32), j.astype(np.int32), k.astype(np.int32)


def _strict_order_mask(p: int) -> np.ndarray:
    """Boolean ``(p, p, p)`` mask of the index triples with ``i < j < k``."""
    idx = np.arange(p)
    return (idx[:, None, None] < idx[None, :, None]) & (idx[None, :, None] < idx[None, None, :])


def triple_mi_gpu(codes: np.ndarray, y: np.ndarray, base: int, n_y: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None":
    """``(out_i, out_j, out_k, out_mi)`` as the host kernel returns them, or ``None`` when the device path does not apply (table too big for shared memory, codes wider than a byte, more than 16 classes, no cupy).

    ``codes`` is ``(p, n)`` with values in ``[0, base)``; ``y`` the dense class code of each row.
    """
    p, n = int(codes.shape[0]), int(codes.shape[1])
    n_cells = int(base) ** 3
    if base > 127 or n_y > 16 or n_cells > 1024 or n_cells * n_y * 4 > SHARED_BUDGET_BYTES or p < 3:
        return None
    try:
        import cupy as cp
    except Exception as exc:
        logger.debug("device triple MI unavailable (%s)", exc)
        return None
    ti, tj, tk = _triples(p)
    m = int(ti.size)
    kernel = _MODULE.get(cp).get_function("triple_mi")
    codes_dev = cp.asarray(np.ascontiguousarray(codes, dtype=np.int8))
    y_dev = cp.asarray(np.ascontiguousarray(y, dtype=np.uint8))
    out = np.empty(m, dtype=np.float64)  # Intentional float64: MI statistic vector compared with the host's
    for s in range(0, m, TRIPLES_PER_LAUNCH):
        e = min(m, s + TRIPLES_PER_LAUNCH)
        # Intentional float64: MI statistic vector compared with the host's
        d_out = cp.empty(e - s, dtype=cp.float64)
        kernel((e - s,), (THREADS,), (codes_dev, y_dev, cp.asarray(ti[s:e]), cp.asarray(tj[s:e]), cp.asarray(tk[s:e]), np.int64(n), np.int32(base), np.int32(n_y), np.int32(n_cells), d_out), shared_mem=n_cells * n_y * 4)
        out[s:e] = cp.asnumpy(d_out)
    return ti.astype(np.int64), tj.astype(np.int64), tk.astype(np.int64), out


def use_device_triples(n_rows: int, n_triples: int) -> bool:
    """Whether ``n_triples`` triples over ``n_rows`` rows are scored on the device: ``MLFRAME_TRIPLE_MI_DEVICE=0|1`` forces a path, else from ``DEVICE_MIN_WORK`` row-triples."""
    forced = os.environ.get("MLFRAME_TRIPLE_MI_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return int(n_rows) * int(n_triples) >= DEVICE_MIN_WORK


__all__ = ["triple_mi_gpu", "use_device_triples"]
