"""Conditional MI of within-stratum shuffled columns, counted per stratum in shared memory instead of through global atomics.

The permutation null of ``CMI(x; y | z)`` shuffles a candidate inside the strata of ``z``. In the z-sorted row order every stratum is a contiguous run of a few rows, so the ``(x, y)`` cell
counts of a stratum fit a small private histogram; the entropy and occupied-cell count of the ``(x, y, z)`` joint are the sums over strata, with no scattered atomics into a table of
``Kx * Kyz`` cells per column (2.5 million of them for 25 permutations of 100000 rows, the slow half of the null once the joint outgrows shared memory).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

from mlframe.utils.log_throttle import log_throttle

from ._kernel_types import render

logger = logging.getLogger(__name__)

#: Cells of one stratum's private ``(x, y)`` histogram the kernel handles, and the threads that each own one stratum.
MAX_CELLS = 64
_THREADS = 128
#: Below this many strata the per-thread serial loops cannot fill the device, and the joint is small enough for the shared-memory count anyway.
MIN_STRATA = 256

# Win axis. Two kernels, one launcher; the shape rule picks (``LONG_STRATUM_ROWS``):
#   direct (v1, ``strata_xy_entropy``): one thread per stratum reads its own run from global memory. Wins with short strata (20 rows and fewer, 5000 strata and more) and wide (x, y) tables, where the
#                private histograms avoid shared atomics. nvprof at 5000 strata: 236 us, gld efficiency 6.7% (9.3 transactions per request: the lanes' runs are one stratum apart), warp execution 78%.
#   rows (v3, ``strata_xy_entropy_rows``): the block's threads walk the rows of its 128 strata together (coalesced 4-byte stratum ranks and byte codes, gld efficiency 64%, warp execution 98%) and count
#                into one shared table with shared atomics. Wins with long strata (300 and 1000 strata: 0.81 / 0.57 ms against 0.96 / 0.77), loses with short ones and wide tables through bank conflicts
#                and same-cell atomics (5000 strata 8 x 8: 1.06 ms against 0.67, kernel time).
#: Average stratum length (rows) from which the rows kernel runs.
LONG_STRATUM_ROWS = 64

# A third, staged variant (rows of the block loaded coalesced into shared memory, then read per thread) measured equal to ``rows`` at long strata and slower at short ones, and was dropped.
_SRC = r"""
// coalescing-ok: the kept direct version: one thread per stratum reads its own run, gld efficiency 6.7 percent (nvprof); it is the faster kernel for short strata (236 us against 270 us for strata_xy_entropy_rows at 5000 strata, 668 us against 1.06 ms with an 8 x 8 table), the rows kernel takes long strata; the win axis is written at the top of the module
extern "C" __global__
void strata_xy_entropy(const unsigned char* __restrict__ xs, const unsigned char* __restrict__ ys, const int* __restrict__ starts,
                       const int S, const int n, const int Ky, const int cells, const {ACC}* __restrict__ ent,
                       {ACC}* __restrict__ h_out, int* __restrict__ k_out) {
    extern __shared__ int hist[];
    __shared__ {ACC} red_h[128];
    __shared__ int red_k[128];
    const int tid = threadIdx.x;
    const int p = blockIdx.y;
    const int stride = cells | 1;
    const int s = blockIdx.x * blockDim.x + tid;
    {ACC} h = 0;
    int k = 0;
    if (s < S) {
        int* mine = hist + tid * stride;
        for (int c = 0; c < cells; ++c) mine[c] = 0;
        const unsigned char* xp = xs + (long long)p * n;
        const int lo = starts[s], hi = starts[s + 1];
        for (int r = lo; r < hi; ++r) mine[(int)xp[r] * Ky + (int)ys[r]]++;
        for (int c = 0; c < cells; ++c) {
            const int v = mine[c];
            if (v > 0) { h += ent[v]; ++k; }
        }
    }
    red_h[tid] = h;
    red_k[tid] = k;
    __syncthreads();
    for (int t = blockDim.x >> 1; t > 0; t >>= 1) {
        if (tid < t) { red_h[tid] += red_h[tid + t]; red_k[tid] += red_k[tid + t]; }
        __syncthreads();
    }
    if (tid == 0) { h_out[(long long)p * gridDim.x + blockIdx.x] = red_h[0]; k_out[(long long)p * gridDim.x + blockIdx.x] = red_k[0]; }
}
"""
_SRC_ROWS = r"""
extern "C" __global__
void strata_xy_entropy_rows(const unsigned char* __restrict__ xs, const unsigned char* __restrict__ ys, const int* __restrict__ srank, const int* __restrict__ starts,
                            const int S, const int n, const int Ky, const int cells, const {ACC}* __restrict__ ent,
                            {ACC}* __restrict__ h_out, int* __restrict__ k_out) {
    extern __shared__ int hist[];
    __shared__ {ACC} red_h[128];
    __shared__ int red_k[128];
    const int tid = threadIdx.x;
    const int p = blockIdx.y;
    const int stride = cells | 1;
    const int s0 = blockIdx.x * blockDim.x;
    const int s1 = min(S, s0 + (int)blockDim.x);
    for (int u = tid; u < (int)blockDim.x * stride; u += blockDim.x) hist[u] = 0;
    __syncthreads();
    const unsigned char* xp = xs + (long long)p * n;
    const int r1 = starts[s1];
    for (int r = starts[s0] + tid; r < r1; r += blockDim.x) atomicAdd(&hist[(srank[r] - s0) * stride + (int)xp[r] * Ky + (int)ys[r]], 1);
    __syncthreads();
    {ACC} h = 0;
    int k = 0;
    if (s0 + tid < S) {
        const int* mine = hist + tid * stride;
        for (int c = 0; c < cells; ++c) {
            const int v = mine[c];
            if (v > 0) { h += ent[v]; ++k; }
        }
    }
    red_h[tid] = h;
    red_k[tid] = k;
    __syncthreads();
    for (int t = blockDim.x >> 1; t > 0; t >>= 1) {
        if (tid < t) { red_h[tid] += red_h[tid + t]; red_k[tid] += red_k[tid + t]; }
        __syncthreads();
    }
    if (tid == 0) { h_out[(long long)p * gridDim.x + blockIdx.x] = red_h[0]; k_out[(long long)p * gridDim.x + blockIdx.x] = red_k[0]; }
}
"""
_MODULES: dict = {}
_LOCK = threading.Lock()


def kernel(cp: Any, real: Any = None) -> Any:
    """``strata_xy_entropy`` compiled once per process (counts are integers and the sums are float64, so ``real`` does not change the module)."""
    with _LOCK:
        mod = _MODULES.get("acc64")
        if mod is None:
            mod = _MODULES["acc64"] = cp.RawModule(code=render(_SRC, acc=np.float64))  # Intentional float64: the entropy sums are the statistic compared with a threshold
        return mod.get_function("strata_xy_entropy")


def kernel_rows(cp: Any, real: Any = None) -> Any:
    """``strata_xy_entropy_rows`` (rows read coalesced, block-wide shared-memory counts) compiled once per process."""
    with _LOCK:
        mod = _MODULES.get("rows64")
        if mod is None:
            mod = _MODULES["rows64"] = cp.RawModule(code=render(_SRC_ROWS, acc=np.float64))  # Intentional float64: the entropy sums are the statistic compared with a threshold
        return mod.get_function("strata_xy_entropy_rows")


_ENT_TABLES: dict = {}
_ENT_LOCK = threading.Lock()
_ENT_TABLES_MAX = 4


def _entropy_table(cp: Any, n: int) -> Any:
    """``ent[v] = -(v / n) ln(v / n)`` for v = 0..n as a float64 device table, cached per ``n`` (a gate round asks for the same ``n`` hundreds of times).

    A table read replaces a double-precision ``log`` per occupied cell, which runs at 1/32 of the FP32 rate on consumer cards (15.1 ms against 1.4 ms at 36698 strata)."""
    with _ENT_LOCK:
        table = _ENT_TABLES.get(n)
        if table is None:
            inv_n = 1.0 / float(max(1, n))
            counts = cp.arange(n + 1, dtype=cp.float64)  # Intentional float64: entropy terms of the MI statistic
            table = cp.zeros(n + 1, dtype=cp.float64)
            table[1:] = -(counts[1:] * inv_n) * cp.log(counts[1:] * inv_n)
            if len(_ENT_TABLES) >= _ENT_TABLES_MAX:
                _ENT_TABLES.pop(next(iter(_ENT_TABLES)))  # evict-ok: a miss rebuilds the table, a pure function of n
            _ENT_TABLES[n] = table
        return table


def strata_joint_entropy(cp: Any, xs: Any, ys: Any, starts: Any, kx: int, ky: int, srank: Any = None) -> "tuple[Any, Any]":
    """``(h, k)`` device vectors of the ``(x, y, stratum)`` joint of every row of ``xs`` ``(P, n)`` uint8: entropy in nats and occupied-cell count, summed over the strata ``starts`` delimits.

    With ``srank`` (the int32 stratum rank of every row) the coalesced rows kernel runs; without it the one-thread-per-stratum kernel does."""
    P, n = int(xs.shape[0]), int(xs.shape[1])
    S = int(starts.shape[0]) - 1
    ent = _entropy_table(cp, n)
    cells = int(kx) * int(ky)
    blocks = (S + _THREADS - 1) // _THREADS
    # Intentional float64: the kernel writes double entropy partials
    h_part = cp.empty((P, blocks), dtype=cp.float64)
    k_part = cp.empty((P, blocks), dtype=cp.int32)
    smem = _THREADS * (cells | 1) * 4
    if srank is not None:
        kernel_rows(cp)((blocks, P), (_THREADS,), (xs, ys, srank, starts, np.int32(S), np.int32(n), np.int32(ky), np.int32(cells), ent, h_part, k_part), shared_mem=smem)  # launch-ok: 128 threads each own one stratum's histogram row in the block's table, and the tree reduction is sized for 128 threads
    else:
        kernel(cp)((blocks, P), (_THREADS,), (xs, ys, starts, np.int32(S), np.int32(n), np.int32(ky), np.int32(cells), ent, h_part, k_part), shared_mem=smem)  # launch-ok: each of the 128 threads owns one stratum's private histogram, and the tree reduction is sized for 128 threads
    return h_part.sum(axis=1), k_part.sum(axis=1, dtype=cp.int64)


def strata_null_cmi(cp: Any, shuffled_sorted: Any, y_sorted: Any, z_rank: Any, dz: Any, dy: Any, kx: int, order_d: Any, x_sorted: Any, variant: Optional[str] = None) -> Optional[Any]:
    """CMI(x_perm; y | z) of every shuffled column, as a device ``(P,)`` vector, or ``None`` when this route does not apply.

    ``shuffled_sorted`` ``(P, n)`` are within-stratum shuffles of ``x_sorted`` in z-sorted order, ``y_sorted`` / ``z_rank`` the target and dense stratum rank in that order, ``dz`` / ``dy`` the support and
    target in the original order (the fixed ``H(z)`` / ``H(y, z)`` terms). Applies when the stratum histogram is small, there are enough strata to fill the device, and codes fit a byte.
    """
    from ._fe_batched_mi import _cmi_assemble, joint_entropy_gpu

    n = int(shuffled_sorted.shape[1])
    ky = int(dy.max()) + 1 if dy.size else 1
    S = int(z_rank[-1]) + 1
    if kx * ky > MAX_CELLS or kx > 255 or ky > 255 or S < MIN_STRATA:
        return None
    try:
        srank = z_rank.astype(cp.int32)
        # run starts from the stratum sizes: no host sync (the number of strata is known)
        starts = cp.zeros(S + 1, dtype=cp.int32)
        starts[1:] = cp.cumsum(cp.bincount(srank, minlength=S)).astype(cp.int32)
        inv_n = 1.0 / float(max(1, n))
        xs8 = shuffled_sorted.astype(cp.uint8)
        ys8 = y_sorted.astype(cp.uint8)
        use_rows = (n >= LONG_STRATUM_ROWS * S) if variant is None else (variant == "rows")
        rk = srank if use_rows else None
        h_xyz, k_xyz = strata_joint_entropy(cp, xs8, ys8, starts, kx, ky, rk)
        h_xz1, k_xz1 = strata_joint_entropy(cp, x_sorted.astype(cp.uint8)[None, :], cp.zeros(n, dtype=cp.uint8), starts, kx, 1, rk)
        kz = int(dz.max()) + 1
        h_z, k_z = joint_entropy_gpu([dz], [kz], inv_n)
        yz = dy * kz + dz
        h_yz, k_yz = joint_entropy_gpu([yz], [int(yz.max()) + 1], inv_n)
        P = int(shuffled_sorted.shape[0])
        h_xz = cp.broadcast_to(h_xz1, (P,))  # the (x, z) joint is the same for every shuffle; stays on the device
        k_xz = cp.broadcast_to(k_xz1, (P,))
        return _cmi_assemble(h_xz, h_xyz, float(h_yz - h_z), k_xyz, k_xz, float(k_z - k_yz), 1.0 / (2.0 * float(max(1, n))))
    except Exception as exc:
        log_throttle(
            logger, "cmi_null_strata_gpu_failed", logging.WARNING,
            "per-stratum null CMI failed (%s: %s); the dense route runs for this call", type(exc).__name__, exc, exc_info=True,
        )
        return None
