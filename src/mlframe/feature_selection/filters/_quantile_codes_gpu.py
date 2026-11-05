"""BENCH-ATTEMPT-REJECTED for the SORT-based variant (``method="sort"``, kept; also the double-precision path) (2026-11-01, GTX 1050 Ti, codes bit-equal to the host kernel): 100000 rows x 1944 columns device 1963 ms (float32) / 2491 ms (double) against
host 2405 / 2572 ms; 256 columns 285 against 260 ms. Phases for 256 columns: host-to-device 27 ms, batched sort 191 ms (0.75 ms per column), distinct count 34 ms, device-to-host of the codes 12 ms: the device paid for sorting every column, while the host kernel only selects the
~22 order statistics the edges read.

DEFAULT for float32 blocks of at least ``DEVICE_MIN_ELEMENTS``: the order statistics are SELECTED on the device too (``_order_statistics_f32``: three radix passes of 11 + 11 + 10 bits over the order-preserving integer key of the float, per-target histograms, no sort).
100000 rows x 1944 columns: 871 ms against 2085 ms on the host (2.4x), 256 columns 130 against 278 ms, codes equal. Remaining cost at 256 columns: transfers about 40 ms, order statistics 37 ms (histogram kernels about 15 ms), code kernel 11 ms.
``MLFRAME_QUANTILE_CODES_DEVICE=0|1`` forces a path.

Device twin of ``_numba_utils.quantile_codes_block``: int16 quantile-bin codes for a block of finite columns.

Win axis: the host kernel selects the ~22 order statistics each column's edges read and then binary-searches every value (about 1 ms per 100000-row column, 2.1 s for the 1944 columns of the family max-T gate). Here a block of columns is sorted
once with the device batched sort, the edges are read off the sorted rows with numpy's linear-interpolation arithmetic in double precision (``_edge_value``), adjacent equal edges are dropped, and one thread per value counts the kept edges
that are <= the value (the code is that count minus one, clipped to ``[0, kept - 2]``), which is the host's upper-bound binary search. A column with at most ``n_bins`` distinct values takes exact level codes on the host, as there.

The edges are computed in double precision from the values at the input width, the comparison is ``edge <= (double) x``: the same numbers as the host kernel, so the codes are equal.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from numba import njit, prange

from ._numba_utils_cross_module import _few_distinct_levels
from ._raw_module_cache import RawModuleCache
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # noqa: F401  (thrust aborts the process on a short pool)

logger = logging.getLogger(__name__)

THREADS = 256
#: Elements (columns x rows) from which the device radix path is the default choice for float32 blocks.
STRICT_DEVICE_MIN_ELEMENTS = 20_000_000  # the strict-resident path keeps its bulk device-to-host traffic out of this coding unless the block is large (tests/feature_selection/gpu/test_cmi_residency_traffic.py)
DEVICE_MIN_ELEMENTS = 1_500_000  # measured crossover on the GTX 1050 Ti (8 host threads): the device wins from about 1M elements at 20000 to 100000 rows (audits note), 20M was the first conservative bound
#: Fraction of the memory the pool can still hand out that one launch may use, and the device bytes held per column row (input, sorted copy, sort workspace, codes).
_BUDGET_FRACTION = 0.25
_BYTES_PER_COLUMN_ROW = 20
_SRC = r"""
extern "C" __global__ void quantile_codes_{TAG}(const {REAL}* __restrict__ V, const double* __restrict__ EDGES, const unsigned char* __restrict__ KEEP,
                                               const int* __restrict__ HI_CODE, const long long n, const int nb1, short* __restrict__ codes) {
    const int col = blockIdx.y;
    __shared__ double se[256];          // this column's edges and keep flags, read for every value: one load per block instead of a global read per value
    __shared__ unsigned char sk[256];
    const bool cached = nb1 <= 256;
    if (cached) for (int b = threadIdx.x; b < nb1; b += blockDim.x) { se[b] = EDGES[(long long)col * nb1 + b]; sk[b] = KEEP[(long long)col * nb1 + b]; }
    __syncthreads();
    const double* e = cached ? se : EDGES + (long long)col * nb1;
    const unsigned char* kp = cached ? sk : KEEP + (long long)col * nb1;
    const int hi_code = HI_CODE[col];
    const {REAL}* v = V + (long long)col * n;
    short* out = codes + (long long)col * n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double x = (double)v[i];
        int cnt = 0;
        for (int b = 0; b < nb1; ++b) cnt += (kp[b] && e[b] <= x) ? 1 : 0;
        int c = cnt - 1;
        c = c < 0 ? 0 : (c > hi_code ? hi_code : c);
        out[i] = (short)c;
    }
}
"""


_RADIX_SRC = r"""
__device__ __forceinline__ unsigned int to_key(float x) {
    const unsigned int u = __float_as_uint(x);
    return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
extern "C" __global__ void radix_hist1(const float* __restrict__ V, const long long n, unsigned int* __restrict__ H1) {
    __shared__ unsigned int sh[2048];
    for (int b = threadIdx.x; b < 2048; b += blockDim.x) sh[b] = 0u;
    __syncthreads();
    const int col = blockIdx.y;
    const float* v = V + (long long)col * n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) atomicAdd(&sh[to_key(v[i]) >> 21], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < 2048; b += blockDim.x) if (sh[b]) atomicAdd(&H1[(long long)col * 2048 + b], sh[b]);
}
extern "C" __global__ void radix_hist2(const float* __restrict__ V, const long long n, const int* __restrict__ BIN1, const int T, unsigned int* __restrict__ H2) {
    __shared__ unsigned char is_target[2048];
    const int col = blockIdx.y;
    const float* v = V + (long long)col * n;
    const int* b1 = BIN1 + (long long)col * T;
    for (int b = threadIdx.x; b < 2048; b += blockDim.x) is_target[b] = 0;
    __syncthreads();
    for (int t = threadIdx.x; t < T; t += blockDim.x) is_target[b1[t]] = 1;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const unsigned int key = to_key(v[i]);
        const int top = (int)(key >> 21);
        if (!is_target[top]) continue;
        for (int t = 0; t < T; ++t)
            if (top == b1[t]) atomicAdd(&H2[((long long)col * T + t) * 2048 + ((key >> 10) & 2047u)], 1u);
    }
}
extern "C" __global__ void radix_hist3(const float* __restrict__ V, const long long n, const int* __restrict__ BIN1, const int* __restrict__ BIN2, const int T, unsigned int* __restrict__ H3) {
    __shared__ unsigned char is_target[2048];
    const int col = blockIdx.y;
    const float* v = V + (long long)col * n;
    const int* b1 = BIN1 + (long long)col * T;
    const int* b2 = BIN2 + (long long)col * T;
    for (int b = threadIdx.x; b < 2048; b += blockDim.x) is_target[b] = 0;
    __syncthreads();
    for (int t = threadIdx.x; t < T; t += blockDim.x) is_target[b1[t]] = 1;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const unsigned int key = to_key(v[i]);
        const int top = (int)(key >> 21);
        if (!is_target[top]) continue;
        const int mid = (int)((key >> 10) & 2047u);
        for (int t = 0; t < T; ++t)
            if (top == b1[t] && mid == b2[t]) atomicAdd(&H3[((long long)col * T + t) * 1024 + (key & 1023u)], 1u);
    }
}
"""

_RADIX = RawModuleCache(_RADIX_SRC)


def _order_statistics_f32(cp, v_dev, ranks: np.ndarray):
    """``(columns, len(ranks))`` float32: the value of rank ``ranks[t]`` (0-based, ascending, distinct) of every row of ``v_dev`` ``(columns, n)``, by three radix passes (11 + 11 + 10 key bits) instead of a sort."""
    cb, n = int(v_dev.shape[0]), int(v_dev.shape[1])
    T = int(ranks.size)
    mod = _RADIX.get(cp)
    w = cp.asarray(ranks.astype(np.int64))
    grid = (min(64, (n + THREADS - 1) // THREADS), cb)

    h1 = cp.zeros((cb, 2048), dtype=cp.uint32)
    mod.get_function("radix_hist1")(grid, (THREADS,), (v_dev, np.int64(n), h1))
    cs1 = cp.cumsum(h1, axis=1, dtype=cp.int64)
    bin1 = (cs1[:, None, :] <= w[None, :, None]).sum(axis=2).astype(cp.int32)  # (cb, T): first bin whose running count passes the rank
    before1 = cp.where(bin1 > 0, cp.take_along_axis(cs1, cp.maximum(bin1 - 1, 0).astype(cp.int64), axis=1), 0)
    r1 = (w[None, :] - before1).astype(cp.int32)

    h2 = cp.zeros((cb, T, 2048), dtype=cp.uint32)
    mod.get_function("radix_hist2")(grid, (THREADS,), (v_dev, np.int64(n), cp.ascontiguousarray(bin1), np.int32(T), h2))
    cs2 = cp.cumsum(h2, axis=2, dtype=cp.int32)
    bin2 = (cs2 <= r1[:, :, None]).sum(axis=2).astype(cp.int32)
    before2 = cp.where(bin2 > 0, cp.take_along_axis(cs2, cp.maximum(bin2 - 1, 0).astype(cp.int64)[:, :, None], axis=2)[:, :, 0], 0)
    r2 = (r1 - before2).astype(cp.int32)

    h3 = cp.zeros((cb, T, 1024), dtype=cp.uint32)
    mod.get_function("radix_hist3")(grid, (THREADS,), (v_dev, np.int64(n), cp.ascontiguousarray(bin1), cp.ascontiguousarray(bin2), np.int32(T), h3))
    cs3 = cp.cumsum(h3, axis=2, dtype=cp.int32)
    bin3 = (cs3 <= r2[:, :, None]).sum(axis=2).astype(cp.int32)

    key = (bin1.astype(cp.uint32) << 21) | (bin2.astype(cp.uint32) << 10) | bin3.astype(cp.uint32)
    bits = cp.where((key >> 31) != 0, key ^ cp.uint32(0x80000000), ~key)
    return bits.view(cp.float32)


@njit(cache=True, parallel=True)
def _few_levels_mask(v: np.ndarray, n_bins: int) -> np.ndarray:
    """True for the rows of ``v`` ``(columns, n)`` with at most ``n_bins`` distinct values (the early-exit scan ends within the first few values of a continuous row)."""
    out = np.zeros(v.shape[0], dtype=np.bool_)
    for j in prange(v.shape[0]):
        out[j] = _few_distinct_levels(v[j], n_bins).shape[0] > 0
    return out


_MODULES = {
    "float": RawModuleCache(_SRC.replace("{REAL}", "float").replace("{TAG}", "f32")),
    "double": RawModuleCache(_SRC.replace("{REAL}", "double").replace("{TAG}", "f64")),
}


def _edge_positions(n: int, n_bins: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Lower index, upper index and gamma of every edge quantile ``b / n_bins`` (the last is 1.0), as ``_edge_value`` computes them."""
    step = 1.0 / n_bins
    i0 = np.empty(n_bins + 1, dtype=np.int64)
    i1 = np.empty(n_bins + 1, dtype=np.int64)
    gamma = np.empty(n_bins + 1, dtype=np.float64)  # Intentional float64: numpy's interpolation weight
    for b in range(n_bins + 1):
        q = 1.0 if b == n_bins else b * step
        virtual = (n - 1) * q
        if virtual >= n - 1:
            lo = hi = n - 1
        elif virtual < 0:
            lo = hi = 0
        else:
            lo = int(np.floor(virtual))
            hi = lo + 1
        i0[b], i1[b], gamma[b] = lo, hi, virtual - lo
    return i0, i1, gamma


def _column_block(cp, n_rows: int) -> int:
    """Columns per launch within ``_BUDGET_FRACTION`` of what the memory pool can still hand out (at least 4)."""
    try:
        pool = cp.get_default_memory_pool()
        free_device = int(cp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        limit = int(pool.get_limit())
        room = free_device if limit <= 0 else min(free_device, max(0, limit - int(pool.used_bytes())))
    except Exception as exc:
        logger.debug("device quantile codes: free memory unknown (%s); small blocks", exc)
        return 16
    return int(max(4, (_BUDGET_FRACTION * room) // (_BYTES_PER_COLUMN_ROW * max(1, n_rows))))


def quantile_codes_block_gpu(v: np.ndarray, n_bins: int, codes: np.ndarray, nx: np.ndarray, method: str = "radix") -> bool:
    """Fill ``codes`` ``(columns, n)`` int16 and ``nx`` as ``quantile_codes_block`` does for the finite block ``v``; ``False`` when the device path does not apply (the caller then runs the host kernel on the whole block)."""
    k, n = int(v.shape[0]), int(v.shape[1])
    if k == 0 or n < 2 or v.dtype not in (np.float32, np.float64):  # Intentional float64: the dtypes the block coder accepts
        return False
    try:
        import cupy as cp
    except Exception as exc:
        logger.debug("device quantile codes unavailable (%s)", exc)
        return False
    from ._numba_utils import quantile_codes_block

    real = "float" if v.dtype == np.float32 else "double"
    kernel = _MODULES[real].get(cp).get_function("quantile_codes_f32" if real == "float" else "quantile_codes_f64")
    i0, i1, gamma = _edge_positions(n, int(n_bins))
    i0_dev, i1_dev, gamma_dev = cp.asarray(i0), cp.asarray(i1), cp.asarray(gamma)
    nb1 = int(n_bins) + 1
    w_unique = np.unique(np.concatenate([i0, i1]))
    block = min(_column_block(cp, n), 256)
    for c0 in range(0, k, block):
        c1 = min(k, c0 + block)
        v_dev = cp.asarray(np.ascontiguousarray(v[c0:c1]))
        if method == "radix" and v.dtype == np.float32:
            few = np.flatnonzero(_few_levels_mask(np.ascontiguousarray(v[c0:c1]), int(n_bins)))
            vals = _order_statistics_f32(cp, v_dev, w_unique)  # (cb, T) values at the ranks the edges read
            # Intentional float64: the edges are computed in numpy's double-precision arithmetic, as the host kernel does
            a = vals[:, cp.asarray(np.searchsorted(w_unique, i0))].astype(cp.float64)
            # Intentional float64: same double-precision edge arithmetic as the lower order statistic
            b = vals[:, cp.asarray(np.searchsorted(w_unique, i1))].astype(cp.float64)
        else:
            s = device_sort(v_dev, axis=1)
            distinct = 1 + (s[:, 1:] != s[:, :-1]).sum(axis=1)
            few = np.flatnonzero(cp.asnumpy(distinct) <= int(n_bins))
            # Intentional float64: the edges are computed in numpy's double-precision arithmetic, as the host kernel does
            a = s[:, i0_dev].astype(cp.float64)
            # Intentional float64: same double-precision edge arithmetic as the lower order statistic
            b = s[:, i1_dev].astype(cp.float64)
        diff = b - a
        edges = a + diff * gamma_dev
        edges = cp.where(gamma_dev >= 0.5, b - diff * (1.0 - gamma_dev), edges)
        keep = cp.ones(edges.shape, dtype=cp.uint8)
        keep[:, 1:] = (edges[:, 1:] != edges[:, :-1]).astype(cp.uint8)
        hi_code = cp.maximum(keep.sum(axis=1).astype(cp.int32) - 2, 0).astype(cp.int32)
        out = cp.empty((c1 - c0, n), dtype=cp.int16)
        grid = (min(1024, (n + THREADS - 1) // THREADS), c1 - c0)
        kernel(grid, (THREADS,), (v_dev, cp.ascontiguousarray(edges), keep, hi_code, np.int64(n), np.int32(nb1), out))
        codes[c0:c1] = cp.asnumpy(out)
        nx[c0:c1] = cp.asnumpy(out.max(axis=1)).astype(np.int64) + 1
        if few.size:  # few distinct values: exact level codes, as the host kernel gives them
            rows = c0 + few
            sub_codes = np.empty((few.size, n), dtype=np.int16)
            sub_nx = np.empty(few.size, dtype=np.int64)
            quantile_codes_block(np.ascontiguousarray(v[rows]), int(n_bins), sub_codes, sub_nx)
            codes[rows] = sub_codes
            nx[rows] = sub_nx
    return True


def quantile_codes_resident_f32(cp, v_dev, n_bins: int, codes: np.ndarray, nx: np.ndarray) -> None:
    """``quantile_codes_block_gpu`` of a RESIDENT finite float32 block ``v_dev`` ``(columns, n)`` (C-contiguous): same edges, same code kernel, codes copied into the
    host ``codes`` / ``nx``. Only the rows that may have at most ``n_bins`` distinct values are copied to the host: a row whose order statistics already hold more
    than ``n_bins`` distinct values has more levels, the others are checked exactly by the host kernel and, when few-valued, take its exact level codes."""
    from ._numba_utils import quantile_codes_block

    k, n = int(v_dev.shape[0]), int(v_dev.shape[1])
    kernel = _MODULES["float"].get(cp).get_function("quantile_codes_f32")
    i0, i1, gamma = _edge_positions(n, int(n_bins))
    gamma_dev = cp.asarray(gamma)
    nb1 = int(n_bins) + 1
    w_unique = np.unique(np.concatenate([i0, i1]))
    lo_idx, hi_idx = cp.asarray(np.searchsorted(w_unique, i0)), cp.asarray(np.searchsorted(w_unique, i1))
    block = min(_column_block(cp, n), 256)
    for c0 in range(0, k, block):
        c1 = min(k, c0 + block)
        vb = v_dev[c0:c1]
        vals = _order_statistics_f32(cp, vb, w_unique)
        srt = device_sort(vals, axis=1)
        maybe_few = np.flatnonzero(cp.asnumpy(1 + (srt[:, 1:] != srt[:, :-1]).sum(axis=1)) <= int(n_bins))
        # Intentional float64: the edges are computed in numpy's double-precision arithmetic, as the host kernel does
        a = vals[:, lo_idx].astype(cp.float64)
        # Intentional float64: same double-precision edge arithmetic as the lower order statistic
        b = vals[:, hi_idx].astype(cp.float64)
        diff = b - a
        edges = cp.where(gamma_dev >= 0.5, b - diff * (1.0 - gamma_dev), a + diff * gamma_dev)
        keep = cp.ones(edges.shape, dtype=cp.uint8)
        keep[:, 1:] = (edges[:, 1:] != edges[:, :-1]).astype(cp.uint8)
        hi_code = cp.maximum(keep.sum(axis=1).astype(cp.int32) - 2, 0).astype(cp.int32)
        out = cp.empty((c1 - c0, n), dtype=cp.int16)
        kernel((min(1024, (n + THREADS - 1) // THREADS), c1 - c0), (THREADS,), (vb, cp.ascontiguousarray(edges), keep, hi_code, np.int64(n), np.int32(nb1), out))
        codes[c0:c1] = cp.asnumpy(out)
        nx[c0:c1] = cp.asnumpy(out.max(axis=1)).astype(np.int64) + 1
        if maybe_few.size:
            host_rows = cp.asnumpy(vb[cp.asarray(maybe_few)])
            few = np.flatnonzero(_few_levels_mask(host_rows, int(n_bins)))
            if few.size:  # few distinct values: exact level codes, as the host kernel gives them
                sub_codes = np.empty((few.size, n), dtype=np.int16)
                sub_nx = np.empty(few.size, dtype=np.int64)
                quantile_codes_block(np.ascontiguousarray(host_rows[few]), int(n_bins), sub_codes, sub_nx)
                codes[c0 + maybe_few[few]] = sub_codes
                nx[c0 + maybe_few[few]] = sub_nx


def _device_min_elements() -> int:
    """Elements from which the device codes are the default: ``DEVICE_MIN_ELEMENTS``, or ``STRICT_DEVICE_MIN_ELEMENTS`` under the strict-resident path, whose contract forbids a bulk device-to-host copy that a host consumer of the codes needs (the codes are copied back here)."""
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled

        if fe_gpu_strict_resident_enabled():
            return STRICT_DEVICE_MIN_ELEMENTS
    except Exception as exc:
        logger.debug("strict-resident probe failed (%s); using the default bound", exc)
    return DEVICE_MIN_ELEMENTS


def quantile_codes_block_auto(v: np.ndarray, n_bins: int, codes: np.ndarray, nx: np.ndarray, host_kernel=None) -> None:
    """``quantile_codes_block`` on the device when the block is a large float32 one and the device is usable, else ``host_kernel`` (default: the njit kernel); any device fault returns to the host kernel."""
    if host_kernel is None:
        from ._numba_utils import quantile_codes_block as host_kernel

    forced = os.environ.get("MLFRAME_QUANTILE_CODES_DEVICE", "").strip()
    use_device = (forced == "1") if forced in ("0", "1") else (v.dtype == np.float32 and int(v.shape[0]) * int(v.shape[1]) >= _device_min_elements())
    if use_device:
        try:
            if quantile_codes_block_gpu(v, n_bins, codes, nx):
                return
        except Exception as exc:
            logger.warning("device quantile codes failed (%s: %s); using the host kernel", type(exc).__name__, exc)
    host_kernel(v, n_bins, codes, nx)


__all__ = ["quantile_codes_block_gpu", "quantile_codes_block_auto", "quantile_codes_resident_f32"]
