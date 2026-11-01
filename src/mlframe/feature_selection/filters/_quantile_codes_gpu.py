"""BENCH-ATTEMPT-REJECTED (2026-11-01, GTX 1050 Ti, codes bit-equal to the host kernel in float32 and double on continuous, tied, few-level, constant and heavy-tailed columns): the device twin is NOT faster than the host kernel.
100000 rows x 1944 columns: device 1963 ms (float32) / 2491 ms (double) against host 2405 / 2572 ms; 256 columns: 285 against 260 ms. Phases for 256 columns: host-to-device 27 ms, batched sort 191 ms (0.75 ms per column),
distinct count 34 ms, device-to-host of the codes 12 ms. The sort of every column is what the device pays for, while the host kernel only selects the ~22 order statistics the edges read (about 1 ms per column on 8 cores).
Retry condition: a device order-statistic selection (radix select by bit-pattern histograms, 3 passes) instead of the full sort, or a card with a much faster sort. Not wired into ``_family_codes``; ``MLFRAME_QUANTILE_CODES_DEVICE=1`` forces the device path in
``quantile_codes_block_auto``, the default is the host kernel. Tests: ``tests/feature_selection/fe/test_quantile_codes_gpu.py`` keep the equivalence.

Device twin of ``_numba_utils.quantile_codes_block``: int16 quantile-bin codes for a block of finite columns.

Win axis: the host kernel selects the ~22 order statistics each column's edges read and then binary-searches every value (about 1 ms per 100000-row column, 2.1 s for the 1944 columns of the family max-T gate). Here a block of columns is sorted
once with the device batched sort, the edges are read off the sorted rows with numpy's linear-interpolation arithmetic in double precision (``_edge_value``), adjacent equal edges are dropped, and one thread per value counts the kept edges
that are <= the value (the code is that count minus one, clipped to ``[0, kept - 2]``), which is the host's upper-bound binary search. A column with at most ``n_bins`` distinct values takes exact level codes on the host, as there.

The edges are computed in float64 from the values at the input width, the comparison is ``edge <= (double) x``: the same numbers as the host kernel, so the codes are equal.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

THREADS = 256
#: Fraction of the memory the pool can still hand out that one launch may use, and the device bytes held per column row (input, sorted copy, sort workspace, codes).
_BUDGET_FRACTION = 0.25
_BYTES_PER_COLUMN_ROW = 20
_SRC = r"""
extern "C" __global__ void quantile_codes_{TAG}(const {REAL}* __restrict__ V, const double* __restrict__ EDGES, const unsigned char* __restrict__ KEEP,
                                               const int* __restrict__ HI_CODE, const long long n, const int nb1, short* __restrict__ codes) {
    const int col = blockIdx.y;
    const double* e = EDGES + (long long)col * nb1;
    const unsigned char* kp = KEEP + (long long)col * nb1;
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


def quantile_codes_block_gpu(v: np.ndarray, n_bins: int, codes: np.ndarray, nx: np.ndarray) -> bool:
    """Fill ``codes`` ``(columns, n)`` int16 and ``nx`` as ``quantile_codes_block`` does for the finite block ``v``; ``False`` when the device path does not apply (the caller then runs the host kernel on the whole block)."""
    k, n = int(v.shape[0]), int(v.shape[1])
    if k == 0 or n < 2 or v.dtype not in (np.float32, np.float64):
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
    block = _column_block(cp, n)
    for c0 in range(0, k, block):
        c1 = min(k, c0 + block)
        v_dev = cp.asarray(np.ascontiguousarray(v[c0:c1]))
        s = cp.sort(v_dev, axis=1)
        distinct = 1 + (s[:, 1:] != s[:, :-1]).sum(axis=1)
        few = np.flatnonzero(cp.asnumpy(distinct) <= int(n_bins))
        a = s[:, i0_dev].astype(cp.float64)
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


def quantile_codes_block_auto(v: np.ndarray, n_bins: int, codes: np.ndarray, nx: np.ndarray) -> None:
    """``quantile_codes_block`` on the device when the block is large enough and the device is usable, else the host kernel; any device fault returns to the host kernel."""
    from ._numba_utils import quantile_codes_block

    forced = os.environ.get("MLFRAME_QUANTILE_CODES_DEVICE", "").strip()
    use_device = forced == "1"  # rejected as a default: no faster than the host kernel (see the module note)
    if use_device:
        try:
            if quantile_codes_block_gpu(v, n_bins, codes, nx):
                return
        except Exception as exc:
            logger.warning("device quantile codes failed (%s: %s); using the host kernel", type(exc).__name__, exc)
    quantile_codes_block(v, n_bins, codes, nx)


__all__ = ["quantile_codes_block_gpu", "quantile_codes_block_auto"]
