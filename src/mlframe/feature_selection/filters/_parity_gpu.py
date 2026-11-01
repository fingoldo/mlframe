"""Device scan of the floor-parity period grid for many columns at once (twin of ``_parity_fe._parity_mi_grid_njit``).

Win axis: the host scan counts a ``2 x n_classes`` table per (column, period) with one pass over the rows each; at 100 columns x 120 periods x 100000 rows that is 12 000 passes (2.6 s on 8 cores). Here one block owns a
column and ``PB`` consecutive periods: its threads stride over the column ONCE, load each value and class once and update ``PB`` periods from registers. The table needs no 2-D histogram: for a period the rows of
parity 1 per class are counted (``ones``), the parity-0 counts are the class totals minus them, so each thread keeps ``PB * NC`` plain counters. A block reduces them into shared memory and the MI of its periods is
finished in the same kernel, with the host's formula and term order.

Values stay double precision (a float32 timestamp cannot resolve a small period; ``floor(v * inv)`` is taken in double as on the host); the class codes are one byte and only ``n_classes <= NC`` is handled (the host
scan stays for more classes). Columns with a non-finite value keep the host path (they are scanned on their own finite rows there).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

PB = 8  # periods per block
NC = 4  # largest class count the kernel handles
THREADS = 256
COLUMN_BLOCK = 32  # columns per launch: bounds the double copy of the block on the device (32 x n x 8 bytes)

_SRC = r"""
#define PB 8
#define NC 4
extern "C" __global__ void parity_mi_grid(const double* __restrict__ V, const unsigned char* __restrict__ Y, const double* __restrict__ INV,
                                          const long long* __restrict__ cls_tot, const long long n, const int G, const int nc, double* __restrict__ out) {
    const int col = blockIdx.x;
    const int p0 = blockIdx.y * PB;
    const double* v = V + (long long)col * n;
    double inv[PB];
    for (int p = 0; p < PB; ++p) inv[p] = (p0 + p < G) ? INV[(long long)col * G + p0 + p] : 1.0;
    unsigned int ones[PB][NC];
    for (int p = 0; p < PB; ++p) for (int k = 0; k < NC; ++k) ones[p][k] = 0u;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const double x = v[i];
        const int y = (int)Y[i];
#pragma unroll
        for (int p = 0; p < PB; ++p) {
            const unsigned int c = (unsigned int)((long long)floor(x * inv[p]) & 1LL);
#pragma unroll
            for (int k = 0; k < NC; ++k) ones[p][k] += c & (unsigned int)(y == k);
        }
    }
    __shared__ unsigned long long hist[PB * NC];
    for (int s = threadIdx.x; s < PB * NC; s += blockDim.x) hist[s] = 0ull;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < PB; ++p)
#pragma unroll
        for (int k = 0; k < NC; ++k) atomicAdd(&hist[p * NC + k], (unsigned long long)ones[p][k]);
    __syncthreads();
    if (threadIdx.x < PB && p0 + (int)threadIdx.x < G) {
        const int p = threadIdx.x;
        const double nn = (double)n;
        double mi = 0.0;
        long long row[2] = {0, 0};
        for (int k = 0; k < nc; ++k) { const long long o = (long long)hist[p * NC + k]; row[1] += o; row[0] += cls_tot[k] - o; }
        for (int c = 0; c < 2; ++c) {
            if (row[c] == 0) continue;
            for (int k = 0; k < nc; ++k) {
                const long long o = (long long)hist[p * NC + k];
                const long long nij = c == 1 ? o : cls_tot[k] - o;
                if (nij == 0) continue;
                mi += ((double)nij / nn) * log(((double)nij * nn) / ((double)row[c] * (double)cls_tot[k]));
            }
        }
        out[(long long)col * G + p0 + p] = mi;
    }
}
"""

_MODULE = RawModuleCache(_SRC)


def parity_mi_grids_gpu(columns: "list[np.ndarray]", y_codes: np.ndarray, n_classes: int, grids: "list[np.ndarray]") -> "list[np.ndarray] | None":
    """MI of ``floor(v / p) mod 2`` against ``y_codes`` for every period ``p`` of each column's grid, one array per column (or ``None`` when the device path does not apply: more than ``NC`` classes, no cupy).

    ``columns`` are all-finite double-precision vectors of one length; ``y_codes`` is the dense class code of every row (the same rows for every column); ``grids[j]`` is the period grid of column ``j``.
    """
    if not columns or n_classes > NC or n_classes < 1:
        return None
    try:
        import cupy as cp
    except Exception as exc:
        logger.debug("device parity scan unavailable (%s)", exc)
        return None
    n = int(columns[0].shape[0])
    G = int(grids[0].shape[0])
    kernel = _MODULE.get(cp).get_function("parity_mi_grid")
    y_dev = cp.asarray(np.ascontiguousarray(y_codes, dtype=np.uint8))
    cls_tot = cp.asarray(np.bincount(np.asarray(y_codes, dtype=np.int64), minlength=NC).astype(np.int64))
    results: "list[np.ndarray]" = []
    for c0 in range(0, len(columns), COLUMN_BLOCK):
        block = columns[c0 : c0 + COLUMN_BLOCK]
        v_dev = cp.asarray(np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.float64) for c in block])))  # Intentional float64: floor(v / period) needs double precision on large values
        inv_dev = cp.asarray(np.ascontiguousarray(np.stack([1.0 / np.asarray(g, dtype=np.float64) for g in grids[c0 : c0 + COLUMN_BLOCK]])))  # Intentional float64: the host multiplies by the double reciprocal of the period
        out = cp.zeros((len(block), G), dtype=cp.float64)  # Intentional float64: MI statistic vector compared with the host's
        kernel((len(block), (G + PB - 1) // PB), (THREADS,), (v_dev, y_dev, inv_dev, cls_tot, np.int64(n), np.int32(G), np.int32(n_classes), out))
        results.extend(list(cp.asnumpy(out)))
    return results


def use_device_parity(n_rows: int) -> bool:
    """Whether a parity scan over ``n_rows`` rows runs on the device: ``MLFRAME_PARITY_DEVICE=0|1`` forces a path, else from ``PARITY_DEVICE_MIN_ROWS`` rows (the measured crossover, see the profile note)."""
    forced = os.environ.get("MLFRAME_PARITY_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return int(n_rows) >= PARITY_DEVICE_MIN_ROWS


PARITY_DEVICE_MIN_ROWS = 5_000

__all__ = ["parity_mi_grids_gpu", "use_device_parity"]
