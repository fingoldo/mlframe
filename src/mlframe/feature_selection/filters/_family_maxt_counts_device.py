"""Family null maxima on the device from fused contingency tables: the replacement for the thread-per-pair local-memory histogram kernel where the bit-set kernels do not apply.

``_family_maxt_cuda_bits`` counts bin x class cells as popcounts of bit sets, but its shared-memory kernel holds the class bit sets of one permutation, ``(classes - 1) * words * 8``
bytes: 20 classes at 30000 rows are 71 KB against the 48 KB block limit, so those families fell to the thread-per-(permutation, column) numba kernel of ``_family_maxt_cuda``, whose
every thread walks all ``n`` rows into a local-memory table (12 to 13 s for 224 permutations of 1200 columns at 30000 rows: 21 percent of the GPU time of the warm profiling fit, in
8 launches). Here the tables come from ``perm_joint_counts`` (a block per (column, permutation), the table in shared memory, 33 G updates/s measured) and one small kernel turns every table into its
plug-in MI with exactly the arithmetic and the summation order of the numba kernel and of the CPU kernel: ``mi += (nab / n) * log((nab * n) / (pxa * pyb))`` over ascending bins, so the maxima are
bit-identical to the numba kernel (the CPU kernel agrees to rounding, 1e-16 relative; the integer marginals ``pxa`` and ``pyb`` are exact whatever the order they are summed in; the column sums are computed once per table instead of once per cell).
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np
from pyutilz.performance.gpu_hw_launch import block_size_for

from ._permutation_null_joint_kernel import perm_joint_counts

MAX_CLASSES = 64  # the per-table class sums live in a register array of this size
_TABLE_BUDGET_BYTES = 96 * 1024 * 1024  # int32 tables of one (column block x permutation block) tile
_PERM_BLOCK = 32

_SRC = r"""
extern "C" __global__
void family_mi_from_tables(const int* __restrict__ counts, const long long* __restrict__ n_x, const int n_y, const int max_jc,
                           const int kc, const int pbc, const int k0, const int p0, const int K, const long long n,
                           double* __restrict__ out) {
    // one thread per table (row = column * pbc + permutation); the table is nx x n_y ints, row a = bins of x, column bb = classes
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (long long)kc * pbc) return;
    const int kk = (int)(row / pbc);
    const int pp = (int)(row - (long long)kk * pbc);
    const int k = k0 + kk;
    const int nx = (int)n_x[k];
    const int* t = counts + row * (long long)max_jc;
    long long pyb[MAXC];
    for (int bb = 0; bb < n_y; ++bb) pyb[bb] = 0;
    for (int a = 0; a < nx; ++a)
        for (int bb = 0; bb < n_y; ++bb) pyb[bb] += t[a * n_y + bb];
    double mi = 0.0;
    for (int a = 0; a < nx; ++a) {
        long long pxa = 0;
        for (int bb = 0; bb < n_y; ++bb) pxa += t[a * n_y + bb];
        if (pxa == 0) continue;
        for (int bb = 0; bb < n_y; ++bb) {
            const long long nab = t[a * n_y + bb];
            if (nab == 0) continue;
            mi += ((double)nab / (double)n) * log((double)(nab * n) / (double)(pxa * pyb[bb]));
        }
    }
    out[(long long)(p0 + pp) * K + k] = mi;
}
"""

_KERNEL: Any = None
_LOCK = threading.Lock()


def _get_kernel(cp: Any) -> Any:
    """Compile (once) the table-to-MI kernel."""
    global _KERNEL
    with _LOCK:
        if _KERNEL is None:
            _KERNEL = cp.RawKernel(_SRC.replace("MAXC", str(MAX_CLASSES)), "family_mi_from_tables")
        return _KERNEL


def family_null_max_counts(cols: np.ndarray, n_x: np.ndarray, perms: np.ndarray, n_y: int) -> Optional[np.ndarray]:
    """Per-permutation maximum plug-in MI over the ``(K, n)`` columns against the ``(B, n)`` permuted labels, or ``None`` when the tables do not fit what the kernel holds (the caller keeps the
    thread-per-pair kernel). Same values, bit for bit, as ``_family_maxt_cuda.family_null_max_device`` and the CPU kernel."""
    import cupy as cp

    K, n = int(cols.shape[0]), int(cols.shape[1])
    B = int(perms.shape[0])
    n_y = int(n_y)
    if K == 0 or B == 0 or n == 0 or n_y > MAX_CLASSES or n_y < 1:
        return None
    nx_host = np.ascontiguousarray(n_x, dtype=np.int64)
    max_jc = int(nx_host.max()) * n_y
    if max_jc * 4 > 48 * 1024:
        return None
    cells_dtype = np.int16 if max_jc < 32768 else np.int32
    # the candidate rows hold cell bases (bin * n_y), read at their own width by the counting kernel
    cols_d = cp.asarray(np.ascontiguousarray(cols)).astype(cells_dtype) * cells_dtype(n_y)
    perms_d = cp.asarray(np.ascontiguousarray(perms)).astype(cp.int8 if n_y <= 127 else cp.int32)
    nx_d = cp.asarray(nx_host)
    out_d = cp.zeros((B, K), dtype=cp.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    kern = _get_kernel(cp)
    pbc_max = min(B, _PERM_BLOCK)
    kc_max = max(1, _TABLE_BUDGET_BYTES // (pbc_max * max_jc * 4))
    for p0 in range(0, B, pbc_max):
        pbc = min(pbc_max, B - p0)
        yp = perms_d[p0 : p0 + pbc]
        for k0 in range(0, K, kc_max):
            kc = min(kc_max, K - k0)
            tables = perm_joint_counts(cp, cols_d[k0 : k0 + kc], yp, max_jc)
            if tables is None:
                return None
            rows = kc * pbc
            threads = block_size_for(kern, work_items=rows)
            kern(((rows + threads - 1) // threads,), (threads,), (tables, nx_d, np.int32(n_y), np.int32(max_jc), np.int32(kc), np.int32(pbc), np.int32(k0), np.int32(p0), np.int32(K), np.int64(n), out_d))
            del tables
    best = cp.maximum(out_d.max(axis=1), 0.0)
    host: np.ndarray = cp.asnumpy(best)
    return host
