"""Device twin of the non-polynomial basis cells (fourier, rbf, sigmoid, pade) of ``_nonpoly_cells``.

Win axis: one launch builds every ``(column, basis, degree)`` cell of a column block, a thread per output element, so it wins wherever the host loop builds a vector per cell (a hundred columns of 100000
rows: milliseconds against seconds); it loses on frames of a few thousand rows where the launch cost dominates (the caller's KTC dispatch decides).

The per-column fit statistics are computed on the device from the (finite) column block: fourier needs the column range, pade the mean and standard deviation, rbf and sigmoid the nine quantile centres /
thresholds (a column sort and the linear interpolation numpy's ``quantile`` uses) plus the Silverman bandwidth / the interquantile slope. Cells are evaluated in ``{ACC}`` (double) and stored as ``{REAL}``.
"""

from __future__ import annotations

import math
import logging
from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache

from ._nonpoly_cells import MAX_CENTRE_DEGREE, NONPOLY_BASES
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # noqa: F401  (thrust aborts the process on a short pool)

logger = logging.getLogger(__name__)

_KIND = {"fourier": 0, "rbf": 1, "sigmoid": 2, "pade": 3}

_SRC = r"""
extern "C" __global__ void nonpoly_cells(const {REAL}* __restrict__ x, const {ACC}* __restrict__ par, const int* __restrict__ cell_col, const int* __restrict__ cell_kind,
                                         const int* __restrict__ cell_deg, const long long n, const int K, const int C, {REAL}* __restrict__ out) {
    const {ACC} two_pi = 6.283185307179586476925286766559;
    // The cell parameters of a block are staged once in shared memory when there are at most 256 cells (the common case); above that the
    // block reads them from global memory. The branch is uniform in the block.
    __shared__ int s_col[256], s_kind[256], s_deg[256];
    __shared__ {ACC} s_p0[256], s_p1[256];
    const bool staged = C <= 256;
    if (staged) {
        for (int t = threadIdx.x; t < C; t += blockDim.x) {
            s_col[t] = cell_col[t]; s_kind[t] = cell_kind[t]; s_deg[t] = cell_deg[t];
            s_p0[t] = par[4 * t]; s_p1[t] = par[4 * t + 1];
        }
        __syncthreads();
    }
    // 32-bit cell index: the output has n * C entries, far below 2^32 for every fit this kernel runs on.
    const unsigned int UC = (unsigned int)C;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (unsigned int)n * UC; idx += gridDim.x * blockDim.x) {
        const unsigned int i = idx / UC;
        const int c = (int)(idx - i * UC);
        const int col = staged ? s_col[c] : cell_col[c];
        const {ACC} v = ({ACC})x[(long long)i * K + col];
        const {ACC} p0 = staged ? s_p0[c] : par[4 * c];
        const {ACC} p1 = staged ? s_p1[c] : par[4 * c + 1];
        const int d = staged ? s_deg[c] : cell_deg[c];
        {ACC} r = 0.0;
        switch (staged ? s_kind[c] : cell_kind[c]) {
        case 0: {   // fourier: p0 = lo, p1 = span (0 marks a degenerate column, z = 0)
            const {ACC} z = p1 > 0.0 ? (v - p0) / p1 : 0.0;
            const int k = (d + 1) / 2;
            const {ACC} ang = two_pi * ({ACC})k * z;
            r = (d & 1) ? sin(ang) : cos(ang);
            break;
        }
        case 1: {   // rbf: p0 = centre, p1 = 1 / bandwidth^2
            const {ACC} dz = v - p0;
            r = exp(-dz * dz * p1);
            break;
        }
        case 2: {   // sigmoid: p0 = threshold, p1 = slope; the sign-split form never overflows exp
            const {ACC} arg = p1 * (v - p0);
            if (arg >= 0.0) { r = 1.0 / (1.0 + exp(-arg)); } else { const {ACC} e = exp(arg); r = e / (1.0 + e); }
            break;
        }
        default: {  // pade: p0 = mean, p1 = std (0 marks a degenerate column); 1 / (1 + z^d), zero within 1e-3 of a pole
            const {ACC} z = p1 > 0.0 ? (v - p0) / p1 : 0.0;
            {ACC} den = 1.0;
            {ACC} zp = 1.0;
            for (int t = 0; t < d; ++t) zp *= z;
            den = 1.0 + zp;
            r = fabs(den) < 1e-3 ? 0.0 : 1.0 / den;
        }
        }
        out[idx] = ({REAL})r;
    }
}
"""

_MODULES_BY_REAL = {"double": RawModuleCache(_SRC.replace("{REAL}", "double").replace("{ACC}", "double")), "float": RawModuleCache(_SRC.replace("{REAL}", "float").replace("{ACC}", "double"))}


def _module_for(M: Any) -> Any:
    """The compiled module for the element type of ``M`` (float32 or double)."""
    return _MODULES_BY_REAL["float" if M.dtype.itemsize == 4 else "double"]


def _quantile_columns(cp: Any, M: Any, fractions: "list[float]") -> Any:
    """``(len(fractions), K)`` linear-interpolation quantiles of every column of the finite ``(n, K)`` matrix ``M``, from one column sort (numpy's ``quantile`` formula)."""
    n = int(M.shape[0])
    # the order statistics by the sort-free radix select (same values, same linear interpolation); the column sort stays the fallback
    from .._gpu_resident_select import _radix_quantiles

    if n > 0 and M.ndim == 2 and M.shape[1] > 0:
        try:
            radix = _radix_quantiles(M, list(fractions))
        except Exception as _silent_exc:
            logger.warning("_nonpoly_cells_gpu.py: radix quantiles failed (%r); the column sort is used", _silent_exc)
            radix = None
        if radix is not None:
            return radix
    srt = device_sort(M, axis=0)
    out = []
    for q in fractions:
        v = q * (n - 1)
        lo = math.floor(v)
        hi = min(lo + 1, n - 1)
        t = v - lo
        # Intentional float64: quantile bounds are read in double
        a = srt[lo].astype(cp.float64)
        # Intentional float64: quantile bounds are read in double
        b = srt[hi].astype(cp.float64)
        diff = b - a
        out.append(b - diff * (1.0 - t) if t >= 0.5 else a + diff * t)
    return cp.stack(out)


def _cell_specs(basis: str, degrees: "list[int]") -> "list[int]":
    """The degrees of ``degrees`` that have a cell for ``basis`` (the centre bases stop at nine)."""
    if basis in ("rbf", "sigmoid"):
        return [d for d in degrees if 1 <= d <= MAX_CENTRE_DEGREE]
    return [d for d in degrees if d >= 1]


def nonpoly_block_gpu(cp: Any, M: Any, basis: str, degrees: "list[int]") -> "tuple[Any, list[tuple[int, int]]] | None":
    """``(cells (n, C), [(column, degree)] * C)`` of the non-polynomial ``basis`` for the finite ``(n, K)`` device matrix ``M``, or ``None`` when no degree has a cell.

    Column ``c`` of the result is the cell of ``(column, degree)`` listed at position ``c``; the order is degree-major, columns inside a degree.
    """
    if basis not in NONPOLY_BASES:
        raise KeyError(basis)
    degs = _cell_specs(basis, [int(d) for d in degrees])
    if not degs:
        return None
    n, K = int(M.shape[0]), int(M.shape[1])
    # Intentional float64: double alias for the quantile kernel
    f64 = cp.float64
    Mf = M.astype(f64, copy=False)
    cols_of_cell: list = []
    deg_of_cell: list = []
    par_blocks: list = []
    if basis == "fourier":
        lo, hi = Mf.min(axis=0), Mf.max(axis=0)
        span = hi - lo
        degenerate = span <= 1e-9 * (cp.abs(lo) + cp.abs(hi) + 1.0)
        p0, p1 = lo, cp.where(degenerate, 0.0, span)
        par_blocks.extend(cp.stack([p0, p1, cp.zeros(K, f64), cp.zeros(K, f64)], axis=1) for _d in degs)
    elif basis == "pade":
        mean = Mf.mean(axis=0)
        std = Mf.std(axis=0)
        degenerate = std <= 1e-9 * (cp.abs(mean) + 1.0)
        p1 = cp.where(degenerate, 0.0, std)
        par_blocks.extend(cp.stack([mean, p1, cp.zeros(K, f64), cp.zeros(K, f64)], axis=1) for _d in degs)
    else:
        fractions = [float(f) for f in np.linspace(0.1, 0.9, 9)]
        quant = _quantile_columns(cp, M, fractions)  # (9, K)
        if basis == "rbf":
            std = Mf.std(axis=0)
            mag = cp.abs(Mf).max(axis=0)
            from mlframe.feature_selection.filters._safe_scale import _REL_TOL

            bandwidth = cp.where(std > _REL_TOL * mag, 1.06 * std * (max(n, 1) ** (-0.2)), cp.maximum(mag, 1.0))
            second = 1.0 / (bandwidth * bandwidth)
        else:
            iqr = quant[8] - quant[0] + 1e-12
            second = 4.0 / (iqr / 9.0)
        par_blocks.extend(cp.stack([quant[d - 1], second, cp.zeros(K, f64), cp.zeros(K, f64)], axis=1) for d in degs)
    cols_of_cell.extend(np.arange(K, dtype=np.int32) for _d in degs)
    deg_of_cell.extend(np.full(K, d, dtype=np.int32) for d in degs)
    cell_col = cp.asarray(np.concatenate(cols_of_cell))
    cell_deg = cp.asarray(np.concatenate(deg_of_cell))
    cell_kind = cp.full(cell_col.shape, _KIND[basis], dtype=cp.int32)
    par = cp.ascontiguousarray(cp.concatenate(par_blocks, axis=0), dtype=f64)
    C = int(cell_col.shape[0])
    out = cp.empty((n, C), dtype=M.dtype)
    kernel = _module_for(M).get(cp).get_function("nonpoly_cells")
    threads = block_size_for(kernel, work_items=n * C)
    blocks = max(1, min((n * C + threads - 1) // threads, 65535 * 4))
    kernel((blocks,), (threads,), (cp.ascontiguousarray(M), par, cell_col, cell_kind, cell_deg, np.int64(n), np.int32(K), np.int32(C), out))
    cells = [(int(c), int(d)) for d, c in zip(np.concatenate(deg_of_cell), np.concatenate(cols_of_cell))]
    return out, cells
