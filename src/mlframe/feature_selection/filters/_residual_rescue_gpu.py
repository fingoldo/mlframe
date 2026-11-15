"""Device twin of the column scoring of the residual rescue (``_residual_rescue_fe._score_columns``).

For every pool column the host takes the residual of the additive model on the even rows, bins it by the column's codes, and reads the held-out evidence on the odd rows: the paired
reduction ``d = e^2 - (e - m_bin)^2`` of the squared residual by the shrunk even-row bin mean, its mean and its t statistic. One block per column does the same: the even rows fill a shared
(sum, count) histogram of the bins (the sums as 2^-36 fixed-point integers, so the result does not depend on the order in which the threads add), the odd rows give the sum of ``d`` and then, with the mean in hand, the sum of ``(d - mean)^2``, so the variance has no ``E[d^2] - mean^2`` cancellation
(the host uses a Welford running mean, the same quantity to rounding).

Columns go to the device in column blocks sized from the room under the pool (``_device_blocks``), halved when a block does not fit; the residual and the even mask are uploaded once.
The bin codes of the pool (``_residual_rescue_fe._codes_matrix``, 60 percent of a rescue call in the cProfile) have a twin too, ``codes_matrix_gpu``: the quantile edges of a strided sample come
from a batched device sort, and one kernel codes every element by a binary search over its column's edges. The caller
(``_residual_rescue_fe``) picks host or device per call from the kernel-tuning entry of ``_residual_rescue_ktc``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._gpu_strict_fe._audit import audit_tag
from ._kernel_types import render

__all__ = ["score_columns_gpu", "codes_matrix_gpu"]

_SCORE_SRC = render(
    r"""
extern "C" __global__
void rescue_score_columns(const {QT}* __restrict__ Q, const {ACC}* __restrict__ resid, const unsigned char* __restrict__ even, const long long n, const int n_bins, const {ACC} shrink,
                          {ACC}* __restrict__ t_out, {ACC}* __restrict__ gain_out) {
    extern __shared__ unsigned long long sh[];            // fixed-point sum[n_bins], count[n_bins], then 8 reduction slots (doubles)
    unsigned long long* s = sh;
    unsigned long long* c = sh + n_bins;
    {ACC}* red = ({ACC}*)(sh + 2 * n_bins);
    const int col = blockIdx.x;
    const {QT}* q = Q + (long long)col * n;
    const {ACC} FIX = 68719476736.0;                      // 2^36: the even-row bin sums are added as integers, whose order does not matter, so a run gives the same bits as the next
    for (int b = threadIdx.x; b < 2 * n_bins; b += blockDim.x) sh[b] = 0ULL;
    __syncthreads();
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        if (even[i]) {
            atomicAdd(&s[q[i]], (unsigned long long)llrint(resid[i] * FIX));
            atomicAdd(&c[q[i]], 1ULL);
        }
    }
    __syncthreads();
    {ACC} sum = 0.0;
    long long cnt = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        if (!even[i]) {
            const int b = q[i];
            const {ACC} mc = ({ACC})(long long)s[b] / FIX / (({ACC})c[b] + shrink);
            const {ACC} e = resid[i];
            sum += e * e - (e - mc) * (e - mc);
            cnt += 1;
        }
    }
    // block reduction of (sum, cnt) through shared memory
    __shared__ long long cnt_sh[32];
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, o);
        cnt += __shfl_down_sync(0xffffffffu, cnt, o);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) { red[warp] = sum; cnt_sh[warp] = cnt; }
    __syncthreads();
    __shared__ {ACC} mean_sh;
    __shared__ long long n_odd;
    if (threadIdx.x == 0) {
        {ACC} total = 0.0; long long k = 0;
        for (int w = 0; w < nwarps; ++w) { total += red[w]; k += cnt_sh[w]; }
        n_odd = k;
        mean_sh = k > 0 ? total / ({ACC})k : 0.0;
    }
    __syncthreads();
    const {ACC} mean = mean_sh;
    {ACC} m2 = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        if (!even[i]) {
            const int b = q[i];
            const {ACC} mc = ({ACC})(long long)s[b] / FIX / (({ACC})c[b] + shrink);
            const {ACC} e = resid[i];
            const {ACC} dd = (e * e - (e - mc) * (e - mc)) - mean;
            m2 += dd * dd;
        }
    }
    for (int o = 16; o > 0; o >>= 1) m2 += __shfl_down_sync(0xffffffffu, m2, o);
    __syncthreads();
    if (lane == 0) red[warp] = m2;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (n_odd > 1) {
            {ACC} tot2 = 0.0;
            for (int w = 0; w < nwarps; ++w) tot2 += red[w];
            const {ACC} var = fmax(tot2 / ({ACC})n_odd, 0.0);
            const {ACC} se = sqrt(var / ({ACC})n_odd);
            t_out[col] = se > 0.0 ? mean / se : mean * 1e300;
            gain_out[col] = mean;
        } else {
            t_out[col] = 0.0;
            gain_out[col] = 0.0;
        }
    }
}
""",
)

_KERNELS: dict = {}
_THREADS = 256


def _kernel(cp: Any, narrow: bool) -> Any:
    """The compiled scoring kernel: bin codes read as bytes when every code fits (``n_bins <= 127``), else as 32-bit integers; one per code type and process."""
    if narrow not in _KERNELS:
        _KERNELS[narrow] = cp.RawKernel(_SCORE_SRC.replace("{QT}", "signed char" if narrow else "int"), "rescue_score_columns")
    return _KERNELS[narrow]


def score_columns_gpu(Q: np.ndarray, resid: np.ndarray, even: np.ndarray, n_bins: int, shrink: float) -> "tuple[np.ndarray, np.ndarray]":
    """``(t, gain)`` per row of ``Q`` (a column of the pool per row, integer bin codes below ``n_bins``): the device twin of ``_residual_rescue_fe._score_columns``.

    Raises on a device fault so the caller can fall back to the host loop."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    p, n = Q.shape
    if p == 0:
        return np.zeros(0), np.zeros(0)
    resid_d = cp.asarray(np.ascontiguousarray(resid, dtype=np.float64))  # Intentional float64: the residual statistic is formed and summed in double, as the host does
    even_d = cp.asarray(np.ascontiguousarray(even, dtype=np.uint8))
    narrow = int(n_bins) <= 127  # codes are below n_bins: a byte holds them, a quarter of the traffic of 32-bit codes
    kernel = _kernel(cp, narrow)
    shared = (2 * int(n_bins) + 8) * 8

    def score_block(lo: int, hi: int) -> np.ndarray:
        """``(2, hi - lo)`` array of t and gain of the columns ``lo:hi``."""
        q_d = cp.asarray(np.ascontiguousarray(Q[lo:hi], dtype=np.int8 if narrow else np.int32))
        width = hi - lo
        t_d = cp.empty(width, dtype=cp.float64)  # Intentional float64: statistic output compared to the host at full precision
        g_d = cp.empty(width, dtype=cp.float64)  # Intentional float64: statistic output compared to the host at full precision
        kernel((width,), (_THREADS,), (q_d, resid_d, even_d, np.int64(n), np.int32(n_bins), np.float64(shrink), t_d, g_d), shared_mem=shared)  # launch-ok: one block per column is the design
        return np.stack([cp.asnumpy(t_d), cp.asnumpy(g_d)])

    block = first_block(cp, n, p, 1 if narrow else 4, 3)
    parts = run_column_blocks(cp, p, block, score_block)
    both = np.concatenate(parts, axis=1)
    return both[0], both[1]


_CODES_SRC_TEMPLATE = r"""
extern "C" __global__
void rescue_codes(const {REAL}* __restrict__ M, const {REAL}* __restrict__ edges, const long long n, const int n_edges, const unsigned char* __restrict__ ok, signed char* __restrict__ codes) {
    const int col = blockIdx.y;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!ok[col]) { codes[(long long)col * n + i] = 0; return; }
    const {REAL} v = M[(long long)col * n + i];
    int code = 0;
    if (isfinite(v)) {
        const {REAL}* e = edges + (long long)col * n_edges;
        int lo = 0, hi = n_edges;                       // np.searchsorted(edges, v, side="right"): the number of edges <= v
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (e[mid] <= v) lo = mid + 1; else hi = mid; }
        code = lo;
    }
    codes[(long long)col * n + i] = (signed char)code;
}
"""
_CODES_SRC = render(_CODES_SRC_TEMPLATE, real=np.float32)  # the float32 build, the form the offline checks compile
_CODES_KERNELS: dict = {}


def _codes_kernel(cp: Any, real: Any) -> Any:
    """The compiled coding kernel for float32 or float64 columns."""
    key = np.dtype(real)
    if key not in _CODES_KERNELS:
        _CODES_KERNELS[key] = cp.RawKernel(render(_CODES_SRC_TEMPLATE, real=real), "rescue_codes")
    return _CODES_KERNELS[key]


def codes_matrix_gpu(M: np.ndarray, n_bins: int, sample: int, min_finite: int, few_classes: int) -> "tuple[np.ndarray, np.ndarray]":
    """``(codes, ok)`` of the rows of ``M`` (a column of the data per row): the device twin of ``_residual_rescue_fe._codes_matrix``.

    The edges are the quantiles of a strided sample of ``sample`` finite values per column (a batched device sort of the sample, non-finite values moved to the end), a column is unusable
    when it has fewer than ``min_finite`` finite values or at most ``few_classes`` distinct sample values, and every finite value is coded by a binary search over its column's edges
    (non-finite values and unusable columns go to code 0). Raises on a device fault so the caller can fall back to the host loop."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    p, n = M.shape
    codes_out = np.zeros((p, n), dtype=np.int64)
    ok_out = np.zeros(p, dtype=bool)
    if p == 0:
        return codes_out, ok_out
    kernel = _codes_kernel(cp, M.dtype)
    ne = int(n_bins) - 1
    step = max(n // int(sample), 1)

    def code_block(lo: int, hi: int) -> int:
        """Code the columns ``lo:hi``; the results are written into ``codes_out`` / ``ok_out``."""
        X = cp.asarray(np.ascontiguousarray(M[lo:hi]))
        fin = cp.isfinite(X)
        n_fin = fin.sum(axis=1)
        s = cp.where(fin[:, ::step], X[:, ::step], cp.asarray(np.inf, dtype=X.dtype))
        s = cp.sort(s, axis=1)
        m = fin[:, ::step].sum(axis=1)  # finite sample values per column, sorted to the front
        width = s.shape[1]
        idx = cp.arange(width)[None, :]
        distinct = 1 + ((s[:, 1:] != s[:, :-1]) & (idx[:, 1:] < m[:, None])).sum(axis=1)
        usable = (n_fin >= int(min_finite)) & (m >= 2) & (distinct > int(few_classes))
        k = cp.arange(1, ne + 1, dtype=cp.int64)[None, :]
        pos = cp.minimum((k * m[:, None]) // int(n_bins), cp.maximum(m[:, None] - 1, 0))
        edges = cp.take_along_axis(s, pos, axis=1)
        q = cp.empty((hi - lo, n), dtype=cp.int8)
        usable_u8 = usable.astype(cp.uint8)
        threads = 256
        kernel(((n + threads - 1) // threads, hi - lo), (threads,), (X, cp.ascontiguousarray(edges), np.int64(n), np.int32(ne), usable_u8, q))  # launch-ok: columns on the grid y axis, rows on x
        # The int8 codes are the result of this function: the rescue scorer that reads them is host (njit) code, so they cross the bus on purpose, one byte per cell;
        # the residency audit files them under their own tag instead of counting them as intermediate traffic.
        with audit_tag("residual_rescue_codes_host"):
            codes_out[lo:hi] = cp.asnumpy(q)
            ok_out[lo:hi] = cp.asnumpy(usable)
        return hi - lo

    block = first_block(cp, n, p, 12, 3)
    run_column_blocks(cp, p, block, code_block)
    return codes_out, ok_out
