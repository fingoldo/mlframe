"""Near-constant mask (and the row extremes) of a column-major candidate block in one pass: ``var(row) > (rel_tol * max|row|)^2`` for every row of a ``(kk, n)`` resident matrix.

The usability pool's sentinel test used ``cand.var(axis=1) > (tol * cp.abs(cand).max(axis=1)) ** 2``: cupy's generic ``var`` reduction (25 ms a launch for a (450, 100000) float32 block, 8 GB/s of the
card's 112), an ``abs`` temporary of the whole block (5 ms) and a segmented ``max``. The kernel here is one launch with one block per row: threads stride the contiguous row (coalesced), accumulate the
shifted sums ``sum(x - x0)`` and ``sum((x - x0)^2)`` (``x0`` the row's first value: the variance of the shifted data, free of the cancellation that offset columns cause in raw power sums) and the maximum
of ``|x|`` in the accumulator type, and a shared-memory tree reduction yields the flag. Row-streamed: nothing sized by ``n`` is held in shared memory. Elements are read in the column dtype
``{REAL}``; the accumulators are ``{ACC}``.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ._kernel_types import render

THREADS = 256  # the TH define of the kernel source: keep the two equal (test_cuda_kernel_register_pressure compiles both)

_SRC = r"""
#define TH 256  // equals THREADS below
extern "C" __global__ void degenerate_rows(const {REAL}* __restrict__ cand, const long long n, const {ACC} rel_tol, unsigned char* __restrict__ live, {ACC}* __restrict__ rmin, {ACC}* __restrict__ rmax) {
    __shared__ {ACC} s0[TH];
    __shared__ {ACC} s1[TH];
    __shared__ {ACC} s2[TH];
    __shared__ {ACC} s3[TH];
    const {REAL}* __restrict__ row = cand + (long long)blockIdx.x * n;
    const {REAL} x0 = (n > 0) ? row[0] : ({REAL})0;
    {REAL} a = 0, b = 0, lo = x0, hi = x0;  // per-thread partials in the column dtype (about n / blockDim terms; the shift makes constants exact), promoted to {ACC} for the block reduction
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const {REAL} v = row[i];
        const {REAL} d = v - x0;
        a += d;
        b += d * d;
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
    }
    s0[threadIdx.x] = ({ACC})a; s1[threadIdx.x] = ({ACC})b; s2[threadIdx.x] = ({ACC})lo; s3[threadIdx.x] = ({ACC})hi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s0[threadIdx.x] += s0[threadIdx.x + s];
            s1[threadIdx.x] += s1[threadIdx.x + s];
            s2[threadIdx.x] = s2[threadIdx.x + s] < s2[threadIdx.x] ? s2[threadIdx.x + s] : s2[threadIdx.x];
            s3[threadIdx.x] = s3[threadIdx.x + s] > s3[threadIdx.x] ? s3[threadIdx.x + s] : s3[threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const {ACC} mean_d = s0[0] / ({ACC})n;
        const {ACC} var = s1[0] / ({ACC})n - mean_d * mean_d;
        const {ACC} amax = (-s2[0] > s3[0]) ? -s2[0] : s3[0];
        const {ACC} bar = rel_tol * (amax < 0 ? -amax : amax);
        rmin[blockIdx.x] = s2[0];
        rmax[blockIdx.x] = s3[0];
        live[blockIdx.x] = (var > bar * bar) ? 1 : 0;
    }
}
"""

_KERNELS: dict = {}
_LOCK = threading.Lock()


def kernel(cp: Any, real: Any) -> Any:
    """``degenerate_rows`` for a block of dtype ``real``; compile point of :func:`live_rows` and of the register-pressure registry."""
    name = "float" if np.dtype(real) == np.float32 else "double"
    with _LOCK:
        ker = _KERNELS.get(name)
        if ker is None:
            ker = _KERNELS[name] = cp.RawKernel(render(_SRC, np.float32 if name == "float" else np.float64), "degenerate_rows")  # Intentional float64: accumulators and the double block type
    return ker


def live_rows(cp: Any, cand: Any, rel_tol: float) -> tuple:
    """``(live, row_min, row_max)`` device vectors for the C-contiguous ``(kk, n)`` block ``cand``: ``live[r]`` is True when row ``r`` has spread above ``rel_tol`` times its largest ``|value|``;
    ``row_min`` / ``row_max`` (double precision) are the row extremes the MI edge dedup needs, produced in the same pass."""
    kk, n = int(cand.shape[0]), int(cand.shape[1])
    out = cp.empty(kk, dtype=cp.uint8)
    rmin = cp.empty(kk, dtype=cp.float64)  # Intentional float64: MI edge bounds, compared at full precision
    rmax = cp.empty(kk, dtype=cp.float64)  # Intentional float64: MI edge bounds, compared at full precision
    kernel(cp, cand.dtype)((kk,), (THREADS,), (cand, np.int64(n), np.float64(rel_tol), out, rmin, rmax))  # Intentional float64: accumulator-typed scalar  # launch-ok: the four shared arrays are statically sized by the TH define (256), so the block is fixed by the source
    return out.view(cp.bool_), rmin, rmax
