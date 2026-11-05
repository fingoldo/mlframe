"""One-launch device build of the external-validation candidate matrix ``out[:, e * n_ops + o] = op_o(param_a, ext_e)``.

``_gpu_resident_extval.gpu_materialise_extval_codes_host`` used to issue one cupy expression (one to three kernels) plus one column-assignment
kernel per (external factor, operand) pair: about 1.3 million launches in a 50k-row conjunction fit. This kernel computes every cell of the
``(n, n_ext * n_ops)`` matrix (the element type of the operands) in a single launch, with the arithmetic of ``_extval_op_gpu`` (and of ``_materialise_extval_njit``) op for op.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._kernel_types import ctype_of, render

_SOURCE = r"""
#define NAN_REAL (({REAL})__longlong_as_double(0x7ff8000000000000LL))
extern "C" __global__ void extval_candidates(const {REAL}* __restrict__ pa, const {REAL}* __restrict__ ext, const int* __restrict__ ops,
                                             {REAL}* __restrict__ out, const long long n, const int n_ops, const long long K, const long long total) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;
    const long long i = idx / K;
    const long long c = idx - i * K;
    const long long e = c / n_ops;
    const {REAL} a = pa[i];
    const {REAL} b = ext[e * n + i];
    {REAL} r;
    switch (ops[c - e * n_ops]) {
        case 0: r = a * b; break;
        case 1: r = a + b; break;
        case 2: r = a - b; break;
        case 3: r = a / (b == ({REAL})0 ? ({REAL})1e-9 : b); break;
        case 4: r = (isnan(a) || isnan(b)) ? NAN_REAL : (a > b ? a : b); break;
        case 5: r = (isnan(a) || isnan(b)) ? NAN_REAL : (a < b ? a : b); break;
        case 6: r = fabs(a - b); break;
        case 7: r = (a > ({REAL})0 ? ({REAL})1 : (a < ({REAL})0 ? ({REAL})-1 : (a == ({REAL})0 ? ({REAL})0 : NAN_REAL))) * fabs(b); break;
        default: r = a / (fabs(b) + ({REAL})1); break;
    }
    out[idx] = r;
}
"""

_KERNELS: dict = {}


def fused_extval_candidates(cp: Any, pa: Any, ext_stack: Any, ops: Any) -> Any:
    """``(n, n_ext * n_ops)`` device matrix of every ``op(pa, ext_e)`` in the element type of ``pa`` (float32 or float64); ``ext_stack`` is ``(n_ext, n)`` of the same type, ``ops`` the int32 op codes.

    The kernel is compiled once per element type and the type follows the buffers that are launched, so a float32 operand never meets a ``double*`` kernel."""
    real = np.float32 if pa.dtype == cp.float32 else np.float64
    if ext_stack.dtype != pa.dtype:
        ext_stack = ext_stack.astype(pa.dtype)
    key = ctype_of(real)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = cp.RawKernel(render(_SOURCE, real=real), "extval_candidates")
    n = int(pa.shape[0])
    n_ext = int(ext_stack.shape[0])
    n_ops = int(ops.shape[0])
    out = cp.empty((n, n_ext * n_ops), dtype=pa.dtype)
    total = n * n_ext * n_ops
    block = block_size_for(kernel, work_items=total)
    kernel(((total + block - 1) // block,), (block,), (pa, ext_stack, ops, out, cp.int64(n), cp.int32(n_ops), cp.int64(n_ext * n_ops), cp.int64(total)))
    return out
