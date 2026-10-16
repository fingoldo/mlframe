"""One-launch device build of the external-validation candidate matrix ``out[:, e * n_ops + o] = op_o(param_a, ext_e)``.

``_gpu_resident_extval.gpu_materialise_extval_codes_host`` used to issue one cupy expression (one to three kernels) plus one column-assignment
kernel per (external factor, operand) pair: about 1.3 million launches in a 50k-row conjunction fit. This kernel computes every cell of the
``(n, n_ext * n_ops)`` double-precision matrix in a single launch, with the arithmetic of ``_extval_op_gpu`` (and of ``_materialise_extval_njit``) op for op.
"""

from __future__ import annotations

_SOURCE = r"""
#define NAN __longlong_as_double(0x7ff8000000000000LL)
extern "C" __global__ void extval_candidates(const double* __restrict__ pa, const double* __restrict__ ext, const int* __restrict__ ops,
                                             double* __restrict__ out, const long long n, const int n_ops, const long long K, const long long total) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;
    const long long i = idx / K;
    const long long c = idx - i * K;
    const long long e = c / n_ops;
    const double a = pa[i];
    const double b = ext[e * n + i];
    double r;
    switch (ops[c - e * n_ops]) {
        case 0: r = a * b; break;
        case 1: r = a + b; break;
        case 2: r = a - b; break;
        case 3: r = a / (b == 0.0 ? 1e-9 : b); break;
        case 4: r = (isnan(a) || isnan(b)) ? NAN : (a > b ? a : b); break;
        case 5: r = (isnan(a) || isnan(b)) ? NAN : (a < b ? a : b); break;
        case 6: r = fabs(a - b); break;
        case 7: r = (a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : (a == 0.0 ? 0.0 : NAN))) * fabs(b); break;
        default: r = a / (fabs(b) + 1.0); break;
    }
    out[idx] = r;
}
"""

_kernel = None


def fused_extval_candidates(cp, pa, ext_stack, ops):
    """``(n, n_ext * n_ops)`` double-precision device matrix of every ``op(pa, ext_e)``; ``ext_stack`` is ``(n_ext, n)`` double precision, ``ops`` the int32 op codes."""
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_SOURCE, "extval_candidates")
    n = int(pa.shape[0])
    n_ext = int(ext_stack.shape[0])
    n_ops = int(ops.shape[0])
    # Intentional float64: the candidate values match the host njit materialisation bit for bit
    out = cp.empty((n, n_ext * n_ops), dtype=cp.float64)
    total = n * n_ext * n_ops
    block = 256
    _kernel(((total + block - 1) // block,), (block,), (pa, ext_stack, ops, out, cp.int64(n), cp.int32(n_ops), cp.int64(n_ext * n_ops), cp.int64(total)))
    return out
