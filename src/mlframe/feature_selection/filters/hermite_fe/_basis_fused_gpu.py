"""One kernel for the orthogonal-polynomial design matrix ``B[i, k] = T_k(z[i])``.

The cupy build (:func:`_hermite_prewarp_gpu_resident._build_basis_matrix_gpu`) runs the three-term recurrence one degree at a time, every step several elementwise launches
over the full column plus a strided column write: ~60 launches for a degree-6 pair of designs, in a function the ALS seed calls dozens of times per fit. Here one thread owns
one row and runs the whole recurrence in registers.

The kernel is compiled with ``--fmad=false`` and evaluates each recurrence step in the order the cupy expression does, so every rounding happens where cupy's separate
elementwise kernels round and the result is bit-identical to the loop it replaces.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_MODULE: Optional[Any] = None
_BASIS_CODE = {"hermite": 0, "legendre": 1, "chebyshev": 2, "laguerre": 3}
MAX_COLUMNS = 40  # register budget of the unrolled recurrence

_SRC = r"""
extern "C" __global__ void build_basis(const double* __restrict__ x, double* __restrict__ B, const long long n, const int nc, const int basis) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double xi = x[i];
        double* row = B + i * nc;
        double p2 = 1.0;  // T_{k-2}
        row[0] = p2;
        if (nc <= 1) continue;
        double p1 = (basis == 3) ? (1.0 - xi) : xi;  // T_{k-1}
        row[1] = p1;
        const double two_x = 2.0 * xi;
        for (int k = 2; k < nc; ++k) {
            double v;
            if (basis == 0) {
                v = xi * p1 - (double)(k - 1) * p2;
            } else if (basis == 1) {
                const double t1 = (double)(2 * k - 1) * xi;
                const double t2 = t1 * p1;
                const double t3 = (double)(k - 1) * p2;
                v = (t2 - t3) / (double)k;
            } else if (basis == 2) {
                v = two_x * p1 - p2;
            } else {
                const double t1 = (double)(2 * k - 1) - xi;
                const double t2 = t1 * p1;
                const double t3 = (double)(k - 1) * p2;
                v = (t2 - t3) / (double)k;
            }
            row[k] = v;
            p2 = p1;
            p1 = v;
        }
    }
}
"""


def _module(cp):
    """Compile the kernel once (``--fmad=false``: every operation rounds where the separate cupy kernels round)."""
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            _MODULE = cp.RawModule(code=_SRC, options=("-std=c++14", "--fmad=false"))
        return _MODULE


def build_basis_fused(cp: Any, basis: str, x: Any, n_columns: int) -> Optional[Any]:
    """``(n, n_columns)`` design of the named basis for the float64 resident column ``x``, or ``None`` when the fused kernel does not cover the request (unknown basis, too wide)."""
    code = _BASIS_CODE.get(basis)
    if code is None or n_columns < 1 or n_columns > MAX_COLUMNS:
        return None
    n = int(x.shape[0])
    B = cp.empty((n, n_columns), dtype=cp.float64)
    blocks = max(1, min(4096, (n + 255) // 256))
    _module(cp).get_function("build_basis")((blocks,), (256,), (x, B, cp.int64(n), cp.int32(n_columns), cp.int32(code)))
    return B
