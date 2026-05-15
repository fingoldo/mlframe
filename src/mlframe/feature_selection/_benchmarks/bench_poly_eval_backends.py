"""Benchmark polynomial-eval backends across array sizes and hardware.

Compares numpy, njit (single-thread), njit (parallel prange), cupy
(elementwise), and a custom cupy RawKernel for Hermite, Legendre,
Chebyshev, Laguerre evaluation. Measures wall-time per call, finds
the crossover point where each backend wins, and emits a recommended
size-aware dispatcher table.

Run::

    python -m mlframe.feature_selection._benchmarks.bench_poly_eval_backends
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


# Backend 1: numpy (reference)
from numpy.polynomial.hermite_e import hermeval as _np_hermeval
from numpy.polynomial.legendre import legval as _np_legval
from numpy.polynomial.chebyshev import chebval as _np_chebval
from numpy.polynomial.laguerre import lagval as _np_lagval


# Backend 2: njit single-thread (already in hermite_fe.py)
from mlframe.feature_selection.filters.hermite_fe import (
    _hermeval_njit, _legval_njit, _chebval_njit, _lagval_njit,
)


# Backend 3: njit parallel (prange over array elements)
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco
    def prange(n):
        return range(n)


@njit(cache=True, fastmath=True, parallel=True)
def _hermeval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Parallel njit Hermite-e evaluation. ``prange`` over array
    elements; the recurrence runs once per element. For n>>cache size
    this hides memory latency across cores."""
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = xi * p_curr - (k - 1) * p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _legval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = (two_km1 * xi * p_curr - km1 * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _chebval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = 2.0 * xi * p_curr - p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _lagval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = 1.0 - xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = ((two_km1 - xi) * p_curr - km1 * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


# Backend 4 + 5: cupy
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None


# Backend 6: numba.cuda.jit (Python-as-CUDA, no inline C++)
try:
    from numba import cuda as numba_cuda
    _NUMBA_CUDA_AVAILABLE = numba_cuda.is_available()
except (ImportError, Exception):
    _NUMBA_CUDA_AVAILABLE = False
    numba_cuda = None


if _NUMBA_CUDA_AVAILABLE:
    @numba_cuda.jit
    def _hermeval_numba_cuda_kernel(x, c, nc, n, out):
        i = numba_cuda.grid(1)
        if i >= n:
            return
        xi = x[i]
        if nc == 0:
            out[i] = 0.0
            return
        if nc == 1:
            out[i] = c[0]
            return
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = xi * p_curr - (k - 1) * p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s

    @numba_cuda.jit
    def _legval_numba_cuda_kernel(x, c, nc, n, out):
        i = numba_cuda.grid(1)
        if i >= n:
            return
        xi = x[i]
        if nc == 0:
            out[i] = 0.0
            return
        if nc == 1:
            out[i] = c[0]
            return
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            p_next = ((2 * k - 1) * xi * p_curr - (k - 1) * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s

    @numba_cuda.jit
    def _chebval_numba_cuda_kernel(x, c, nc, n, out):
        i = numba_cuda.grid(1)
        if i >= n:
            return
        xi = x[i]
        if nc == 0:
            out[i] = 0.0
            return
        if nc == 1:
            out[i] = c[0]
            return
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = 2.0 * xi * p_curr - p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s

    @numba_cuda.jit
    def _lagval_numba_cuda_kernel(x, c, nc, n, out):
        i = numba_cuda.grid(1)
        if i >= n:
            return
        xi = x[i]
        if nc == 0:
            out[i] = 0.0
            return
        if nc == 1:
            out[i] = c[0]
            return
        p_prev = 1.0
        p_curr = 1.0 - xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            p_next = ((2 * k - 1 - xi) * p_curr - (k - 1) * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s

    def _hermeval_numba_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        # numba.cuda kernels accept cupy arrays directly via __cuda_array_interface__
        _hermeval_numba_cuda_kernel[grid, block](x_gpu, c_gpu, c_gpu.shape[0], n, out)
        return out

    def _legval_numba_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _legval_numba_cuda_kernel[grid, block](x_gpu, c_gpu, c_gpu.shape[0], n, out)
        return out

    def _chebval_numba_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _chebval_numba_cuda_kernel[grid, block](x_gpu, c_gpu, c_gpu.shape[0], n, out)
        return out

    def _lagval_numba_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _lagval_numba_cuda_kernel[grid, block](x_gpu, c_gpu, c_gpu.shape[0], n, out)
        return out


if _CUPY_AVAILABLE:
    # Element-wise: vectorized cupy operations, leverages cuBLAS-style fusion
    def _hermeval_cupy(x_gpu, c_gpu):
        """Hermite-e on GPU via cupy element-wise loop."""
        nc = c_gpu.shape[0]
        if nc == 0:
            return cp.zeros_like(x_gpu)
        out = cp.full_like(x_gpu, c_gpu[0])
        if nc == 1:
            return out
        p_prev = cp.ones_like(x_gpu)
        p_curr = x_gpu.copy()
        out += c_gpu[1] * p_curr
        for k in range(2, nc):
            p_next = x_gpu * p_curr - (k - 1) * p_prev
            out += c_gpu[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        return out

    def _legval_cupy(x_gpu, c_gpu):
        nc = c_gpu.shape[0]
        if nc == 0:
            return cp.zeros_like(x_gpu)
        out = cp.full_like(x_gpu, c_gpu[0])
        if nc == 1:
            return out
        p_prev = cp.ones_like(x_gpu)
        p_curr = x_gpu.copy()
        out += c_gpu[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = (two_km1 * x_gpu * p_curr - km1 * p_prev) * inv_k
            out += c_gpu[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        return out

    def _chebval_cupy(x_gpu, c_gpu):
        nc = c_gpu.shape[0]
        if nc == 0:
            return cp.zeros_like(x_gpu)
        out = cp.full_like(x_gpu, c_gpu[0])
        if nc == 1:
            return out
        p_prev = cp.ones_like(x_gpu)
        p_curr = x_gpu.copy()
        out += c_gpu[1] * p_curr
        for k in range(2, nc):
            p_next = 2.0 * x_gpu * p_curr - p_prev
            out += c_gpu[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        return out

    def _lagval_cupy(x_gpu, c_gpu):
        nc = c_gpu.shape[0]
        if nc == 0:
            return cp.zeros_like(x_gpu)
        out = cp.full_like(x_gpu, c_gpu[0])
        if nc == 1:
            return out
        p_prev = cp.ones_like(x_gpu)
        p_curr = 1.0 - x_gpu
        out += c_gpu[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = ((two_km1 - x_gpu) * p_curr - km1 * p_prev) * inv_k
            out += c_gpu[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        return out

    # Custom CUDA RawKernel: one thread per output element, registers for
    # recurrence state (no extra memory traffic).
    _HERMEVAL_CUDA_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void hermeval_kernel(const double* __restrict__ x,
                     const double* __restrict__ c,
                     int nc, int n,
                     double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0;
    double p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double p_next = xi * p_curr - (double)(k - 1) * p_prev;
        s += c[k] * p_next;
        p_prev = p_curr;
        p_curr = p_next;
    }
    out[i] = s;
}
""", "hermeval_kernel")

    _LEGVAL_CUDA_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void legval_kernel(const double* __restrict__ x,
                    const double* __restrict__ c,
                    int nc, int n,
                    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0;
    double p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double inv_k = 1.0 / (double)k;
        double two_km1 = (double)(2 * k - 1);
        double km1 = (double)(k - 1);
        double p_next = (two_km1 * xi * p_curr - km1 * p_prev) * inv_k;
        s += c[k] * p_next;
        p_prev = p_curr;
        p_curr = p_next;
    }
    out[i] = s;
}
""", "legval_kernel")

    _CHEBVAL_CUDA_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void chebval_kernel(const double* __restrict__ x,
                     const double* __restrict__ c,
                     int nc, int n,
                     double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0;
    double p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double p_next = 2.0 * xi * p_curr - p_prev;
        s += c[k] * p_next;
        p_prev = p_curr;
        p_curr = p_next;
    }
    out[i] = s;
}
""", "chebval_kernel")

    _LAGVAL_CUDA_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void lagval_kernel(const double* __restrict__ x,
                    const double* __restrict__ c,
                    int nc, int n,
                    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0;
    double p_curr = 1.0 - xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double inv_k = 1.0 / (double)k;
        double two_km1 = (double)(2 * k - 1);
        double km1 = (double)(k - 1);
        double p_next = ((two_km1 - xi) * p_curr - km1 * p_prev) * inv_k;
        s += c[k] * p_next;
        p_prev = p_curr;
        p_curr = p_next;
    }
    out[i] = s;
}
""", "lagval_kernel")

    def _hermeval_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _HERMEVAL_CUDA_KERNEL((grid,), (block,), (x_gpu, c_gpu, c_gpu.shape[0], n, out))
        return out

    def _legval_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _LEGVAL_CUDA_KERNEL((grid,), (block,), (x_gpu, c_gpu, c_gpu.shape[0], n, out))
        return out

    def _chebval_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _CHEBVAL_CUDA_KERNEL((grid,), (block,), (x_gpu, c_gpu, c_gpu.shape[0], n, out))
        return out

    def _lagval_cuda(x_gpu, c_gpu):
        n = x_gpu.shape[0]
        out = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _LAGVAL_CUDA_KERNEL((grid,), (block,), (x_gpu, c_gpu, c_gpu.shape[0], n, out))
        return out


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _time_call(fn, args, n_warmup=3, n_iter=200):
    """Return median per-call wall-time over ``n_iter`` calls after
    ``n_warmup`` warmup calls."""
    for _ in range(n_warmup):
        fn(*args)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = fn(*args)
        if _CUPY_AVAILABLE and hasattr(out, "device"):
            cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter() - t0
        times.append(t)
    times.sort()
    return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="hermite",
                        choices=["hermite", "legendre", "chebyshev", "laguerre"])
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--include-h2d", action="store_true",
                        help="for cupy backends, include host->device transfer in timing")
    args = parser.parse_args()

    BACKENDS = {
        "hermite": [
            ("numpy", _np_hermeval),
            ("njit", _hermeval_njit),
            ("njit_par", _hermeval_njit_parallel),
        ],
        "legendre": [
            ("numpy", _np_legval),
            ("njit", _legval_njit),
            ("njit_par", _legval_njit_parallel),
        ],
        "chebyshev": [
            ("numpy", _np_chebval),
            ("njit", _chebval_njit),
            ("njit_par", _chebval_njit_parallel),
        ],
        "laguerre": [
            ("numpy", _np_lagval),
            ("njit", _lagval_njit),
            ("njit_par", _lagval_njit_parallel),
        ],
    }
    cupy_backends = {
        "hermite": [("cupy", _hermeval_cupy), ("cuda_kernel", _hermeval_cuda)],
        "legendre": [("cupy", _legval_cupy), ("cuda_kernel", _legval_cuda)],
        "chebyshev": [("cupy", _chebval_cupy), ("cuda_kernel", _chebval_cuda)],
        "laguerre": [("cupy", _lagval_cupy), ("cuda_kernel", _lagval_cuda)],
    }
    if _NUMBA_CUDA_AVAILABLE:
        cupy_backends["hermite"].append(("numba_cuda", _hermeval_numba_cuda))
        cupy_backends["legendre"].append(("numba_cuda", _legval_numba_cuda))
        cupy_backends["chebyshev"].append(("numba_cuda", _chebval_numba_cuda))
        cupy_backends["laguerre"].append(("numba_cuda", _lagval_numba_cuda))

    sizes = [500, 2000, 10_000, 100_000, 1_000_000]
    coef_lengths = [3, 5]  # degree 2 and degree 4

    print(f"\n=== Polynomial-eval backend bench: {args.basis} ===")
    print(f"  CuPy available: {_CUPY_AVAILABLE}")
    print(f"  iterations per measurement: {args.n_iter}")
    print()

    rng = np.random.default_rng(42)
    results = []
    for nc in coef_lengths:
        c_cpu = rng.normal(size=nc).astype(np.float64)
        if _CUPY_AVAILABLE:
            c_gpu = cp.asarray(c_cpu)
        for n in sizes:
            x_cpu = rng.normal(size=n).astype(np.float64)
            row = {"basis": args.basis, "n": n, "nc": nc, "backends": {}}
            print(f"  --- n={n}, degree={nc - 1} ---")
            for name, fn in BACKENDS[args.basis]:
                t = _time_call(fn, (x_cpu, c_cpu), n_warmup=3, n_iter=args.n_iter)
                row["backends"][name] = t * 1e6  # microseconds
                print(f"    {name:10s}: {t * 1e6:8.1f}us")
            if _CUPY_AVAILABLE:
                x_gpu = cp.asarray(x_cpu)
                cp.cuda.runtime.deviceSynchronize()
                for name, fn in cupy_backends[args.basis]:
                    if args.include_h2d:
                        wrapped = lambda x_local=x_cpu, c_local=c_cpu, f=fn: f(cp.asarray(x_local), cp.asarray(c_local))
                        t = _time_call(wrapped, (), n_warmup=3, n_iter=max(20, args.n_iter // 5))
                    else:
                        t = _time_call(fn, (x_gpu, c_gpu), n_warmup=3, n_iter=args.n_iter)
                    row["backends"][name] = t * 1e6
                    label = name + ("+h2d" if args.include_h2d else "")
                    print(f"    {label:10s}: {t * 1e6:8.1f}us")
            print()
            results.append(row)

    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"poly_eval_backends_{args.basis}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"  -> {out_file}")

    # Final dispatcher recommendation
    print("\n  === Recommended dispatch table (microseconds, lower is better) ===")
    print(f"  {'n':>10s}  {'degree':>6s}  {'numpy':>8s}  {'njit':>8s}  {'njit_par':>9s}  "
          f"{'cupy':>8s}  {'cuda_ker':>9s}  {'numb_cuda':>9s}  {'WINNER':>11s}")
    for row in results:
        cells = []
        for k in ("numpy", "njit", "njit_par", "cupy", "cuda_kernel", "numba_cuda"):
            v = row["backends"].get(k)
            cells.append(f"{v:8.1f}" if v else "       -")
        winner = min(row["backends"].items(), key=lambda kv: kv[1])[0]
        print(f"  {row['n']:>10d}  {row['nc'] - 1:>6d}  " +
              "  ".join(cells) + f"  {winner:>11s}")


if __name__ == "__main__":
    main()
