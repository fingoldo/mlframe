"""Bench S46 / W4b: backend ladder for the EWMA and frac-diff-inverse
recurrences in ``mlframe.training.composite_transforms``.

Both kernels are LEFT-RECURRENT in row order: ``out[i] = f(out[i-1], ...)``.
That rules out ``numba.prange`` over rows. The win comes from a BATCHED
kernel ``(K, N)`` that runs K specs in parallel -- each spec's row
recurrence stays serial but K parallel threads share the heavy lifting.

Variants:

* ``v1_njit_single``       -- production: per-spec call into the scalar-args njit kernel.
* ``v2_njit_par_batched``  -- (K, N) kernel with ``prange`` over K (batched anchor + alpha per spec).
* ``v3_cuda_batched``      -- ``cp.RawKernel`` with one CUDA block per spec; threads inside a block cooperate over N (each handles its own k slice and the warp leader reduces -- here just one thread per spec since the recurrence is fully serial). Only built when cupy + a CUDA device are available.

Output: prints a table of (variant, n, k, ms) plus a JSON dump in ``_results/``.

Run:
    D:/ProgramData/anaconda3/python.exe -c "import sys; sys.path.insert(0,'src'); from mlframe.training._benchmarks import bench_ewma_frac_diff_backends as B; B.run()"
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import numba as _nb
    _HAS_NB = True
except Exception:
    _nb = None
    _HAS_NB = False

try:
    import cupy as _cp
    _HAS_CP = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None
    _HAS_CP = False

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "_results"
_RESULTS.mkdir(exist_ok=True)


# ---------------- EWMA kernels ----------------

if _HAS_NB:
    @_nb.njit(cache=True)
    def _ewma_v1_njit_kernel(base_f, alpha, anchor):
        n = base_f.size
        out = np.empty(n, dtype=np.float64)
        state = anchor
        for i in range(n):
            x = base_f[i]
            if np.isfinite(x):
                state = (1.0 - alpha) * state + alpha * x
            out[i] = state
        return out

    @_nb.njit(cache=True, parallel=True)
    def _ewma_v2_njit_par_batched_kernel(base_batch, alphas, anchors):
        K, N = base_batch.shape
        out = np.empty((K, N), dtype=np.float64)
        for s in _nb.prange(K):
            state = anchors[s]
            a = alphas[s]
            for i in range(N):
                x = base_batch[s, i]
                if np.isfinite(x):
                    state = (1.0 - a) * state + a * x
                out[s, i] = state
        return out


def _ewma_v1_njit_single(base_batch, alphas, anchors):
    K, N = base_batch.shape
    out = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        out[s] = _ewma_v1_njit_kernel(np.ascontiguousarray(base_batch[s]), float(alphas[s]), float(anchors[s]))
    return out


def _ewma_v2_njit_par_batched(base_batch, alphas, anchors):
    return _ewma_v2_njit_par_batched_kernel(
        np.ascontiguousarray(base_batch),
        np.ascontiguousarray(alphas, dtype=np.float64),
        np.ascontiguousarray(anchors, dtype=np.float64),
    )


_EWMA_CUDA_KERNEL = None


def _ensure_ewma_cuda_kernel():
    global _EWMA_CUDA_KERNEL
    if _EWMA_CUDA_KERNEL is not None or not _HAS_CP:
        return _EWMA_CUDA_KERNEL
    src = r"""
    extern "C" __global__
    void ewma_batched(const double* __restrict__ base,
                      const double* __restrict__ alphas,
                      const double* __restrict__ anchors,
                      double* __restrict__ out,
                      int K, int N) {
        int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= K) return;
        double state = anchors[s];
        double a = alphas[s];
        long long row_off = (long long)s * (long long)N;
        for (int i = 0; i < N; ++i) {
            double x = base[row_off + i];
            if (isfinite(x)) {
                state = (1.0 - a) * state + a * x;
            }
            out[row_off + i] = state;
        }
    }
    """
    _EWMA_CUDA_KERNEL = _cp.RawKernel(src, "ewma_batched")
    return _EWMA_CUDA_KERNEL


def _ewma_v3_cuda_batched(base_batch, alphas, anchors):
    if not _HAS_CP:
        return _ewma_v1_njit_single(base_batch, alphas, anchors)
    kernel = _ensure_ewma_cuda_kernel()
    K, N = base_batch.shape
    d_base = _cp.asarray(base_batch, dtype=_cp.float64)
    d_alphas = _cp.asarray(alphas, dtype=_cp.float64)
    d_anchors = _cp.asarray(anchors, dtype=_cp.float64)
    d_out = _cp.empty((K, N), dtype=_cp.float64)
    block = 32
    grid = (K + block - 1) // block
    kernel((grid,), (block,), (d_base, d_alphas, d_anchors, d_out, np.int32(K), np.int32(N)))
    _cp.cuda.runtime.deviceSynchronize()
    return _cp.asnumpy(d_out)


# ---------------- frac_diff_inverse kernels ----------------

if _HAS_NB:
    @_nb.njit(cache=True)
    def _frac_diff_inverse_v1_njit_kernel(t_f, lags, weights, anchor):
        n = t_f.size
        out = np.empty(n, dtype=np.float64)
        inv_w0 = 1.0 / weights[0]
        for i in range(n):
            lag_sum = 0.0
            upper = min(i + 1, lags + 1)
            for k_idx in range(1, upper):
                lag_sum += weights[k_idx] * out[i - k_idx]
            for k_idx in range(upper, lags + 1):
                lag_sum += weights[k_idx] * anchor
            out[i] = (t_f[i] - lag_sum) * inv_w0
        return out

    @_nb.njit(cache=True, parallel=True)
    def _frac_diff_inverse_v2_njit_par_batched_kernel(t_batch, lags, weights_batch, anchors):
        K, N = t_batch.shape
        out = np.empty((K, N), dtype=np.float64)
        for s in _nb.prange(K):
            inv_w0 = 1.0 / weights_batch[s, 0]
            anchor = anchors[s]
            for i in range(N):
                lag_sum = 0.0
                upper = min(i + 1, lags + 1)
                for k_idx in range(1, upper):
                    lag_sum += weights_batch[s, k_idx] * out[s, i - k_idx]
                for k_idx in range(upper, lags + 1):
                    lag_sum += weights_batch[s, k_idx] * anchor
                out[s, i] = (t_batch[s, i] - lag_sum) * inv_w0
        return out


def _frac_diff_inverse_v1_njit_single(t_batch, lags, weights_batch, anchors):
    K, N = t_batch.shape
    out = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        out[s] = _frac_diff_inverse_v1_njit_kernel(
            np.ascontiguousarray(t_batch[s]),
            int(lags),
            np.ascontiguousarray(weights_batch[s]),
            float(anchors[s]),
        )
    return out


def _frac_diff_inverse_v2_njit_par_batched(t_batch, lags, weights_batch, anchors):
    return _frac_diff_inverse_v2_njit_par_batched_kernel(
        np.ascontiguousarray(t_batch),
        int(lags),
        np.ascontiguousarray(weights_batch),
        np.ascontiguousarray(anchors, dtype=np.float64),
    )


_FRAC_DIFF_CUDA_KERNEL = None


def _ensure_frac_diff_cuda_kernel():
    global _FRAC_DIFF_CUDA_KERNEL
    if _FRAC_DIFF_CUDA_KERNEL is not None or not _HAS_CP:
        return _FRAC_DIFF_CUDA_KERNEL
    src = r"""
    extern "C" __global__
    void frac_diff_inv_batched(const double* __restrict__ t_batch,
                                const double* __restrict__ weights_batch,
                                const double* __restrict__ anchors,
                                double* __restrict__ out,
                                int K, int N, int lags) {
        int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= K) return;
        long long row_off = (long long)s * (long long)N;
        long long w_off = (long long)s * (long long)(lags + 1);
        double inv_w0 = 1.0 / weights_batch[w_off + 0];
        double anchor = anchors[s];
        for (int i = 0; i < N; ++i) {
            double lag_sum = 0.0;
            int upper = (i + 1) < (lags + 1) ? (i + 1) : (lags + 1);
            for (int k = 1; k < upper; ++k) {
                lag_sum += weights_batch[w_off + k] * out[row_off + (i - k)];
            }
            for (int k = upper; k < lags + 1; ++k) {
                lag_sum += weights_batch[w_off + k] * anchor;
            }
            out[row_off + i] = (t_batch[row_off + i] - lag_sum) * inv_w0;
        }
    }
    """
    _FRAC_DIFF_CUDA_KERNEL = _cp.RawKernel(src, "frac_diff_inv_batched")
    return _FRAC_DIFF_CUDA_KERNEL


def _frac_diff_inverse_v3_cuda_batched(t_batch, lags, weights_batch, anchors):
    if not _HAS_CP:
        return _frac_diff_inverse_v1_njit_single(t_batch, lags, weights_batch, anchors)
    kernel = _ensure_frac_diff_cuda_kernel()
    K, N = t_batch.shape
    d_t = _cp.asarray(t_batch, dtype=_cp.float64)
    d_w = _cp.asarray(weights_batch, dtype=_cp.float64)
    d_anchors = _cp.asarray(anchors, dtype=_cp.float64)
    d_out = _cp.empty((K, N), dtype=_cp.float64)
    block = 32
    grid = (K + block - 1) // block
    kernel((grid,), (block,), (d_t, d_w, d_anchors, d_out, np.int32(K), np.int32(N), np.int32(lags)))
    _cp.cuda.runtime.deviceSynchronize()
    return _cp.asnumpy(d_out)


# ---------------- Bench harness ----------------

def _frac_diff_weights(d: float, lags: int) -> np.ndarray:
    w = np.empty(lags + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, lags + 1):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def _time_call(fn, *args, repeat: int = 5) -> float:
    fn(*args)  # warm
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        dt = (time.perf_counter() - t0) * 1000.0
        if dt < best:
            best = dt
    return best


def run() -> dict:
    sizes = [
        (1, 10_000), (1, 100_000), (1, 1_000_000),
        (10, 100_000), (10, 1_000_000),
        (100, 100_000),
    ]
    rng = np.random.default_rng(13)
    out = {"ewma": [], "frac_diff_inverse": [], "has_cupy": _HAS_CP, "has_numba": _HAS_NB}
    print(f"backends: numba={_HAS_NB} cupy={_HAS_CP}")
    for K, N in sizes:
        base = rng.standard_normal((K, N)).astype(np.float64)
        alphas = rng.uniform(0.1, 0.4, size=K).astype(np.float64)
        anchors = rng.standard_normal(K).astype(np.float64)
        t_v1 = _time_call(_ewma_v1_njit_single, base, alphas, anchors)
        t_v2 = _time_call(_ewma_v2_njit_par_batched, base, alphas, anchors)
        ref = _ewma_v1_njit_single(base, alphas, anchors)
        got_v2 = _ewma_v2_njit_par_batched(base, alphas, anchors)
        np.testing.assert_allclose(got_v2, ref, rtol=1e-12)
        if _HAS_CP:
            t_v3 = _time_call(_ewma_v3_cuda_batched, base, alphas, anchors)
            got_v3 = _ewma_v3_cuda_batched(base, alphas, anchors)
            # CUDA float64 fma path produces ~1e-12 rel error vs CPU sequential fmadd
            np.testing.assert_allclose(got_v3, ref, rtol=1e-10, atol=1e-12)
        else:
            t_v3 = float("nan")
        print(f"EWMA K={K:>3d} N={N:>8d}  v1={t_v1:8.2f}ms  v2_par={t_v2:8.2f}ms  v3_cuda={t_v3:8.2f}ms  (v2/v1: {t_v1 / max(t_v2, 1e-9):.2f}x)")
        out["ewma"].append({"K": K, "N": N, "v1_ms": round(t_v1, 3), "v2_ms": round(t_v2, 3), "v3_ms": round(t_v3, 3) if not np.isnan(t_v3) else None})

        lags = 30
        d = 0.5
        weights = _frac_diff_weights(d, lags)
        weights_batch = np.tile(weights, (K, 1))
        anchors_f = rng.standard_normal(K).astype(np.float64)
        t_batch = rng.standard_normal((K, N)).astype(np.float64)
        t_f1 = _time_call(_frac_diff_inverse_v1_njit_single, t_batch, lags, weights_batch, anchors_f)
        t_f2 = _time_call(_frac_diff_inverse_v2_njit_par_batched, t_batch, lags, weights_batch, anchors_f)
        ref_f = _frac_diff_inverse_v1_njit_single(t_batch, lags, weights_batch, anchors_f)
        got_f2 = _frac_diff_inverse_v2_njit_par_batched(t_batch, lags, weights_batch, anchors_f)
        np.testing.assert_allclose(got_f2, ref_f, rtol=1e-9, atol=1e-9)
        if _HAS_CP:
            t_f3 = _time_call(_frac_diff_inverse_v3_cuda_batched, t_batch, lags, weights_batch, anchors_f)
            got_f3 = _frac_diff_inverse_v3_cuda_batched(t_batch, lags, weights_batch, anchors_f)
            np.testing.assert_allclose(got_f3, ref_f, rtol=1e-9, atol=1e-9)
        else:
            t_f3 = float("nan")
        print(f"FDI  K={K:>3d} N={N:>8d}  v1={t_f1:8.2f}ms  v2_par={t_f2:8.2f}ms  v3_cuda={t_f3:8.2f}ms  (v2/v1: {t_f1 / max(t_f2, 1e-9):.2f}x)")
        out["frac_diff_inverse"].append({"K": K, "N": N, "v1_ms": round(t_f1, 3), "v2_ms": round(t_f2, 3), "v3_ms": round(t_f3, 3) if not np.isnan(t_f3) else None})

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = _RESULTS / f"bench_ewma_frac_diff_backends_{ts}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults: {path}")
    return out


if __name__ == "__main__":
    run()
