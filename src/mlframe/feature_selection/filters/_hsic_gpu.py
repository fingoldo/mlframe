"""Device twin of ``_orthogonal_hsic_fe._hsic_biased_columns_njit``: biased HSIC of many columns against one centred y-side Gram matrix.

``out[j] = (trace(Lc) + 2 * sum_{a<b} exp(-(x_a - x_b)^2 / (2 s_j^2)) * Lc[a, b]) / denom`` with ``s_j`` the median pairwise distance of column ``j`` (1 for a constant
column), as the host kernel computes it. One block per column: the column's values sit in shared memory, each thread sums the rows ``a`` it owns over ``b > a``
and the block reduces. ``{REAL}`` is the type of the values and the exponent (``crit_float_dtype``: float32 by default); the sums are double
(``{ACC}``, CUDA.md): the scores are ranked against each other; the centred Gram matrix stays double (its rows sum to 0, which a
flat column's score relies on). The bandwidths are computed on the host (O(m log m) per column).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

logger = logging.getLogger(__name__)

_SRC = r"""
extern "C" __global__ void hsic_biased_cols(const {REAL}* __restrict__ xs, const double* __restrict__ lc, const double* __restrict__ inv_two_s2,
                                            const int m, const int k, double* __restrict__ tot_out) {
    extern __shared__ unsigned char smem[];
    {REAL}* sx = ({REAL}*)smem;                                  // this column's m values
    double* red = (double*)(smem + ((m * sizeof({REAL}) + 7) / 8) * 8);
    const int j = blockIdx.x;
    if (j >= k) return;
    for (int a = threadIdx.x; a < m; a += blockDim.x) sx[a] = xs[(long long)j * m + a];
    __syncthreads();
    const {REAL} g = ({REAL})inv_two_s2[j];
    double acc = 0.0;
    for (int a = threadIdx.x; a < m - 1; a += blockDim.x) {
        const {REAL} xa = sx[a];
        const double* row = lc + (long long)a * m;
        double part = 0.0;
        for (int b = a + 1; b < m; ++b) {
            const {REAL} d = xa - sx[b];
            part += (double)EXPF(-d * d * g) * row[b];   // Lc stays double: its rows sum to 0, which float32 rounding would break for flat columns
        }
        acc += part;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) tot_out[j] = red[0];
}
"""

_MAX_BLOCK = 256  # widest block the kernel's dynamic scratch allows
_MAX_M = 4096  # the column's values live in shared memory (16 KB float32 / 32 KB double)
_MODULES: dict = {}
_MODULES_LOCK = threading.Lock()
# k * m * m pair terms from which the device wins when the kernel-tuning cache has no measurement on this host (GTX 1050 Ti, m=500: about 5 columns)
_HSIC_DEVICE_MIN_WORK = 1_000_000
_HSIC_CODE_VERSION = "hsic_biased_backend-v1"


def _hsic_kernel(real: str) -> Any:
    """The compiled kernel for ``real`` ("float" / "double"), one module per type."""
    import cupy as cp

    with _MODULES_LOCK:
        kern = _MODULES.get(real)
        if kern is None:
            src = _SRC.replace("{REAL}", real).replace("EXPF", "expf" if real == "float" else "exp")
            kern = _MODULES[real] = cp.RawKernel(src, "hsic_biased_cols")
    return kern


def hsic_biased_columns_gpu(xs_rows: np.ndarray, lc: np.ndarray, sigmas: np.ndarray, denom: float) -> Optional[np.ndarray]:
    """Biased HSIC of every row of ``xs_rows`` ``(k, m)`` against the centred ``lc`` ``(m, m)`` with bandwidths ``sigmas`` ``(k,)``; ``None`` on any device fault or
    a subsample over ``_MAX_M`` (the caller runs the host kernel). The columns go up in blocks bounded by a quarter of the free VRAM."""
    try:
        import cupy as cp

        from ._fe_gpu_batch._devices import crit_float_dtype

        k, m = int(xs_rows.shape[0]), int(xs_rows.shape[1])
        if k == 0:
            return np.empty(0, dtype=np.float64)  # Intentional float64: HSIC score vector (output contract of the host kernel)
        if m < 2 or m > _MAX_M:
            return None
        dt = crit_float_dtype()
        real = "float" if dt == cp.float32 else "double"
        kern = _hsic_kernel(real)
        lc64 = np.ascontiguousarray(lc, dtype=np.float64)  # Intentional float64: the trace is summed in double, as the host kernel sums it
        diag = float(np.trace(lc64))
        lc_d = cp.asarray(lc64)
        s = np.where(sigmas > 0.0, sigmas, 1.0)
        inv = np.ascontiguousarray(1.0 / (2.0 * s * s), dtype=np.float64)  # Intentional float64: the bandwidth term is formed in double, as on the host
        free = int(cp.cuda.Device().mem_info[0])
        width = max(1, min(k, (free // 4) // max(1, m * np.dtype(dt).itemsize)))
        values_bytes = ((m * np.dtype(dt).itemsize + 7) // 8) * 8
        # the reduction halves blockDim.x over a dynamic scratch of one double per thread: a power of two; 256 keeps the scratch within the 48 KB block limit at m = 4096
        threads = block_size_for(kern, dyn_smem=values_bytes + _MAX_BLOCK * 8, work_items=m, pow2=True, max_width=_MAX_BLOCK)
        shared = values_bytes + threads * 8
        out = np.empty(k, dtype=np.float64)  # Intentional float64: HSIC score vector (output contract of the host kernel)
        for start in range(0, k, width):
            stop = min(k, start + width)
            # Centre each row in float64 BEFORE narrowing: the kernel uses only pairwise differences, which a row offset does not change, but a
            # float32 offset of 1e6 has a step of 0.06 and snaps a 0.01-spread column.
            rows_c = np.asarray(xs_rows[start:stop], dtype=np.float64)  # Intentional float64: the kernel matrices are double, as the HSIC reference
            rows_c = rows_c - rows_c.mean(axis=1, keepdims=True)
            xs_d = cp.asarray(np.ascontiguousarray(rows_c).astype(dt, copy=False))
            tot = cp.empty(stop - start, dtype=cp.float64)  # Intentional float64: {ACC} sums
            kern((stop - start,), (threads,), (xs_d, lc_d, cp.asarray(inv[start:stop]), np.int32(m), np.int32(stop - start), tot), shared_mem=shared)
            out[start:stop] = (diag + 2.0 * tot.get()) / float(denom)
        out[(out < 0.0) & (out > -1e-12)] = 0.0
        return out
    except Exception as exc:
        logger.debug("device HSIC not used (%s: %s); host kernel", type(exc).__name__, exc)
        return None


def hsic_backend(m: int, k: int) -> str:
    """'device' or 'cpu' for one biased-HSIC batch, from the per-host kernel-tuning cache (async sweep), else the work crossover; 'cpu' without a usable GPU
    or with ``MLFRAME_HSIC_DEVICE=0``."""
    import os

    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled() or os.environ.get("MLFRAME_HSIC_DEVICE", "1").strip() == "0":
        return "cpu"
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return "cpu"
    except Exception as _silent_exc:
        logger.debug("_hsic_gpu.py: suppressed %r", _silent_exc)
        return "cpu"
    fallback = "device" if int(k) * int(m) * int(m) >= _HSIC_DEVICE_MIN_WORK else "cpu"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        res = KernelTuningCache.load_or_create().get_or_tune(
            "hsic_biased_backend",
            dims={"m": int(m), "k": int(k)},
            tuner=_run_hsic_sweep,
            axes=["m", "k"],
            fallback={"kernel_choice": fallback},
            code_version=_HSIC_CODE_VERSION,
            async_sweep=True,
        )
        if isinstance(res, str):
            return res
        if res:
            return str(res.get("kernel_choice", fallback))
    except Exception as exc:
        logger.debug("HSIC backend selection failed: %r", exc)
    return fallback


def _run_hsic_sweep():
    """Per-host host-vs-device sweep of the biased HSIC batch over (m, k); [] when the sweep helper is unavailable."""
    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
    except Exception as exc:
        logger.debug("HSIC sweep unavailable (%s: %s); no regions recorded", type(exc).__name__, exc)
        return []
    from ._orthogonal_hsic_fe import _hsic_biased_columns_njit, hsic_column_sigmas

    def _cpu(xs, lc, denom):
        """Host kernel."""
        out = np.empty(xs.shape[0], dtype=np.float64)  # Intentional float64: HSIC score vector (output contract)
        _hsic_biased_columns_njit(xs, lc, denom, out)
        return out

    def _device(xs, lc, denom):
        """Device kernel, including the host bandwidths and the uploads."""
        return hsic_biased_columns_gpu(xs, lc, hsic_column_sigmas(xs), denom)

    def _make_inputs(dims):
        """Random columns and a centred label Gram matrix of the given size."""
        m, k = int(dims["m"]), int(dims["k"])
        rng = np.random.default_rng(0)
        xs = rng.standard_normal((k, m))
        lab = rng.integers(0, 2, m).astype(np.float64)  # Intentional float64: the kernel matrices are double, as the HSIC reference
        L = (lab[:, None] == lab[None, :]).astype(np.float64)
        H = np.eye(m) - 1.0 / m
        return (np.ascontiguousarray(xs), np.ascontiguousarray(H @ L @ H), float((m - 1) * (m - 1)))

    return list(
        sweep_backend_grid(
            {"cpu": _cpu, "device": _device},
            {"m": [200, 500, 1000], "k": [8, 100, 400]},
            _make_inputs,
            reference="cpu",
            repeats=2,
            equiv_rtol=1e-5,
            equiv_atol=1e-8,
            decision_key="kernel_choice",
        )
        or []
    )
