"""Device twin of ``_orthogonal_dcor_fe._dcor_columns_kernel``: ``dcov2[j] = mean(a_j * B)`` and ``dvar2_x[j]`` of the double-centred distances of every column.

One block per column, the column's subsample in shared memory; thread ``i`` owns the rows ``i`` it strides over and accumulates the row sum, ``sum d^2``, ``sum d B[i, q]``
and ``(row / m)^2``; the block reduces the four totals and the host forms the same expressions as the host kernel. Everything is double: ``dvar2_x`` is
``sum d^2 - 2 m sum r^2 + m^2 g^2``, a cancellation that float32 sums would not survive (CUDA.md ``{ACC}``; the host kernel's note records the same for the dense form).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

logger = logging.getLogger(__name__)

_SRC = r"""
// dtype-ok: reviewed 2026-11-05: float32 values with double sums measured 45 ms against 46-49 ms (no gain, the double accumulations set the time); the host loads float64 on purpose (dvar2_x is a cancellation).
extern "C" __global__ void dcor_cols(const double* __restrict__ xs, const double* __restrict__ b, const int m, const int k, double* __restrict__ sums) {
    extern __shared__ double sm[];
    double* sx = sm;                     // this column's m values
    double* red = sm + m;                // 4 * blockDim.x partials
    const int j = blockIdx.x;
    if (j >= k) return;
    for (int a = threadIdx.x; a < m; a += blockDim.x) sx[a] = xs[(long long)j * m + a];
    __syncthreads();
    double total = 0.0, sq = 0.0, cross = 0.0, sum_r2 = 0.0;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const double xi = sx[i];
        const double* brow = b + (long long)i * m;
        double row = 0.0;
        for (int q = 0; q < m; ++q) {
            const double d = fabs(xi - sx[q]);
            row += d;
            sq += d * d;
            cross += d * brow[q];
        }
        total += row;
        const double r = row / m;
        sum_r2 += r * r;
    }
    const int t = threadIdx.x, nt = blockDim.x;
    red[t] = total; red[nt + t] = sq; red[2 * nt + t] = cross; red[3 * nt + t] = sum_r2;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) { red[t] += red[t + s]; red[nt + t] += red[nt + t + s]; red[2 * nt + t] += red[2 * nt + t + s]; red[3 * nt + t] += red[3 * nt + t + s]; }
        __syncthreads();
    }
    if (t == 0) { sums[j * 4] = red[0]; sums[j * 4 + 1] = red[nt]; sums[j * 4 + 2] = red[2 * nt]; sums[j * 4 + 3] = red[3 * nt]; }
}
"""

_MAX_BLOCK = 256  # widest block the kernel's dynamic partials allow
_MAX_M = 4096  # shared memory: m doubles + 4 x 256 partials (40 KB at m = 4096)
_KERNEL: list = []
_DCOR_DEVICE_MIN_WORK = 1_000_000  # k * m * m pair terms from which the device wins without a tuning-cache measurement (GTX 1050 Ti)
_DCOR_CODE_VERSION = "dcor_columns_backend-v1"


def _kernel() -> Any:
    """The compiled kernel (compiled once)."""
    import cupy as cp

    if not _KERNEL:
        _KERNEL.append(cp.RawKernel(_SRC, "dcor_cols"))
    return _KERNEL[0]


def dcor_columns_gpu(xs: np.ndarray, b: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """``(dcov2, dvar2_x)`` of every row of ``xs`` ``(k, m)`` against the double-centred ``b`` ``(m, m)``, as ``_dcor_columns_kernel`` fills them; ``None`` on a device
    fault or a subsample over ``_MAX_M`` (the caller runs the host kernel). The columns go up in blocks bounded by a quarter of the free VRAM."""
    try:
        import cupy as cp

        k, m = int(xs.shape[0]), int(xs.shape[1])
        # Intentional float64: dCor moments in double (cancellation in dvar2_x)
        dcov2, dvar2 = np.empty(k, dtype=np.float64), np.empty(k, dtype=np.float64)
        if k == 0:
            return dcov2, dvar2
        if m < 1 or m > _MAX_M:
            return None
        kern = _kernel()
        b_d = cp.asarray(np.ascontiguousarray(b, dtype=np.float64))  # Intentional float64: dCor moments in double
        free = int(cp.cuda.Device().mem_info[0])
        width = max(1, min(k, (free // 4) // max(1, m * 8)))
        # the reduction halves blockDim.x over 4 dynamic partials per thread: a power of two; 256 keeps m doubles + the partials within the 48 KB block limit at m = 4096
        threads = block_size_for(kern, dyn_smem=(m + 4 * _MAX_BLOCK) * 8, work_items=m, pow2=True, max_width=_MAX_BLOCK)
        shared = (m + 4 * threads) * 8
        mm = float(m) * float(m)
        for start in range(0, k, width):
            stop = min(k, start + width)
            xs_d = cp.asarray(np.ascontiguousarray(xs[start:stop], dtype=np.float64))  # Intentional float64: dCor moments in double
            sums = cp.empty((stop - start) * 4, dtype=cp.float64)  # Intentional float64: dCor moments in double
            kern((stop - start,), (threads,), (xs_d, b_d, np.int32(m), np.int32(stop - start), sums), shared_mem=shared)
            s = sums.get().reshape(-1, 4)
            total, sq, cross, sum_r2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            g = total / mm
            dcov2[start:stop] = cross / mm
            dvar2[start:stop] = (sq - 2.0 * m * sum_r2 + mm * g * g) / mm  # moment-ok: distances are translation-invariant, as in the host kernel
        return dcov2, dvar2
    except Exception as exc:
        logger.debug("device dCor not used (%s: %s); host kernel", type(exc).__name__, exc)
        return None


def dcor_backend(m: int, k: int) -> str:
    """'device' or 'cpu' for one dCor batch, from the per-host kernel-tuning cache (async sweep), else the work crossover; 'cpu' without a usable GPU or with
    ``MLFRAME_DCOR_DEVICE=0``."""
    import os

    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled() or os.environ.get("MLFRAME_DCOR_DEVICE", "1").strip() == "0":
        return "cpu"
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return "cpu"
    except Exception as _silent_exc:
        logger.debug("_dcor_gpu.py: suppressed %r", _silent_exc)
        return "cpu"
    fallback = "device" if int(k) * int(m) * int(m) >= _DCOR_DEVICE_MIN_WORK else "cpu"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        res = KernelTuningCache.load_or_create().get_or_tune(
            "dcor_columns_backend",
            dims={"m": int(m), "k": int(k)},
            tuner=_run_dcor_sweep,
            axes=["m", "k"],
            fallback={"kernel_choice": fallback},
            code_version=_DCOR_CODE_VERSION,
            async_sweep=True,
        )
        if isinstance(res, str):
            return res
        if res:
            return str(res.get("kernel_choice", fallback))
    except Exception as exc:
        logger.debug("dCor backend selection failed: %r", exc)
    return fallback


def _run_dcor_sweep():
    """Per-host host-vs-device sweep of the dCor batch over (m, k); [] when the sweep helper is unavailable."""
    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
    except Exception as exc:
        logger.debug("dCor sweep unavailable (%s: %s); no regions recorded", type(exc).__name__, exc)
        return []
    from ._orthogonal_dcor_fe import _dcor_columns_kernel, _u_centered_distance_matrix

    def _cpu(xs, b):
        """Host kernel."""
        dcov2, dvar2 = np.empty(xs.shape[0]), np.empty(xs.shape[0])
        _dcor_columns_kernel(xs, b, dcov2, dvar2)
        return np.concatenate([dcov2, dvar2])

    def _device(xs, b):
        """Device kernel, uploads included."""
        r = dcor_columns_gpu(xs, b)
        return None if r is None else np.concatenate(r)

    def _make_inputs(dims):
        """Random columns and a double-centred binary-label distance matrix."""
        m, k = int(dims["m"]), int(dims["k"])
        rng = np.random.default_rng(0)
        return (np.ascontiguousarray(rng.standard_normal((k, m))), np.ascontiguousarray(_u_centered_distance_matrix(rng.integers(0, 2, m).astype(np.float64))))

    return list(
        sweep_backend_grid(
            {"cpu": _cpu, "device": _device},
            {"m": [200, 500, 1000], "k": [8, 100, 400]},
            _make_inputs,
            reference="cpu",
            repeats=2,
            equiv_rtol=1e-9,
            equiv_atol=1e-12,
            decision_key="kernel_choice",
        )
        or []
    )
