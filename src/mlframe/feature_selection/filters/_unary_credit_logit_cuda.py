"""Fused device kernel of the unary credit's logit bootstrap null: the log-likelihood of every (candidate, draw) one-feature logit, in one launch.

The credit tests a column by the maximum over its candidate forms of the deviance gain of a logit ``offset + a + b * z_f`` (``_unary_credit_logit``). Its null redraws the classes ``D`` times and refits every
candidate on every draw: ``K`` candidates times ``D`` draws of a two-parameter Newton fit over the same ``n`` rows. The first device version ran each Newton step as numpy-style passes over ``(rows, candidates,
draws)`` arrays: six passes of elementwise work and sums, each writing or reading ``n * K * D`` cells (a few hundred MB at 10000 rows, 13 candidates, 50 draws) with a launch per pass.

Here one block owns one ``(candidate, draw)`` pair: its threads walk the ``n`` rows, keep the five sums of a Newton step (the two gradient and three Hessian entries) in registers, reduce them with warp shuffles and a
single shared-memory exchange, and thread 0 solves the 2 x 2 system; after the last step the log-likelihood is summed once. Nothing of size ``n * K * D`` exists (the working set is ``K * n`` values, ``D * n`` class bytes and
a ``K * D`` result), so the null needs no row streaming: the whole call is bounded by ``K * n * itemsize + D * n`` bytes, which the caller checks against the free device memory (``fits_device``) before it offers the device
path. The last candidate row of ``z`` is all zeros, so the same kernel fits the offset-plus-intercept model that every deviance is measured against.

Win axis: many draws of many candidates over thousands of rows (the credit's 50-draw batches at 10000 rows and 13 candidates); below a few hundred thousand pair-rows the launch and the transfers dominate and the
njit twin (``_unary_credit_logit._loglik_njit``, same arithmetic, parallel over pairs) wins: the kernel-tuning entry ``unary_credit_logit_null`` picks per host.

``{REAL}`` types the per-row terms and the Newton iterates (the crit dtype); ``{ACC}`` the log-likelihood that the deviance compares with a threshold.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry
from mlframe.utils.log_throttle import log_throttle
from pyutilz.performance.gpu_hw_launch import block_size_for

from ._fe_usability_signal import _crit_np_dtype

logger = logging.getLogger(__name__)

_SRC = r"""
typedef {REAL} real_t;
typedef {ACC} acc_t;
#define MAX_WARPS 32

__device__ __forceinline__ real_t warp_sum(real_t v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ acc_t warp_sum_acc(acc_t v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
// coalescing-ok: the threads of a block read consecutive rows of one candidate row of z, of the offset and of one draw row of y (stride-1 reads); the result is one value per block
extern "C" __global__ void logit_null_loglik(const real_t* __restrict__ offset, const real_t* __restrict__ z, const unsigned char* __restrict__ y, const int n, const int steps,
                                             acc_t* __restrict__ ll) {
    const int k = blockIdx.x;  // candidate row of z; the last row is all zeros: the offset-plus-intercept model
    const int d = blockIdx.y;  // draw
    const real_t* __restrict__ zk = z + (size_t)k * n;
    const unsigned char* __restrict__ yd = y + (size_t)d * n;
    __shared__ real_t red[5][MAX_WARPS];
    __shared__ acc_t red_acc[MAX_WARPS];
    __shared__ real_t tot[5];
    __shared__ real_t ab[2];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (threadIdx.x == 0) { ab[0] = (real_t)0; ab[1] = (real_t)0; }
    __syncthreads();
    for (int s = 0; s < steps; ++s) {
        const real_t a = ab[0], b = ab[1];
        real_t g_a = 0, g_b = 0, h_aa = 0, h_ab = 0, h_bb = 0;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const real_t zi = zk[i];
            const real_t eta = offset[i] + a + b * zi;
            const real_t p = (real_t)0.5 * ((real_t)1 + tanh((real_t)0.5 * eta));
            const real_t r = (real_t)yd[i] - p;
            const real_t w = p * ((real_t)1 - p);
            g_a += r; g_b += r * zi; h_aa += w; h_ab += w * zi; h_bb += w * zi * zi;
        }
        g_a = warp_sum(g_a); g_b = warp_sum(g_b); h_aa = warp_sum(h_aa); h_ab = warp_sum(h_ab); h_bb = warp_sum(h_bb);
        if (lane == 0) { red[0][warp] = g_a; red[1][warp] = g_b; red[2][warp] = h_aa; red[3][warp] = h_ab; red[4][warp] = h_bb; }
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (int v = 0; v < 5; ++v) {
                const real_t t = warp_sum(lane < nwarps ? red[v][lane] : (real_t)0);
                if (lane == 0) tot[v] = t;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const real_t hess_a = tot[2] + (real_t)1e-9, hess_ab = tot[3], hess_b = tot[4] + (real_t)1e-9;
            real_t det = hess_a * hess_b - hess_ab * hess_ab;
            if (fabs(det) < (real_t)1e-12) det = (real_t)1e-12;
            ab[0] = a + (hess_b * tot[0] - hess_ab * tot[1]) / det;
            ab[1] = b + (hess_a * tot[1] - hess_ab * tot[0]) / det;
        }
        __syncthreads();
    }
    const real_t a = ab[0], b = ab[1];
    acc_t acc = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const real_t eta = offset[i] + a + b * zk[i];
        const real_t softplus = fmax(eta, (real_t)0) + log1p(exp(-fabs(eta)));
        acc += (acc_t)((real_t)yd[i] * eta - softplus);
    }
    acc = warp_sum_acc(acc);
    if (lane == 0) red_acc[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const acc_t t = warp_sum_acc(lane < nwarps ? red_acc[lane] : (acc_t)0);
        if (lane == 0) ll[(size_t)k * gridDim.y + d] = t;
    }
}
"""

_MODULES: dict = {}
_LOCK = threading.Lock()
#: A block wider than this gains nothing on a few thousand rows and shrinks the number of resident blocks.
_MAX_BLOCK = 256


def _module(cp: Any, real: str, acc: str) -> Any:
    """The compiled kernel for one value type and one accumulator type, built once per process."""
    with _LOCK:
        mod = _MODULES.get((real, acc))
        if mod is None:
            mod = cp.RawModule(code=_SRC.replace("{REAL}", real).replace("{ACC}", acc))
            _MODULES[(real, acc)] = mod
    return mod


def fits_device(cp: Any, n: int, n_rows_z: int, n_draws: int, itemsize: int) -> bool:
    """Whether the call's working set (``n_rows_z`` candidate rows of ``n`` values, the offset, ``n_draws`` rows of class bytes and the result) fits a quarter of the free device memory."""
    need = (int(n_rows_z) + 1) * int(n) * int(itemsize) + int(n_draws) * int(n) + int(n_rows_z) * int(n_draws) * 8
    return need * 4 <= int(cp.cuda.runtime.memGetInfo()[0])


def loglik_device(cp: Any, offset: Any, z: Any, y: Any, steps: int) -> Any:
    """The log-likelihood of the logit ``offset + a + b * z[k]`` of every candidate row ``k`` of ``z`` on every class row ``d`` of ``y``, fitted by ``steps`` Newton steps: a ``(len(z), len(y))`` float64 device array.

    ``offset`` is ``(n,)``, ``z`` ``(K, n)`` C-contiguous and ``y`` ``(D, n)`` of 0 / 1 bytes (a bool array is read as it is), ``offset`` and ``z`` of the crit dtype (float32 or float64). Row ``K - 1`` of ``z`` is
    expected to be zeros when the caller wants the offset-plus-intercept model (see the module docstring)."""
    real = "float" if z.dtype == np.float32 else "double"
    k_rows, n = int(z.shape[0]), int(z.shape[1])
    n_draws = int(y.shape[0])
    out = cp.full((k_rows, n_draws), cp.nan, dtype=cp.float64)  # a sentinel: a pair the kernel misses must not look like a log-likelihood
    kernel = _module(cp, real, "double").get_function("logit_null_loglik")
    block = min(_MAX_BLOCK, block_size_for(kernel, work_items=n, max_width=_MAX_BLOCK))
    kernel((k_rows, n_draws), (block,), (offset, z, y.view(cp.uint8), np.int32(n), np.int32(steps), out))
    return out


def draw_classes(cp: Any, rng: Any, p: Any, n_draws: int) -> Any:
    """``n_draws`` class vectors (bytes, ``(n_draws, n)``) drawn from the per-row probabilities ``p`` (``(n,)``) by the device generator ``rng``."""
    return (rng.random((int(n_draws), int(p.shape[0])), dtype=p.dtype) < p[None, :]).view(cp.uint8)


def try_loglik_device(cp: Any, offset: Any, z: Any, y: Any, steps: int) -> "Any | None":
    """:func:`loglik_device`, or ``None`` (with a throttled warning) when the device cannot run it, so the caller keeps the host kernel."""
    try:
        return loglik_device(cp, offset, z, y, steps)
    except Exception as exc:  # out of device memory or a driver error: the njit twin gives the same log-likelihoods
        log_throttle(logger, "unary_credit_logit_cuda.launch", logging.WARNING, "logit credit: the device kernel failed (%r); the host kernel runs", exc)
        return None


# ---- the host twin: the same fits as one njit kernel, parallel over (candidate, draw) pairs --------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, parallel=True)
def _loglik_njit(offset, z, y, steps):
    """CPU kernel of the null: the log-likelihood of the logit ``offset + a + b * z[k]`` on every class row ``y[d]``, fitted by ``steps`` Newton steps (the arithmetic of ``logit_null_loglik``, in double precision
    sums). One thread per (candidate, draw) pair, no ``(rows, candidates, draws)`` array: a ``(len(z), len(y))`` float64 result."""
    k_rows, n = z.shape
    n_draws = y.shape[0]
    out = np.empty((k_rows, n_draws), dtype=np.float64)
    for t in prange(k_rows * n_draws):
        k = t // n_draws
        d = t % n_draws
        a = 0.0
        b = 0.0
        for _ in range(steps):
            g_a = 0.0
            g_b = 0.0
            h_aa = 0.0
            h_ab = 0.0
            h_bb = 0.0
            for i in range(n):
                zi = float(z[k, i])
                p = 0.5 * (1.0 + np.tanh(0.5 * (float(offset[i]) + a + b * zi)))
                r = float(y[d, i]) - p
                w = p * (1.0 - p)
                g_a += r
                g_b += r * zi
                h_aa += w
                h_ab += w * zi
                h_bb += w * zi * zi
            hess_a = h_aa + 1e-9
            hess_b = h_bb + 1e-9
            det = hess_a * hess_b - h_ab * h_ab
            if abs(det) < 1e-12:
                det = 1e-12
            a_new = a + (hess_b * g_a - h_ab * g_b) / det
            b = b + (hess_a * g_b - h_ab * g_a) / det
            a = a_new
        acc = 0.0
        for i in range(n):
            eta = float(offset[i]) + a + b * float(z[k, i])
            acc += float(y[d, i]) * eta - (max(eta, 0.0) + np.log1p(np.exp(-abs(eta))))
        out[k, d] = acc
    return out


def loglik_host(offset: np.ndarray, z: np.ndarray, y: np.ndarray, steps: int) -> np.ndarray:
    """:func:`loglik_device` on the host: ``offset`` ``(n,)``, ``z`` ``(K, n)`` and ``y`` ``(D, n)`` of 0 / 1; the ``(K, D)`` float64 log-likelihoods."""
    with parallel_kernel_entry():
        # one compiled signature (double values, byte classes): the kernel sums in double precision whatever the input type, and every signature is a separate numba cache load of about 0.8 s per process
        return _loglik_njit(np.ascontiguousarray(offset, dtype=np.float64), np.ascontiguousarray(z, dtype=np.float64), np.ascontiguousarray(y, dtype=np.uint8), int(steps))


# ---- backend choice through the kernel tuning cache -------------------------------------------------------------------------------------------------------------------------------------------------
# The kernel needs no row block or column block streaming: its working set is bounded by ``fits_device`` (a quarter of the free device memory), and a call that does not fit is answered by the host twin.
# The two backends cross over where the host's cores stop hiding the launch and the transfers of the offset, the candidate rows and the class bytes; the crossover is measured per host by the sweep below.
_SWEEP_N_SAMPLES = [2000, 10000]
_SWEEP_N_CANDIDATES = [5, 13, 25]
_SWEEP_DRAWS = 50  # the credit's draw batch (``_unary_credit_stats.BATCH``)
_SWEEP_STEPS = 5
_FALLBACK_MIN_CELLS = 100_000  # rows x candidates x draws from which the device kernel is the fallback choice until the host's sweep is stored (the smallest swept cell, 500000, is already 28 times faster on the device here)
_SALT = 1


def _fallback_choice(n_samples: int, n_candidates: int) -> str:
    """The backend while the host has no tuning-cache entry: the device kernel from ``_FALLBACK_MIN_CELLS`` pair-rows, the njit twin below it."""
    return "cuda" if int(n_samples) * int(n_candidates) * _SWEEP_DRAWS >= _FALLBACK_MIN_CELLS else "cpu"


def _sweep_inputs(dims: dict) -> tuple:
    """Synthetic ``(offset, z rows, class bytes)`` of ``dims['n_samples']`` rows and ``dims['n_candidates']`` candidates (the last row zeros) for the backend sweep."""
    rng = np.random.default_rng(0)
    n, k = int(dims["n_samples"]), int(dims["n_candidates"])
    dtype = _crit_np_dtype()
    z = np.vstack([rng.standard_normal((k, n)).astype(dtype), np.zeros((1, n), dtype=dtype)])
    offset = (0.5 * rng.standard_normal(n)).astype(dtype)
    p = 1.0 / (1.0 + np.exp(-(offset + 0.3 * z[0])))
    y = (rng.random((_SWEEP_DRAWS, n)) < p[None, :]).astype(np.uint8)
    return offset, z, y


def _run_sweep() -> list:
    """Sweep ``cpu`` (the njit twin) against ``cuda`` (the fused kernel, upload and copy back included: the credit's draws start on the host) over rows x candidates; fastest equivalent backend per cell."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def _cpu(offset: np.ndarray, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The host twin."""
        return loglik_host(offset, z, y, _SWEEP_STEPS)

    def _cuda(offset: np.ndarray, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The device kernel on uploaded copies, raising when the device cannot run it so the sweep leaves the cell to the host."""
        import cupy as cp

        out = try_loglik_device(cp, cp.asarray(offset), cp.asarray(z), cp.asarray(y), _SWEEP_STEPS)
        if out is None:
            raise RuntimeError("the device kernel declined this shape")
        return cp.asnumpy(out)

    # the two kernels do the same Newton steps in different precision (float32 terms on the device, double on the host): the values agree to about 1e-4 relative
    return sweep_backend_grid(
        {"cpu": _cpu, "cuda": _cuda},
        {"n_samples": _SWEEP_N_SAMPLES, "n_candidates": _SWEEP_N_CANDIDATES},
        _sweep_inputs,
        reference="cpu",
        repeats=3,
        equiv_rtol=1e-3,
        equiv_atol=1e-2,
    )


def _register_spec() -> Any:
    """Register the backend choice with the kernel tuning registry (discovered by ``mlframe-tune-kernels``)."""
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    return kernel_tuner(
        kernel_name="unary_credit_logit_null",
        variant_fns=(_loglik_njit,),
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_N_SAMPLES), "n_candidates": list(_SWEEP_N_CANDIDATES)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="unary_credit_logit_null",
    )


_SPEC = _register_spec()


def backend_choice(n_samples: int, n_candidates: int) -> str:
    """``"cuda"`` or ``"cpu"`` for a null batch of ``n_samples`` rows and ``n_candidates`` candidate rows, from the host's tuning-cache entry (or the fallback)."""
    try:
        return str(_SPEC.choose(n_samples=int(n_samples), n_candidates=int(n_candidates)))
    except Exception as exc:  # nosec B110 - a cache that cannot be read leaves the fallback in charge
        log_throttle(logger, "unary_credit_logit_cuda.ktc_choose_failed", logging.WARNING, "unary_credit_logit_null choose() failed (%s: %s); using the fallback", type(exc).__name__, exc)
        return _fallback_choice(n_samples, n_candidates)
