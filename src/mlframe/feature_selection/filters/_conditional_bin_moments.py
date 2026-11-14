"""Per-bin count, mean and centred sum of squares of many columns grouped by one conditioning column's bin codes: the fit-time statistic of the conditional-residual (Family B) and conditional-dispersion (Family D) features of ``_extra_fe_families`` and ``_extra_fe_families_dispersion``.

For one ``x_j`` (its bin codes) and ``p`` columns ``x_i`` the host walks every column twice (``_bin_sum_cnt_njit`` for sum and count, then ``_bin_css_njit`` for the centred sum of squares), one column at a time.
``bin_moments_gpu`` does the same for a block of columns in one launch: one block per column, the finite rows fill a shared (sum, count) histogram, the bin means come out, a second pass over the rows fills the
centred sums of squares. Sums are added as integers (a per-column power-of-two scale keeps the bound below 2^62), so the result does not depend on the order in which the threads add and a run gives the same bits as the next.
Columns go to the device in column blocks sized from the room under the pool (``_device_blocks``), halved when a block does not fit. Host or device is chosen per call from the kernel-tuning entry ``cond_bin_moments_backend``
by the work ``n * p``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["bin_moments_host", "bin_moments_gpu", "bin_moments_dispatch", "moments_device_chosen"]

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void cond_bin_moments(const {REAL}* __restrict__ X, const int* __restrict__ codes, const long long n, const int nb, const {ACC}* __restrict__ scales,
                      {ACC}* __restrict__ cnt_out, {ACC}* __restrict__ mean_out, {ACC}* __restrict__ css_out) {
    extern __shared__ unsigned long long sh[];                // fixed-point sum[nb], count[nb], css[nb]; then mean[nb] ({ACC})
    unsigned long long* s = sh;
    unsigned long long* c = sh + nb;
    unsigned long long* q = sh + 2 * nb;
    {ACC}* mean = ({ACC}*)(sh + 3 * nb);
    const int col = blockIdx.x;
    const {REAL}* x = X + (long long)col * n;
    const {ACC} fs = scales[2 * col], fq = scales[2 * col + 1];
    for (int b = threadIdx.x; b < 3 * nb; b += blockDim.x) sh[b] = 0ULL;
    __syncthreads();
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const {ACC} v = ({ACC})x[i];
        if (isfinite(v)) {
            const int b = codes[i];
            atomicAdd(&s[b], (unsigned long long)llrint(v * fs));
            atomicAdd(&c[b], 1ULL);
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nb; b += blockDim.x) mean[b] = c[b] > 0ULL ? ({ACC})(long long)s[b] / fs / ({ACC})c[b] : 0.0;
    __syncthreads();
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const {ACC} v = ({ACC})x[i];
        if (isfinite(v)) {
            const int b = codes[i];
            const {ACC} d = v - mean[b];
            atomicAdd(&q[b], (unsigned long long)llrint(d * d * fq));
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        cnt_out[(long long)col * nb + b] = ({ACC})c[b];
        mean_out[(long long)col * nb + b] = mean[b];
        css_out[(long long)col * nb + b] = ({ACC})(long long)q[b] / fq;
    }
}
""",
    real=np.float64,
)

_KERNEL: Any = None
_THREADS = 256
_BOUND = 2.0**61  # the fixed-point sums stay below this


def _kernel(cp: Any) -> Any:
    """The compiled kernel (one per process)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC, "cond_bin_moments")
    return _KERNEL


def _scales(cp: Any, block: Any, n: int) -> Any:
    """``(p, 2)`` power-of-two fixed-point scales of the sums and of the squared deviations for each column of ``block``: ``sum |x| * scale`` and ``sum (2 max|x|)^2 * scale`` stay below ``_BOUND``."""
    fin = cp.isfinite(block)
    m = cp.where(fin, cp.abs(block), 0.0).max(axis=1)
    m = cp.maximum(m, 1e-300)
    exp_s = cp.clip(cp.floor(cp.log2(_BOUND / (m * n))), -1000, 1000)
    exp_q = cp.clip(cp.floor(cp.log2(_BOUND / (4.0 * m * m * n))), -1000, 1000)
    return cp.stack([cp.exp2(exp_s), cp.exp2(exp_q)], axis=1)


def bin_moments_host(codes: np.ndarray, cols: Sequence[np.ndarray], n_bins: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(count, mean, css)`` of shape ``(p, n_bins)``: per bin of ``codes`` the finite rows of each column in ``cols``; a bin without rows has mean 0. The numba loops of the dispersion family."""
    from ._extra_fe_families import _bin_sum_count_masked_njit
    from ._extra_fe_families_dispersion import _bin_css_njit

    p = len(cols)
    cnt = np.zeros((p, n_bins))
    mean = np.zeros((p, n_bins))
    css = np.zeros((p, n_bins))
    codes64 = np.ascontiguousarray(codes, dtype=np.int64)
    for k, x in enumerate(cols):
        fin = np.isfinite(x)
        xf = np.ascontiguousarray(x[fin], dtype=np.float64)
        cf = codes64[fin]
        s, c = _bin_sum_count_masked_njit(codes64, np.ascontiguousarray(x, dtype=np.float64), fin, n_bins)
        m = np.where(c > 0.0, s / np.maximum(c, 1.0), 0.0)
        cnt[k], mean[k] = c, m
        css[k] = _bin_css_njit(xf, cf, m, n_bins)
    return cnt, mean, css


def bin_moments_gpu(codes: np.ndarray, cols: Sequence[np.ndarray], n_bins: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The device twin of ``bin_moments_host`` (``cols`` a list of host columns or a ``(p, n)`` device matrix); raises on a device fault so the caller can fall back to the host loop."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    resident = hasattr(cols, "device")  # a (p, n) device matrix of the resident path: no upload
    p = len(cols)
    n = int(codes.shape[0])
    out = (np.zeros((p, n_bins)), np.zeros((p, n_bins)), np.zeros((p, n_bins)))
    if p == 0:
        return out
    codes_d = cp.asarray(np.ascontiguousarray(codes, dtype=np.int32))
    kernel = _kernel(cp)
    shared = 4 * int(n_bins) * 8

    def block_work(lo: int, hi: int) -> int:
        """Moments of the columns ``lo:hi`` written into ``out``."""
        width = hi - lo
        xb = cp.ascontiguousarray(cols[lo:hi], dtype=cp.float64) if resident else cp.asarray(np.stack([np.ascontiguousarray(c, dtype=np.float64) for c in cols[lo:hi]]))  # Intentional float64: the statistic is formed in double, as the host does
        sc = cp.ascontiguousarray(_scales(cp, xb, n))
        cnt_d = cp.empty((width, n_bins), dtype=cp.float64)  # Intentional float64: the outputs are compared to the host at full precision
        mean_d = cp.empty((width, n_bins), dtype=cp.float64)  # Intentional float64: the outputs are compared to the host at full precision
        css_d = cp.empty((width, n_bins), dtype=cp.float64)  # Intentional float64: the outputs are compared to the host at full precision
        kernel((width,), (_THREADS,), (xb, codes_d, np.int64(n), np.int32(n_bins), sc, cnt_d, mean_d, css_d), shared_mem=shared)  # launch-ok: one block per column is the design
        out[0][lo:hi], out[1][lo:hi], out[2][lo:hi] = cp.asnumpy(cnt_d), cp.asnumpy(mean_d), cp.asnumpy(css_d)
        return width

    block = first_block(cp, n, p, 8, 3)
    run_column_blocks(cp, p, block, block_work)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 2_000_000  # column-rows (n * p) from which the device takes the moments before a sweep has run (see the profile record)
_SWEEP_N = 50_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (1, 4, 16, 64, 256)]
_SWEEP_BINS = 10
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(codes, cols)`` for a work of ``dims["n_samples"]`` column-rows: ``p`` columns over 50000 rows, some non-finite, grouped by ten bins."""
    rng = np.random.default_rng(0)
    p = max(1, int(dims["n_samples"]) // _SWEEP_N)
    codes = rng.integers(0, _SWEEP_BINS, _SWEEP_N)
    cols = [rng.standard_normal(_SWEEP_N) * (1 + k) + 0.1 * codes for k in range(p)]
    cols[0][::11] = np.nan
    return (codes, cols)


def _sweep_host(codes: np.ndarray, cols: list) -> np.ndarray:
    """The numba loops, as the comparison vector of the sweep."""
    return np.concatenate([a.ravel() for a in bin_moments_host(codes, cols, _SWEEP_BINS)])


def _sweep_device(codes: np.ndarray, cols: list) -> np.ndarray:
    """The device kernel; a failed launch raises, so the sweep never records a host fallback as a device time."""
    return np.concatenate([a.ravel() for a in bin_moments_gpu(codes, cols, _SWEEP_BINS)])


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="cond_bin_moments_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-9),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="cond_bin_moments_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for cond_bin_moments_backend failed: %s", e)
    _SPEC = None


def moments_device_chosen(n: int, p: int) -> bool:
    """Whether ``p`` columns over ``n`` rows take their moments on the device: the per-host cache by the work ``n * p``, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(p)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("cond_bin_moments_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= DEVICE_FALLBACK_ROWS


def bin_moments_dispatch(codes: np.ndarray, cols: Sequence[np.ndarray], n_bins: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(count, mean, css)`` per column and bin: on the device when the tuned choice says so and a device is usable, else the numba loops. A device fault falls back to the host."""
    if len(cols) and (hasattr(cols, "device") or moments_device_chosen(codes.shape[0], len(cols))):  # data already on the device never pays the upload
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                return bin_moments_gpu(codes, cols, n_bins)
        except Exception as exc:  # a device fault must not lose the fit
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "cond_bin_moments.device_failed", logging.WARNING, "conditional bin moments device path failed (%s: %s); host loop", type(exc).__name__, exc)
    if hasattr(cols, "device"):
        cols = [c for c in cols.get()]
    return bin_moments_host(codes, cols, n_bins)
