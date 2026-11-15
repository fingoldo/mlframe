"""Device twin of the per-category order statistics of ``_target_encoding_order_stats`` (``np.lexsort`` grouping + the njit segment sweep ``_segment_order_stats``).

The host sorts y inside each category with ``np.lexsort((y, category))`` and a single numba kernel sweeps the contiguous segments for the median, the symmetric-trimmed mean, the 10 and 90 percent quantiles,
their difference, the minimum and the maximum. On the device the same grouping is ``cupy.lexsort`` and the sweep is one kernel with one block per category: thread 0 reads the quantile and extreme positions
(numpy's ``linear`` interpolation, the same expressions), the block sums the trimmed segment with a warp-shuffle reduction. The values equal the host's (the trimmed mean to the summation order of a segment). Host or
device is chosen per call from the kernel-tuning entry ``order_stats_backend`` by rows x categories; a device fault falls back to the host with a throttled warning. The categories are one column block of
the encoder's input: a single device sort covers every category of the column.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["segment_order_stats_gpu", "order_stats_dispatch", "order_device_chosen"]

_THREADS = 128

_SRC = render(
    r"""
__device__ {ACC} seg_quantile(const {ACC}* y, const long long start, const long long k, const {ACC} p) {
    if (k == 1) return y[start];
    const {ACC} pos = ({ACC})(k - 1) * p;
    const long long lo = (long long)pos;
    const {ACC} frac = pos - ({ACC})lo;
    {ACC} v = y[start + lo];
    if (frac > 0.0 && lo + 1 < k) v += frac * (y[start + lo + 1] - v);
    return v;
}

extern "C" __global__ __launch_bounds__(128)
void segment_order_stats(const {ACC}* __restrict__ ys, const long long* __restrict__ starts, const long long* __restrict__ counts, const int flags, const {ACC} trim_frac,
                         const int n_cats, {ACC}* __restrict__ out) {
    // out[stat * n_cats + c], stat order: median, trimmed mean, q10, q90, iqr, min, max; NaN for a stat not asked for and for an empty category
    __shared__ {ACC} red[32];
    const int c = blockIdx.x;
    const long long k = counts[c];
    const long long s = starts[c];
    const {ACC} nan_v = nan("");
    if (k <= 0) {
        if (threadIdx.x == 0) for (int t = 0; t < 7; ++t) out[t * n_cats + c] = nan_v;
        return;
    }
    {ACC} part = 0.0;
    long long lo = s, hi = s + k;
    if (flags & 2) {                                       // trimmed mean: cut floor(k * trim) per tail; everything trimmed -> the untrimmed mean
        const long long cut = (long long)((({ACC})k) * trim_frac);
        lo = s + cut;
        hi = s + k - cut;
        if (hi - lo <= 0) { lo = s; hi = s + k; }
        for (long long j = lo + threadIdx.x; j < hi; j += blockDim.x) part += ys[j];
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = part;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} v10 = nan_v, v90 = nan_v;
        if (flags & 4 || flags & 16) v10 = seg_quantile(ys, s, k, 0.10);
        if (flags & 8 || flags & 16) v90 = seg_quantile(ys, s, k, 0.90);
        out[0 * n_cats + c] = (flags & 1) ? seg_quantile(ys, s, k, 0.5) : nan_v;
        {ACC} trm = nan_v;
        if (flags & 2) {
            {ACC} acc = 0.0;
            for (int w = 0; w < (blockDim.x >> 5); ++w) acc += red[w];
            trm = acc / ({ACC})(hi - lo);
        }
        out[1 * n_cats + c] = trm;
        out[2 * n_cats + c] = (flags & 4) ? v10 : nan_v;
        out[3 * n_cats + c] = (flags & 8) ? v90 : nan_v;
        out[4 * n_cats + c] = (flags & 16) ? v90 - v10 : nan_v;
        out[5 * n_cats + c] = (flags & 32) ? ys[s] : nan_v;
        out[6 * n_cats + c] = (flags & 64) ? ys[s + k - 1] : nan_v;
    }
}
""",
)

_KERNEL: Any = None


def _kernel(cp: Any) -> Any:
    """The compiled kernel (one per process)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC.replace("{ACC}", "double"), "segment_order_stats")
    return _KERNEL


def kernel(cp: Any, real: Any) -> Any:
    """``segment_order_stats`` (double only: the statistics are formed in double); the register-pressure registry compiles it through this."""
    return _kernel(cp)


_STAT_BITS = ("median", "trimmed_mean", "q10", "q90", "iqr", "min", "max")


def segment_order_stats_gpu(inverse: np.ndarray, y: np.ndarray, counts: np.ndarray, wanted: Sequence[str], trim_frac: float) -> "tuple[np.ndarray, ...]":
    """``(median, trimmed_mean, q10, q90, iqr, min, max)`` per category, NaN for the stats not in ``wanted`` and for empty categories, as ``_segment_order_stats`` returns them. Raises on a device fault."""
    import cupy as cp

    n_cats = int(counts.shape[0])
    inv_d = cp.asarray(np.ascontiguousarray(inverse, dtype=np.int64))
    y_d = cp.asarray(np.ascontiguousarray(y, dtype=np.float64))  # Intentional float64: order statistics and the trimmed sum are formed in double, as on the host
    order = cp.lexsort(cp.stack([y_d, inv_d.astype(cp.float64)]))  # primary key last: rows grouped by category, y ascending inside each group
    ys = y_d[order]
    counts_d = cp.asarray(np.ascontiguousarray(counts, dtype=np.int64))
    starts_d = cp.cumsum(counts_d) - counts_d
    flags = sum(1 << i for i, name in enumerate(_STAT_BITS) if name in wanted)
    out = cp.empty((7, n_cats), dtype=cp.float64)  # Intentional float64: statistics compared to the host's at full precision
    _kernel(cp)((n_cats,), (_THREADS,), (ys, starts_d, counts_d, np.int32(flags), np.float64(trim_frac), np.int32(n_cats), out))  # launch-ok: one block per category is the design
    res = cp.asnumpy(out)
    return tuple(res[i] for i in range(7))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 50_000  # rows from which the device sorts and sweeps before a sweep has run
_SWEEP_ROWS = [5_000, 20_000, 100_000, 500_000]
_SWEEP_CATS = 200
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``dims["n_samples"]`` rows over 200 categories, a skewed float target."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    inverse = rng.integers(0, _SWEEP_CATS, n).astype(np.int64)
    y = rng.standard_exponential(n) + 0.1 * inverse
    return (inverse, y, np.bincount(inverse, minlength=_SWEEP_CATS).astype(np.int64))


def _sweep_host(inverse: np.ndarray, y: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The host path: ``np.lexsort`` + the numba segment sweep."""
    from ._target_encoding_order_stats import _TRIM_FRAC, _segment_order_stats

    order = np.lexsort((y, inverse))
    ys = np.ascontiguousarray(y[order])
    starts = np.zeros(counts.shape[0], dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    res = _segment_order_stats(ys, starts, counts, True, True, True, True, True, True, True, _TRIM_FRAC)
    return np.concatenate([np.asarray(r) for r in res])


def _sweep_device(inverse: np.ndarray, y: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The device path; a failed launch raises, so the sweep never records a host fallback as a device time."""
    from ._target_encoding_order_stats import _TRIM_FRAC, ORDER_STATS

    return np.concatenate(segment_order_stats_gpu(inverse, y, counts, ORDER_STATS, _TRIM_FRAC))


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="order_stats_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="order_stats_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for order_stats_backend failed: %s", e)
    _SPEC = None


def order_device_chosen(n_rows: int) -> bool:
    """Whether the order statistics of ``n_rows`` rows are made on the device: the per-host cache by the row count, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("order_stats_backend: KTC choose() failed, using the floor: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def order_stats_dispatch(inverse: np.ndarray, y: np.ndarray, counts: np.ndarray, wanted: Sequence[str], trim_frac: float) -> "tuple[np.ndarray, ...] | None":
    """The seven raw per-category arrays from the device when the tuned choice says so and a device is usable, else ``None`` (the host sort and sweep run); a device fault returns ``None`` with a throttled warning."""
    if int(y.shape[0]) and order_device_chosen(int(y.shape[0])):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                return segment_order_stats_gpu(inverse, y, counts, wanted, trim_frac)
        except Exception as exc:  # a device fault must not lose the encoder
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "order_stats.device_failed", logging.WARNING, "per-category order statistics device path failed (%s: %s); host sort and sweep", type(exc).__name__, exc)
    return None
