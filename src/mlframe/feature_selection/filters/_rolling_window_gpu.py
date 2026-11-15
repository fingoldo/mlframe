"""Device twin of the per-entity rolling time-window statistic ``_rolling_stat_past_only_njit`` of ``_temporal_agg_fe_rolling``.

The host kernel walks, for every row, the entity's history newest to oldest through a ``prev`` linked list while the time stays inside the window, and folds the strictly past finite values into count, sum,
Welford mean/m2, minimum and maximum. Rows are independent once ``prev`` is known, so on the device one thread takes one row and performs the same walk in the same order: the count, minimum and maximum are
identical, the mean and the standard deviation follow the same Welford recurrence. ``prev`` is built on the host in one stable argsort (no per-row Python). Times are int64 nanoseconds for a datetime column or
double for a numeric one (two compiled variants). Host or device is chosen per call from the kernel-tuning entry ``rolling_window_backend`` by the row count; a device fault falls back to the host
with a throttled warning. The arrays of one value column and its entity codes are one column block, uploaded whole: 28 bytes a row, so 10 million rows are 280 MB and a block never has to be split.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["rolling_stat_past_only_gpu", "rolling_dispatch", "rolling_device_chosen"]

_THREADS = 256  # upper bound of the block; the launch takes the size from the device (no ``__launch_bounds__``: on sm_61 any bound caps the kernel at 40 registers and spills 12 bytes)

_SRC = render(
    r"""
extern "C" __global__
void rolling_stat_past_only(const {TT}* __restrict__ times, const {ACC}* __restrict__ vals, const long long* __restrict__ prev,
                            const {TT} td, const int stat_code, const long long n, {ACC}* __restrict__ out) {
    // prev[j] = the next older row of j's entity (-1 when none); the walk of row i starts at prev[i], the newest earlier row of its entity
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const {TT} t = times[i];
    const {TT} lo = t - td;
    long long cnt = 0;
    {ACC} s = 0.0, mean_w = 0.0, m2 = 0.0;
    const {ACC} inf_v = ({ACC})__longlong_as_double(0x7FF0000000000000LL);
    {ACC} mn = inf_v, mx = -inf_v;
    long long j = prev[i];
    while (j != -1 && times[j] >= lo) {
        if (times[j] < t) {
            const {ACC} v = vals[j];
            if (isfinite(v)) {
                cnt += 1;
                s += v;
                const {ACC} delta = v - mean_w;
                mean_w += delta / ({ACC})cnt;
                m2 += delta * (v - mean_w);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        j = prev[j];
    }
    {ACC} r = ({ACC})__longlong_as_double(0x7FF8000000000000LL);
    if (cnt > 0) {
        if (stat_code == 0) r = ({ACC})cnt;
        else if (stat_code == 1) r = s / ({ACC})cnt;
        else if (stat_code == 2) {
            if (cnt <= 1) r = 0.0;
            else { const {ACC} var = m2 / ({ACC})(cnt - 1); r = var > 0.0 ? sqrt(var) : 0.0; }
        }
        else if (stat_code == 3) r = mn;
        else if (stat_code == 4) r = mx;
    }
    out[i] = r;
}
""",
)

_KERNELS: dict = {}


def _kernel(cp: Any, integer_time: bool) -> Any:
    """The compiled kernel for int64 (datetime) or double (numeric) times, one per process."""
    key = bool(integer_time)
    if key not in _KERNELS:
        src = _SRC.replace("{TT}", "long long" if key else "double")
        _KERNELS[key] = cp.RawKernel(src, "rolling_stat_past_only")
    return _KERNELS[key]


def _prev_links(group_codes: np.ndarray) -> np.ndarray:
    """``prev[j]``: the newest row before j in j's entity, -1 when absent; the same links the host sweep builds in row order."""
    n = group_codes.shape[0]
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    same = g[1:] == g[:-1]
    prev = np.full(n, -1, dtype=np.int64)
    prev[order[1:]] = np.where(same, order[:-1], -1)
    return prev


def rolling_stat_past_only_gpu(times: np.ndarray, vals: np.ndarray, group_codes: np.ndarray, td: Any, stat_code: int) -> np.ndarray:
    """The rolling statistic per row as ``_rolling_stat_past_only_njit`` returns it (NaN where the window holds no finite past value). ``times`` is int64 ns or double, as the host takes it. Raises on a device fault."""
    import cupy as cp

    n = int(vals.shape[0])
    integer_time = np.asarray(times).dtype.kind in "iu"
    gc = np.ascontiguousarray(group_codes, dtype=np.int64)
    prev = _prev_links(gc)
    tt = np.int64 if integer_time else np.float64
    times_d = cp.asarray(np.ascontiguousarray(times, dtype=tt))
    vals_d = cp.asarray(np.ascontiguousarray(vals, dtype=np.float64))  # Intentional float64: the running mean and m2 are formed in double, as on the host
    prev_d = cp.asarray(prev)
    out = cp.empty(n, dtype=cp.float64)  # Intentional float64: statistics compared to the host's at full precision
    kernel = _kernel(cp, integer_time)
    threads = min(_THREADS, block_size_for(kernel, work_items=n))
    kernel(((n + threads - 1) // threads,), (threads,), (times_d, vals_d, prev_d, tt(td), np.int32(stat_code), np.int64(n), out))
    return np.asarray(cp.asnumpy(out))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 100_000  # rows from which the device walks before a sweep has run
_SWEEP_ROWS = [5_000, 20_000, 100_000, 500_000]
_SWEEP_ENTITIES = 500
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``dims["n_samples"]`` rows of ``_SWEEP_ENTITIES`` entities in (entity, time) order, a numeric time axis, a noisy value column with some NaN, a window of 30 time units, the std statistic."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    gc = np.sort(rng.integers(0, _SWEEP_ENTITIES, n)).astype(np.int64)
    times = np.empty(n, dtype=np.float64)  # Intentional float64: epoch-scale numeric time
    for g in np.unique(gc):
        idx = np.flatnonzero(gc == g)
        times[idx] = np.sort(rng.uniform(0.0, 365.0, idx.size))
    vals = rng.standard_normal(n)
    vals[rng.random(n) < 0.02] = np.nan
    return (times, vals, gc, 30.0, 2)


def _sweep_host(times: np.ndarray, vals: np.ndarray, gc: np.ndarray, td: float, stat_code: int) -> np.ndarray:
    """The host path: the numba linked-list walk."""
    from ._temporal_agg_fe_rolling import _rolling_stat_past_only_njit

    return np.asarray(_rolling_stat_past_only_njit(times, vals, gc, td, stat_code, int(gc.max()) + 1))


def _sweep_device(times: np.ndarray, vals: np.ndarray, gc: np.ndarray, td: float, stat_code: int) -> np.ndarray:
    """The device path; a failed launch raises, so the sweep never records a host fallback as a device time."""
    return rolling_stat_past_only_gpu(times, vals, gc, td, stat_code)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="rolling_window_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="rolling_window_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for rolling_window_backend failed: %s", e)
    _SPEC = None


def rolling_device_chosen(n_rows: int) -> bool:
    """Whether the rolling statistic of ``n_rows`` rows is made on the device: the per-host cache by the row count, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("rolling_window_backend: KTC choose() failed, using the floor: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def rolling_dispatch(times: np.ndarray, vals: np.ndarray, group_codes: np.ndarray, td: Any, stat_code: int) -> "np.ndarray | None":
    """The device result when the tuned choice says so and a device is usable, else ``None`` (the host walk runs); a device fault returns ``None`` with a throttled warning."""
    if int(vals.shape[0]) and rolling_device_chosen(int(vals.shape[0])):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                return rolling_stat_past_only_gpu(times, vals, group_codes, td, stat_code)
        except Exception as exc:  # a device fault must not lose the feature
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "rolling_window.device_failed", logging.WARNING, "rolling window statistic device path failed (%s: %s); host walk", type(exc).__name__, exc)
    return None
