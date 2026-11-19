"""Equal-frequency codes of a block of columns on the device, with the exact semantics of ``_fe_gain_stats.quantile_codes`` (no de-duplication of tied edges).

``_quantile_codes_gpu`` is the device twin of the block coder ``_numba_utils.quantile_codes_block`` (tied edges dropped, int16 codes, few-level columns coded by level); this module is
the twin of ``quantile_codes`` of the gain statistics, whose convention counts every edge. The sort is the project's pool-safe ``device_sort`` and the block width follows the pool.

The scan families that screen a target for structure (``_target_mix_gate``, ``_pair_residual_screen``, ``_row_stat_fe``) code every candidate column twice per call with
``quantile_codes``: ``np.quantile`` of the finite values for the edges, then the number of edges <= each value. At 50000 rows and 100 columns the two numpy calls per column
were 1.4 s of the 2.5 s of one ``target_has_structure`` call. This module does both for a whole block of columns at once: the exact order statistics come from the
batched sort of the block (ONE sort serves every bin count asked for), interpolated by numpy's own rule in the dtype numpy uses, so the edges are BIT-IDENTICAL to ``np.quantile`` (the
radix-select edges of ``_gpu_resident_select`` differ from numpy's lerp by about 1e-16 and decline more than 33 bins); ONE launch per bin count counts, per value, the edges that are <= it.

Semantics kept: every edge counts (no de-duplication of tied edges, unlike ``_fe_batched_qbin``), a non-finite value gets code 0, the comparison is made in double. A column with a non-finite value is coded on the host, because the sort would put it among the order statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
from mlframe.utils.array_shape import length_of

from ._fe_gain_stats import plugin_mi_of_codes, quantile_codes
from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import RawModuleCache

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# One block row per column (blockIdx.y); the column's edges sit in shared memory, a thread walks the rows with a grid stride and binary-searches them. The column is read coalesced
# (consecutive threads, consecutive rows of the (k, n) column-major block) and the codes are written the same way, one byte each.
_SRC = render(
    r"""
extern "C" __global__ void quantile_codes_right(const {REAL}* __restrict__ x, const {EDGE}* __restrict__ edges, const long long n, const int n_edges, {CODE}* __restrict__ out) {
    extern __shared__ {EDGE} sh_edges[];
    const long long col = blockIdx.y;
    for (int e = threadIdx.x; e < n_edges; e += blockDim.x) sh_edges[e] = edges[col * n_edges + e];
    __syncthreads();
    const {REAL}* xc = x + col * n;
    {CODE}* oc = out + col * n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const {EDGE} v = ({EDGE})xc[i];
        int lo = 0;
        if (isfinite(v)) {
            int hi = n_edges;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (sh_edges[mid] <= v) lo = mid + 1; else hi = mid;
            }
        }
        oc[i] = ({CODE})lo;
    }
}
""",
    edge=np.float64,  # Intentional float64: the comparison sees the double edge, as numpy's searchsorted does
).replace("{CODE}", "unsigned char")
_MODULES = {"float": RawModuleCache(_SRC.replace("{REAL}", "float")), "double": RawModuleCache(_SRC.replace("{REAL}", "double"))}

_THREADS = 256
_BUDGET_FRACTION = 0.25  # of what the memory pool can still hand out
_BLOCKS_PER_SM = 8
MAX_DEVICE_BINS = 255  # the codes are one byte
DEVICE_FALLBACK_ROWS = 20_000  # the hand-set mark used until a sweep has run on this host (a single block of 100 columns pays from about here)
_SWEEP_ROWS = [5_000, 20_000, 50_000, 200_000]
_SWEEP_COLS = 64
_SWEEP_BINS = 15
_SALT = 1


def _bin_list(bins: Any) -> list:
    """The bin counts asked for as a list of ints (one int or a sequence of them)."""
    if isinstance(bins, (int, np.integer)):
        return [int(bins)]
    return [int(b) for b in bins]


def kernel(cp: Any, real: Any) -> Any:
    """``quantile_codes_right`` for columns of dtype ``real`` (float32 or double precision); compile point of :func:`_codes_of_block` and of the register-pressure registry."""
    return _MODULES["float" if np.dtype(real) == np.float32 else "double"].get(cp).get_function("quantile_codes_right")


def _host_codes(cols: Sequence[np.ndarray], n_bins: int) -> np.ndarray:
    """``(k, n)`` int64 codes by the host ``quantile_codes`` of every column (the reference form and the fallback)."""
    return np.ascontiguousarray(np.stack([quantile_codes(np.asarray(c), n_bins) for c in cols]))


def _numpy_edges(cp: Any, sorted_cols: Any, n_bins: int) -> Any:
    """``(k, n_bins - 1)`` interior edges of the ascending ``(k, n)`` block ``sorted_cols`` by numpy's 'linear' rule (virtual index ``(n - 1) q``, difference taken in the data dtype, the
    ``t >= 0.5`` branch ``b - diff (1 - t)``), so each edge equals ``np.quantile`` of that column bit for bit; double precision whatever the data dtype."""
    n = int(sorted_cols.shape[1])
    q = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    virtual = (n - 1) * q
    prev = np.floor(virtual).astype(np.int64)
    nxt = np.minimum(prev + 1, n - 1)
    t = cp.asarray(virtual - prev)[None, :]
    a = sorted_cols[:, cp.asarray(prev)]
    b = sorted_cols[:, cp.asarray(nxt)]
    diff = b - a
    return cp.ascontiguousarray(cp.where(t >= 0.5, b - diff * (1 - t), a + diff * t))


def _codes_of_block(cp: Any, block: np.ndarray, bins: Sequence[int], on_device: bool = False) -> list:
    """For an all-finite ``(kb, n)`` host block, one ``(kb, n)`` uint8 code matrix per entry of ``bins`` (one sort of the block serves them all); with ``on_device`` the matrices stay cupy arrays."""
    from pyutilz.performance.gpu_hw_launch import block_size_for

    from ._device_sort_safe import device_sort

    xd = cp.asarray(block)
    kb, n = int(xd.shape[0]), int(xd.shape[1])
    s = device_sort(xd, axis=1)  # raises OutOfMemoryError instead of letting thrust abort the process on a short pool
    kern = kernel(cp, xd.dtype)
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    results = []
    for b in bins:
        edges = _numpy_edges(cp, s, int(b))
        out = cp.empty((kb, n), dtype=cp.uint8)
        shared = int(edges.shape[1]) * 8
        threads = min(_THREADS, block_size_for(kern, dyn_smem=shared, work_items=n))
        grid_x = int(max(1, min((n + threads - 1) // threads, sms * _BLOCKS_PER_SM)))
        kern((grid_x, kb), (threads,), (xd, edges, np.int64(n), np.int32(edges.shape[1]), out), shared_mem=shared)  # launch-ok: one launch codes every column of the block
        results.append(out if on_device else out.get())
    return results


def _block_width(cp: Any, n: int, k: int, itemsize: int) -> int:
    """Columns per device block (the column blocks the launch loop streams through): the input block, its sorted copy, the gathered order statistics and the byte codes must fit the VRAM cushion; halved from ``k``."""
    from ._fe_gpu_vram import pool_column_block

    return min(int(k), pool_column_block(cp, n, bytes_per_column_row=3 * itemsize + 6, budget_fraction=_BUDGET_FRACTION, min_block=1, fallback=16, what="device gain-stats codes"))


def quantile_codes_columns_device(cols: Sequence[np.ndarray], bins: Any) -> Optional[list]:
    """One ``(k, n)`` int64 code matrix per bin count in ``bins`` (an int or a sequence), coded on the device, or ``None`` when the device path does not apply (no card, a one-bin or wide-bin
    request) or faulted.

    Columns with a non-finite value are coded on the host, the rest in blocks of columns that fit the VRAM cushion; a column's codes depend on that column alone, so the blocks give the
    one-pass result."""
    bin_list = _bin_list(bins)
    k = len(cols)
    if k == 0 or not bin_list or min(bin_list) < 2 or max(bin_list) > MAX_DEVICE_BINS:
        return None
    try:
        import cupy as cp

        n = int(length_of(cols[0]))
        if n < 2:
            return None
        dtypes = {np.asarray(c).dtype for c in cols}
        work = np.float32 if dtypes == {np.dtype(np.float32)} else np.float64  # Intentional float64: any other mix is binned in double, as numpy would compare it
        x = np.ascontiguousarray(np.stack([np.asarray(c, dtype=work) for c in cols]))
        finite_cols = np.isfinite(x).all(axis=1)
        outs = [np.empty((k, n), dtype=np.int64) for _ in bin_list]
        for j in np.flatnonzero(~finite_cols):
            for o, b in zip(outs, bin_list):
                o[j] = quantile_codes(x[j], b)
        idx = np.flatnonzero(finite_cols)
        width = _block_width(cp, n, int(idx.size), int(x.dtype.itemsize)) if idx.size else 1
        for c0 in range(0, int(idx.size), width):
            sel = idx[c0 : c0 + width]
            for o, got in zip(outs, _codes_of_block(cp, np.ascontiguousarray(x[sel]), bin_list)):
                o[sel] = got
        return outs
    except Exception as exc:
        log_throttle(logger, "gain_stats_codes_gpu_device_quantile_codes_failed", logging.WARNING, "device quantile codes failed (%s: %s); host form", type(exc).__name__, exc)
        return None


def _main_effect_z(cp: Any, codes: Any, t_dev: Any, even_idx: Any, odd_idx: Any, bins: int, base: float) -> Any:
    """``(kb,)`` t statistics of the held-out gain of a per-bin mean model: the target mean of each bin of the even rows predicts the odd rows against the constant ``base``, the statistic is
    mean(d) / (std(d) / sqrt(m)) of ``d = (t - base)^2 - (t - mean[bin])^2`` over the m odd rows (0 where d has no spread; a one-level column has rounding noise in d on the host as well, and the callers skip such columns). Every column of the ``(kb, n)`` device ``codes`` at once, no Python loop."""
    kb = int(codes.shape[0])
    t_even, t_odd = t_dev[even_idx], t_dev[odd_idx]
    key = codes[:, even_idx].astype(cp.int64) + (cp.arange(kb, dtype=cp.int64) * bins)[:, None]
    flat = key.ravel()
    cnt = cp.bincount(flat, minlength=kb * bins).astype(cp.float64)  # Intentional float64: bin counts and sums of the target feed a threshold test
    total = cp.bincount(flat, weights=cp.broadcast_to(t_even, key.shape).ravel(), minlength=kb * bins)
    mean = cp.where(cnt > 0, total / cp.maximum(cnt, 1.0), base).reshape(kb, bins)
    mean_odd = cp.take_along_axis(mean, codes[:, odd_idx].astype(cp.int64), axis=1)
    d = (t_odd - base) ** 2 - (t_odd[None, :] - mean_odd) ** 2
    sd = d.std(axis=1)
    return cp.where(sd > 0.0, d.mean(axis=1) / cp.maximum(sd / np.sqrt(float(d.shape[1])), 1e-300), 0.0)


def main_effect_z_device(cols: Sequence[np.ndarray], bins: int, t: np.ndarray, even: np.ndarray, base: float) -> Optional[np.ndarray]:
    """``(k,)`` held-out-gain t statistics of the columns ``cols`` on the device (codes never leave the card), or ``None`` when the device path does not apply (no card, a bin count outside
    2 to 255, a column with a non-finite value, fewer than two rows of either parity) or faulted; the caller then codes on the host and runs its own loop."""
    k = len(cols)
    if k == 0 or int(bins) < 2 or int(bins) > MAX_DEVICE_BINS:
        return None
    try:
        import cupy as cp

        n = int(length_of(cols[0]))
        even_idx, odd_idx = np.flatnonzero(even), np.flatnonzero(~even)
        if n < 2 or even_idx.size < 2 or odd_idx.size < 2:
            return None
        dtypes = {np.asarray(c).dtype for c in cols}
        work = np.float32 if dtypes == {np.dtype(np.float32)} else np.float64  # Intentional float64: any other mix is binned in double, as numpy would compare it
        x = np.ascontiguousarray(np.stack([np.asarray(c, dtype=work) for c in cols]))
        if not np.isfinite(x).all():
            return None
        t_dev = cp.asarray(np.asarray(t, dtype=np.float64))  # Intentional float64: target labels kept exact
        ev, od = cp.asarray(even_idx), cp.asarray(odd_idx)
        out = np.empty(k, dtype=np.float64)  # Intentional float64: the statistic is compared with a threshold
        width = _block_width(cp, n, k, int(x.dtype.itemsize))
        for c0 in range(0, k, width):
            codes = _codes_of_block(cp, np.ascontiguousarray(x[c0 : c0 + width]), [int(bins)], on_device=True)[0]
            out[c0 : c0 + width] = cp.asnumpy(_main_effect_z(cp, codes, t_dev, ev, od, int(bins), float(base)))
        return out
    except Exception as exc:
        log_throttle(logger, "gain_stats_codes_gpu_device_main_effect_statistics_failed", logging.WARNING, "device main-effect statistics failed (%s: %s); host form", type(exc).__name__, exc)
        return None


def main_effect_z_columns(cols: Sequence[np.ndarray], bins: int, t: np.ndarray, even: np.ndarray, base: float) -> Optional[np.ndarray]:
    """:func:`main_effect_z_device` when the tuner says the device for this row count, else ``None`` (the caller's host loop)."""
    if len(cols) and quantile_codes_on_device(int(length_of(cols[0]))):
        return main_effect_z_device(cols, bins, t, even, base)
    return None


def _plugin_mi_of_device_codes(cp: Any, codes: Any, bins: int, y_dev: Any, ky: int) -> Any:
    """``(kb,)`` plug-in MI (nats) of each row of the device code matrix ``codes`` with the labels ``y_dev``: the sum over occupied cells of ``p(x, y) * log(p(x, y) / (p(x) p(y)))``, the mean of the pointwise MI."""
    kb, n = int(codes.shape[0]), int(codes.shape[1])
    key = (codes.astype(cp.int64) * ky + y_dev[None, :]) + (cp.arange(kb, dtype=cp.int64) * (bins * ky))[:, None]
    joint = cp.bincount(key.ravel(), minlength=kb * bins * ky).reshape(kb, bins, ky).astype(cp.float64)  # Intentional float64: counts feed a logarithm that is summed
    px = joint.sum(axis=2, keepdims=True)
    py = joint.sum(axis=1, keepdims=True)
    safe = cp.where(joint > 0, joint, 1.0)
    term = cp.where(joint > 0, joint * cp.log(safe * float(n) / cp.maximum(px * py, 1.0)), 0.0)
    return term.sum(axis=(1, 2)) / float(n)


def plugin_mi_columns_device(cols: Sequence[np.ndarray], bins: int, y_codes: np.ndarray, ky: int) -> Optional[np.ndarray]:
    """``(k,)`` plug-in MI of the ``bins``-level quantile codes of each column with the integer labels ``y_codes`` (in ``[0, ky)``), the codes never leaving the card; ``None`` when the device path
    does not apply (no card, a bin count outside 2 to 255, a non-finite value) or faulted, and the caller codes on the host."""
    k = len(cols)
    if k == 0 or int(bins) < 2 or int(bins) > MAX_DEVICE_BINS:
        return None
    try:
        import cupy as cp

        n = int(length_of(cols[0]))
        if n < 2:
            return None
        dtypes = {np.asarray(c).dtype for c in cols}
        work = np.float32 if dtypes == {np.dtype(np.float32)} else np.float64  # Intentional float64: any other mix is binned in double, as numpy would compare it
        x = np.ascontiguousarray(np.stack([np.asarray(c, dtype=work) for c in cols]))
        if not np.isfinite(x).all():
            return None
        y_dev = cp.asarray(np.asarray(y_codes, dtype=np.int64))
        out = np.empty(k, dtype=np.float64)  # Intentional float64: the statistic ranks columns
        width = _block_width(cp, n, k, int(x.dtype.itemsize))
        for c0 in range(0, k, width):
            codes = _codes_of_block(cp, np.ascontiguousarray(x[c0 : c0 + width]), [int(bins)], on_device=True)[0]
            out[c0 : c0 + width] = cp.asnumpy(_plugin_mi_of_device_codes(cp, codes, int(bins), y_dev, int(ky)))
        return out
    except Exception as exc:
        log_throttle(logger, "gain_stats_codes_gpu_device_plugin_mi_failed", logging.WARNING, "device plug-in MI of quantile codes failed (%s: %s); host form", type(exc).__name__, exc)
        return None


def plugin_mi_columns(cols: Sequence[np.ndarray], bins: int, y_codes: np.ndarray, ky: int) -> np.ndarray:
    """``(k,)`` plug-in MI of the ``bins``-level quantile codes of each column with ``y_codes``: :func:`plugin_mi_columns_device` when the tuner says the device for this row count, else the host
    codes and the compiled plug-in MI per column."""
    if len(cols) and quantile_codes_on_device(int(length_of(cols[0]))):
        got = plugin_mi_columns_device(cols, bins, y_codes, ky)
        if got is not None:
            return got
    q = _host_codes(cols, int(bins))
    return np.array([plugin_mi_of_codes(q[i], int(bins), y_codes, int(ky)) for i in range(len(cols))], dtype=np.float64)  # Intentional float64: ranking statistic


def _sweep_inputs(dims: dict) -> tuple:
    """``(columns, n_bins)``: ``_SWEEP_COLS`` float32 columns of ``dims["n_samples"]`` rows."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((_SWEEP_COLS, int(dims["n_samples"]))).astype(np.float32)
    return (list(x), _SWEEP_BINS)


def _sweep_host(cols: Sequence[np.ndarray], n_bins: int) -> np.ndarray:
    """The host form: ``quantile_codes`` per column."""
    return _host_codes(cols, n_bins)


def _sweep_device(cols: Sequence[np.ndarray], n_bins: int) -> np.ndarray:
    """The device form."""
    got = quantile_codes_columns_device(cols, n_bins)
    if got is None:
        raise RuntimeError("device quantile codes declined")
    codes: np.ndarray = got[0]
    return codes


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="gain_stats_codes_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=0.0, atol=0.0),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="gain_stats_codes_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for gain_stats_codes_backend failed: %s", e)
    _SPEC = None


def quantile_codes_on_device(n_rows: int) -> bool:
    """Whether a block of columns of ``n_rows`` rows is coded on the device: the per-host kernel-tuning cache, else the ``DEVICE_FALLBACK_ROWS`` mark."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("gain_stats_codes_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def quantile_codes_columns(cols: Sequence[np.ndarray], bins: Any) -> list:
    """One ``(k, n)`` int64 code matrix per bin count in ``bins`` (an int gives a one-element list) for the equal-length host vectors ``cols``: the device when the tuner says so for this row
    count and a card is usable, else the host ``quantile_codes`` per column."""
    bin_list = _bin_list(bins)
    if len(cols) == 0:
        return [np.empty((0, 0), dtype=np.int64) for _ in bin_list]
    if quantile_codes_on_device(int(length_of(cols[0]))):
        got = quantile_codes_columns_device(cols, bin_list)
        if got is not None:
            return got
    return [_host_codes(cols, b) for b in bin_list]


__all__ = ["plugin_mi_columns", "plugin_mi_columns_device", "main_effect_z_columns", "main_effect_z_device", "quantile_codes_columns", "quantile_codes_columns_device", "quantile_codes_on_device", "kernel"]
