"""Within-group percentile ranks and group spreads of a block of columns: a numba form, a fused CUDA form and the per-host choice between them.

``_grouped_quantile_fe.generate_grouped_quantile_features`` builds, for every (group column, numeric column), the percentile rank of each value within its group (ties take the average rank), and the
group's IQR and p90 - p10 spreads. On the host that is one radix order per group and per numeric column on the pool (0.9 s for 4 group columns x 32 numeric columns x 100000 rows).

Here the whole block of numeric columns is done at once. Each column is sorted ONCE by the pair (group, value): the 32-bit order-preserving key of the float32 value under the group code in the
high word, so one 64-bit sort per column puts every group's values side by side (a ``-0.0`` is canonicalised to ``+0.0`` first, so equal values have equal keys). Then
``group_pct_ranks`` (one thread per sorted position, a binary search for the bounds of the thread's run of equal values inside its group's segment) writes ``(lo + hi) / (2 m)`` to the
member's own row, and ``group_quantiles`` (one block per (group, column)) reads the four linear quantiles off the segment with numpy's own lerp (the float32 difference, the double precision interpolation),
so the values equal the host's bit for bit.

Two forms of one function: ``_stats_host`` (the per-group radix order and the nogil rank kernel of ``_grouped_quantile_ranks``, the host form the family used) and ``_stats_device`` (the kernels above);
``grouped_stats_on_device`` reads the per-host kernel-tuning cache (``grouped_quantile_stats_backend``, a sweep over the row count) with a row-count fallback; ``MLFRAME_GROUPED_QUANTILE_DEVICE=0|1``
forces a side. Only float32 columns without a non-finite value are taken (the host path keeps the rest); the columns go to the device in column blocks sized from the room under the pool.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

from ._grouped_quantile_ranks import _sorted_linear_quantiles
from ._numba_utils_shared import _member_pct_ranks_njit
from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._numba_utils_shared import radix_order_f64
from ._raw_module_cache import RawModuleCache

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

#: Quantile positions read off every group's sorted segment: q25, q75, p10, p90 (the IQR is q75 - q25, the spread p90 - p10).
QUANTILES = (0.25, 0.75, 0.10, 0.90)

_SRC = render(
    r"""
extern "C" __global__ void group_pct_ranks(const {REAL}* __restrict__ sv, const int* __restrict__ gs, const long long* __restrict__ order, const long long* __restrict__ seg,
                                           const long long n, {REAL}* __restrict__ out) {
    // sorted position i of column c: its group g = gs[i], the segment [s, e) of that group, the run of values equal to sv[i] inside the segment by two binary searches
    const long long col = blockIdx.y;
    const {REAL}* v = sv + col * n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int g = gs[i];
        const long long s = seg[g], e = seg[g + 1];
        const {REAL} x = v[i];
        long long lo = s, hi = i;               // first position in [s, i] with value >= x: sv is sorted, so the answer is at most i
        while (lo < hi) { const long long mid = (lo + hi) >> 1; if (v[mid] < x) lo = mid + 1; else hi = mid; }
        const long long first = lo;
        lo = i + 1; hi = e;                     // first position in (i, e) with value > x
        while (lo < hi) { const long long mid = (lo + hi) >> 1; if (v[mid] <= x) lo = mid + 1; else hi = mid; }
        const {ACC} m = ({ACC})(e - s);
        const {ACC} r = (({ACC})(first - s) + ({ACC})(lo - s)) / (2.0 * m);
        out[col * n + order[col * n + i]] = ({REAL})r;
    }
}

extern "C" __global__ void group_quantiles(const {REAL}* __restrict__ sv, const long long* __restrict__ seg, const long long n, const int n_groups,
                                           {ACC}* __restrict__ iqr, {ACC}* __restrict__ p90p10) {
    // one thread per (group, column): the four linear quantiles of the segment as numpy forms them (virtual index q * (m - 1), the difference in the value type, the interpolation in the accumulator type)
    const long long col = blockIdx.y;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const long long s = seg[g], m = seg[g + 1] - s;
    if (m <= 0) { iqr[col * n_groups + g] = 0; p90p10[col * n_groups + g] = 0; return; }
    const {REAL}* v = sv + col * n + s;
    {ACC} q[4];
    const {ACC} qs[4] = {0.25, 0.75, 0.10, 0.90};
    for (int k = 0; k < 4; ++k) {
        const {ACC} pos = qs[k] * ({ACC})(m - 1);
        long long lo = (long long)floor(pos);
        const long long hi = lo + 1 < m - 1 ? lo + 1 : m - 1;
        const {ACC} t = pos - ({ACC})lo;
        const {REAL} a = v[lo], b = v[hi];
        const {REAL} diff = b - a;
        {ACC} r = ({ACC})a + ({ACC})diff * t;
        if (t >= 0.5) r = ({ACC})b - ({ACC})diff * (1.0 - t);
        q[k] = r;
    }
    iqr[col * n_groups + g] = q[1] - q[0];
    p90p10[col * n_groups + g] = q[3] - q[2];
}
"""
).replace("{REAL}", "float")
_MODULE = RawModuleCache(_SRC)

_THREADS = 256


def kernel(cp: Any, real: Any = np.float32) -> Any:
    """``group_pct_ranks`` for float32 columns; compile point of :func:`_stats_device` and of the register-pressure registry."""
    if np.dtype(real) != np.float32:
        raise ValueError("the grouped-quantile kernels are instantiated for float32 columns only")
    return _MODULE.get(cp).get_function("group_pct_ranks")


def kernel_quantiles(cp: Any, real: Any = np.float32) -> Any:
    """``group_quantiles`` for float32 columns; compile point of :func:`_stats_device` and of the register-pressure registry."""
    if np.dtype(real) != np.float32:
        raise ValueError("the grouped-quantile kernels are instantiated for float32 columns only")
    return _MODULE.get(cp).get_function("group_quantiles")


def _group_layout(codes: np.ndarray, n_groups: int) -> tuple:
    """``(counts, seg)``: rows per group and the ``(G + 1,)`` offsets of the group segments in a group-major order."""
    counts = np.bincount(codes, minlength=n_groups).astype(np.int64)
    seg = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(counts, out=seg[1:])
    return counts, seg


def _stats_host(xs: np.ndarray, codes: np.ndarray, n_groups: int) -> dict:
    """Host form for the finite float32 block ``xs`` ``(k, n)``: per group the radix order, the sorted segment, the average-rank percentiles and the linear quantiles."""
    k, n = xs.shape
    counts, seg = _group_layout(codes, n_groups)
    rows_by_group = np.argsort(codes, kind="stable")  # the rows of each group, group-major, in row order
    pct = np.empty((k, n), dtype=np.float32)
    srt = np.empty((k, n), dtype=np.float32)
    iqr = np.zeros((k, n_groups), dtype=np.float64)  # Intentional float64: spread statistics, as np.quantile returns them
    p90p10 = np.zeros((k, n_groups), dtype=np.float64)  # Intentional float64: spread statistics, as np.quantile returns them
    for c in range(k):
        for g in range(n_groups):
            m = int(counts[g])
            if m == 0:
                continue
            rows = rows_by_group[seg[g] : seg[g + 1]]
            vals = xs[c, rows]
            order = radix_order_f64(np.ascontiguousarray(vals, dtype=np.float64))  # Intentional float64: radix_order_f64 takes float64 keys
            sv = vals[order]
            ranks = np.empty(m, dtype=np.float64)  # Intentional float64: rank positions
            _member_pct_ranks_njit(np.ascontiguousarray(order, dtype=np.int64), sv, ranks)
            pct[c, rows] = ranks
            srt[c, seg[g] : seg[g + 1]] = sv
            q25, q75, p10, p90 = _sorted_linear_quantiles(sv, QUANTILES)
            iqr[c, g] = q75 - q25
            p90p10[c, g] = p90 - p10
    return {"pct": pct, "sorted": srt, "seg": seg, "iqr": iqr, "p90p10": p90p10}


def _stats_device(xs: np.ndarray, codes: np.ndarray, n_groups: int) -> Optional[dict]:
    """Device form of :func:`_stats_host`, or ``None`` (no cupy, a device fault, a pool too short for the block)."""
    try:
        import cupy as cp

        from ._device_sort_safe import device_argsort, device_room

        k, n = xs.shape
        counts, seg = _group_layout(codes, n_groups)
        per_col = n * (4 + 8 + 8 + 4 + 4)  # values, 64-bit keys, sort order, sorted values, ranks
        block = int(max(1, min(k, (0.25 * device_room(cp)) // max(1, per_col))))
        gs = cp.asarray(np.repeat(np.arange(n_groups, dtype=np.int32), counts))
        seg_d = cp.asarray(seg)
        codes_hi = cp.asarray(codes.astype(np.uint64) << np.uint64(32))
        pct = np.empty((k, n), dtype=np.float32)
        srt = np.empty((k, n), dtype=np.float32)
        iqr = np.empty((k, n_groups), dtype=np.float64)  # Intentional float64: spread statistics
        p90p10 = np.empty((k, n_groups), dtype=np.float64)  # Intentional float64: spread statistics
        rank_kernel, quant_kernel = kernel(cp), kernel_quantiles(cp)
        for c0 in range(0, k, block):
            c1 = min(k, c0 + block)
            xd = cp.asarray(np.ascontiguousarray(xs[c0:c1], dtype=np.float32)) + np.float32(0.0)  # -0.0 becomes +0.0: equal values get equal keys
            bits = xd.view(cp.uint32)
            mono = cp.where(bits >> 31 == 1, ~bits, bits | np.uint32(0x80000000)).astype(cp.uint64)
            order = device_argsort(mono | codes_hi[None, :], axis=1)
            sv = cp.ascontiguousarray(cp.take_along_axis(xd, order, axis=1))
            width = c1 - c0
            out = cp.empty((width, n), dtype=cp.float32)
            iq = cp.empty((width, n_groups), dtype=cp.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
            pp = cp.empty((width, n_groups), dtype=cp.float64)
            rank_kernel((64, width), (_THREADS,), (sv, gs, order, seg_d, np.int64(n), out))  # launch-ok: a grid-stride loop over the sorted positions, any block of a warp multiple is correct
            quant_kernel(((n_groups + 31) // 32, width), (32,), (sv, seg_d, np.int64(n), np.int32(n_groups), iq, pp))  # launch-ok: one thread per group, a warp of groups per block
            pct[c0:c1], srt[c0:c1], iqr[c0:c1], p90p10[c0:c1] = cp.asnumpy(out), cp.asnumpy(sv), cp.asnumpy(iq), cp.asnumpy(pp)
            del xd, bits, mono, order, sv, out, iq, pp
        return {"pct": pct, "sorted": srt, "seg": seg, "iqr": iqr, "p90p10": p90p10}
    except Exception as exc:
        log_throttle(logger, "grouped_quantile_gpu_grouped_quantile_the_device_form_is_unavailable", logging.WARNING, "grouped quantile: the device form is unavailable (%s: %s)", type(exc).__name__, exc)
        return None


DEVICE_FALLBACK_ROWS = 20_000
_SWEEP_ROWS = [5_000, 20_000, 50_000, 200_000]
_SWEEP_COLS = 16
_SWEEP_GROUPS = 12
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(xs, codes, n_groups)``: ``_SWEEP_COLS`` float32 columns of ``dims["n_samples"]`` rows split into ``_SWEEP_GROUPS`` groups."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    xs = rng.standard_normal((_SWEEP_COLS, n)).astype(np.float32)
    xs[0] = np.round(xs[0], 1)  # ties
    return (xs, rng.integers(0, _SWEEP_GROUPS, size=n).astype(np.int64), _SWEEP_GROUPS)


def _sweep_host(xs: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """The host form's percentile ranks and spreads on the sweep inputs, as one comparable array."""
    got = _stats_host(xs, codes, n_groups)
    return np.concatenate([got["pct"].ravel().astype(np.float64), got["iqr"].ravel(), got["p90p10"].ravel()])  # Intentional float64: statistic / reference vector in double precision (compared exactly with its host twin)


def _sweep_device(xs: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """The device form's percentile ranks and spreads on the sweep inputs, as one comparable array."""
    got = _stats_device(xs, codes, n_groups)
    if got is None:
        raise RuntimeError("device grouped quantile declined")
    return np.concatenate([got["pct"].ravel().astype(np.float64), got["iqr"].ravel(), got["p90p10"].ravel()])  # Intentional float64: statistic / reference vector in double precision (compared exactly with its host twin)


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="grouped_quantile_stats_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=0.0, atol=0.0),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="grouped_quantile_stats_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for grouped_quantile_stats_backend failed: %s", e)
    _SPEC = None


def grouped_stats_on_device(n_rows: int) -> bool:
    """Whether a block of columns of ``n_rows`` rows is ranked on the device: ``MLFRAME_GROUPED_QUANTILE_DEVICE=0|1`` forces a side, else the per-host kernel-tuning cache, else the row mark."""
    forced = os.environ.get("MLFRAME_GROUPED_QUANTILE_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("grouped_quantile_stats_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def grouped_stats_block(xs: np.ndarray, codes: np.ndarray, n_groups: int) -> Optional[dict]:
    """Percentile ranks within groups and group spreads of the finite float32 block ``xs`` ``(k, n)`` for the group codes ``codes`` ``(n,)``: ``{"pct": (k, n) float32, "sorted": (k, n) float32 (values in
    group-major sorted order), "seg": (G + 1,) offsets, "iqr": (k, G), "p90p10": (k, G)}`` from the backend the kernel-tuning cache picks. ``None`` for a block the forms do not take (not float32, a non-finite
    value, codes out of range): the caller keeps its per-group loop."""
    if xs.ndim != 2 or xs.dtype != np.float32 or not xs.shape[0] or not np.isfinite(xs).all():
        return None
    if codes.size != xs.shape[1] or int(codes.min()) < 0 or int(codes.max()) >= n_groups:
        return None
    if grouped_stats_on_device(int(xs.shape[1])):
        got = _stats_device(xs, codes, n_groups)
        if got is not None:
            return got
    return _stats_host(xs, codes, n_groups)
