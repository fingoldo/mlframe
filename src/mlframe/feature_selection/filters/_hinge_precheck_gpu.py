"""Slope-change precheck of the hinge basis for a whole block of columns: a numba kernel, a fused CUDA kernel and the per-host choice between them.

``_hinge_basis_fe._hinge_slope_change_plausible`` is the cheap gate in front of the 24-cut hinge scan: it fits the continuous two-segment hinge at three coarse inner quantiles and asks whether the best
fractional SSE drop over the plain line clears ``_HINGE_PRECHECK_MIN_SSE_DROP``. Per column it factored ``B = [1, x]`` with ``np.linalg.qr`` and projected each cut's relu leg through
``Q @ (Q.T @ relu)``: 15 ms a column at 100000 rows (1.56 s of the 2.11 s the family costs on 100 columns).

For ``B = [1, x]`` the projection is closed form. With ``xc = x - mean(x)`` and ``yc = y - mean(y)`` (``Sxx = xc.xc``, ``Sxy = xc.yc``, ``Syy = yc.yc``) the line leaves ``sse_lin = Syy - Sxy^2 / Sxx``.
For a cut ``c`` and ``relu = max(x - c, 0)`` the sums ``R = sum relu``, ``Rx = relu.xc``, ``Rr = relu.relu``, ``Ry = relu.yc`` give the Frisch-Waugh-Lovell terms
``r_relu.r_relu = Rr - R^2 / n - Rx^2 / Sxx`` and ``r_relu.r_y = Ry - Rx Sxy / Sxx``, so ``sse_hinge = sse_lin - (r_relu.r_y)^2 / (r_relu.r_relu)`` (``sse_lin`` when the leg is degenerate) and
``drop = 1 - sse_hinge / sse_lin``: two passes over a column (the means, then the sums of all three cuts), no QR, no ``Q``. The verdict agrees with the QR path to the rounding of a double (the drop is
compared with 0.005); the detector keeps its own QR for the columns that pass, because the round-0 scan reuses ``Q`` and is pinned bit for bit.

Two backends of one function: ``_drops_njit`` (``prange`` over columns, ``np.quantile`` for the cuts) and ``hinge_precheck_drops`` (one CUDA block per column, the cuts from the batched sort of the block).
``hinge_precheck_on_device`` reads the per-host kernel-tuning cache (``hinge_precheck_backend``, a sweep over the row count) with a row-count fallback; ``MLFRAME_HINGE_PRECHECK_DEVICE=0|1`` forces a side.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np
from numba import njit, prange

from mlframe._numerical_floors import VARIANCE_FLOOR

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

N_CUTS = 3  # the precheck is defined on exactly three coarse quantiles (_HINGE_PRECHECK_QS)
_THREADS = 256

_SRC = render(
    r"""
extern "C" __global__ void hinge_precheck_drops(const {REAL}* __restrict__ x, const {ACC}* __restrict__ y_centered, const {ACC}* __restrict__ cuts,
                                                const long long n, const int min_seg, const {ACC} floor_var, {ACC}* __restrict__ drops) {
    // one block per column: pass 1 gives the column mean and Sxx / Sxy, pass 2 the five sums of every cut (the three legs share one read of the column)
    const long long col = blockIdx.x;
    const {REAL}* xc_ptr = x + col * n;
    const {ACC}* cut = cuts + col * 3;
    __shared__ {ACC} red[32][8];
    {ACC} s_x = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) s_x += ({ACC})xc_ptr[i];
    for (int o = 16; o > 0; o >>= 1) s_x += __shfl_down_sync(0xffffffffu, s_x, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][0] = s_x;
    __syncthreads();
    __shared__ {ACC} mean_x;
    if (threadIdx.x == 0) { {ACC} t = 0; for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w][0]; mean_x = t / ({ACC})n; }
    __syncthreads();
    {ACC} sxx = 0, sxy = 0;
    {ACC} acc[3][5];
#pragma unroll
    for (int k = 0; k < 3; ++k) { acc[k][0] = 0; acc[k][1] = 0; acc[k][2] = 0; acc[k][3] = 0; acc[k][4] = 0; }
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const {ACC} xv = ({ACC})xc_ptr[i] - mean_x;
        const {ACC} yv = y_centered[i];
        sxx += xv * xv;
        sxy += xv * yv;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const {ACC} r = ({ACC})xc_ptr[i] - ({ACC})cut[k];
            if (r > 0) { acc[k][0] += 1; acc[k][1] += r; acc[k][2] += r * xv; acc[k][3] += r * r; acc[k][4] += r * yv; }
        }
    }
    // block reduction of the 2 + 15 sums, one value at a time through the warp shuffles and a shared row per warp
    __shared__ {ACC} tot[17];
    for (int s = 0; s < 17; ++s) {
        {ACC} v = (s == 0) ? sxx : (s == 1) ? sxy : acc[(s - 2) / 5][(s - 2) % 5];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][1] = v;
        __syncthreads();
        if (threadIdx.x == 0) { {ACC} t = 0; for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w][1]; tot[s] = t; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        {ACC} syy = 0;  // yc.yc is column independent, passed in y_centered[n]
        syy = y_centered[n];
        const {ACC} sse_lin = syy - tot[1] * tot[1] / tot[0];
        {ACC} best = 0.0;
        if (tot[0] > 0 && isfinite(sse_lin) && sse_lin > floor_var) {
            for (int k = 0; k < 3; ++k) {
                if (k > 0 && cut[k] == cut[k - 1]) continue;
                const long long n_right = (long long)tot[2 + 5 * k];
                if (n_right < min_seg || (n - n_right) < min_seg) continue;
                const {ACC} R = tot[3 + 5 * k], Rx = tot[4 + 5 * k], Rr = tot[5 + 5 * k], Ry = tot[6 + 5 * k];
                const {ACC} denom = Rr - R * R / ({ACC})n - Rx * Rx / tot[0];
                {ACC} sse_h = sse_lin;
                if (!(denom < floor_var)) { const {ACC} num = Ry - Rx * tot[1] / tot[0]; sse_h = sse_lin - num * num / denom; }
                if (!isfinite(sse_h)) continue;
                const {ACC} drop = 1.0 - sse_h / sse_lin;
                if (drop > best) best = drop;
            }
            drops[col] = best;
        } else {
            drops[col] = -1.0;  // a column the precheck rejects outright (a perfect line or no spread)
        }
    }
}
"""
)
_MODULES = {"float": RawModuleCache(_SRC.replace("{REAL}", "float")), "double": RawModuleCache(_SRC.replace("{REAL}", "double"))}


def kernel(cp: Any, real: Any) -> Any:
    """``hinge_precheck_drops`` for columns of dtype ``real`` (float32 or float64); compile point of :func:`_drops_device` and of the register-pressure registry."""
    return _MODULES["float" if np.dtype(real) == np.float32 else "double"].get(cp).get_function("hinge_precheck_drops")


@njit(cache=True, parallel=True, nogil=True)
def _drops_njit(xs: np.ndarray, y: np.ndarray, qs: np.ndarray, min_seg: int, floor_var: float) -> np.ndarray:
    """Best fractional SSE drop of the three-cut hinge over the line for every row of ``xs`` ``(columns, n)``; ``-1`` for a column the precheck rejects outright. ``y`` is the raw float64 target."""
    k, n = xs.shape
    out = np.empty(k, dtype=np.float64)
    my = 0.0
    for i in range(n):
        my += y[i]
    my /= n
    yc = np.empty(n, dtype=np.float64)
    syy = 0.0
    for i in range(n):
        yc[i] = y[i] - my
        syy += yc[i] * yc[i]
    for j in prange(k):
        mean_x = 0.0
        for i in range(n):
            mean_x += xs[j, i]
        mean_x /= n
        sxx = 0.0
        sxy = 0.0
        for i in range(n):
            xv = xs[j, i] - mean_x
            sxx += xv * xv
            sxy += xv * yc[i]
        if sxx <= 0.0:
            out[j] = -1.0
            continue
        sse_lin = syy - sxy * sxy / sxx
        if not np.isfinite(sse_lin) or sse_lin <= floor_var:
            out[j] = -1.0
            continue
        cuts = np.quantile(xs[j], qs)
        best = 0.0
        for c in range(cuts.shape[0]):
            if c > 0 and cuts[c] == cuts[c - 1]:
                continue
            cnt = 0
            r1 = 0.0
            rx = 0.0
            rr = 0.0
            ry = 0.0
            for i in range(n):
                xv = xs[j, i] - mean_x
                r = xs[j, i] - cuts[c]
                if r > 0.0:
                    cnt += 1
                    r1 += r
                    rx += r * xv
                    rr += r * r
                    ry += r * yc[i]
            if cnt < min_seg or (n - cnt) < min_seg:
                continue
            denom = rr - r1 * r1 / n - rx * rx / sxx
            sse_h = sse_lin
            if not (denom < floor_var):
                num = ry - rx * sxy / sxx
                sse_h = sse_lin - num * num / denom
            if not np.isfinite(sse_h):
                continue
            drop = 1.0 - sse_h / sse_lin
            if drop > best:
                best = drop
        out[j] = best
    return out


def _drops_device(xs: np.ndarray, y: np.ndarray, qs: Sequence[float], min_seg: int, floor_var: float) -> Optional[np.ndarray]:
    """The device form of :func:`_drops_njit`, or ``None`` when cupy is absent or the device refuses (the caller runs the njit form)."""
    try:
        import cupy as cp
        from pyutilz.performance.gpu_hw_launch import block_size_for  # noqa: F401  (the block is fixed: warp-shuffle reductions need a multiple of 32)

        from ._gpu_resident_select import _radix_quantiles

        k, n = xs.shape
        xd = cp.asarray(xs)
        real = xd.dtype
        # the three cuts are order statistics: the sort-free radix select gives them (bit-identical to cp.percentile) without sorting the columns (the sort was 43 % of the device call)
        got_q = _radix_quantiles(cp.ascontiguousarray(xd.T), np.asarray(qs, dtype=np.float64))
        if got_q is None:
            return None
        cuts = got_q.T
        yc = np.empty(n + 1, dtype=np.float64)
        yc[:n] = np.asarray(y, dtype=np.float64) - float(np.mean(y))
        yc[n] = float(yc[:n] @ yc[:n])
        out = cp.empty(k, dtype=cp.float64)
        kernel(cp, real)(
            (k,), (_THREADS,),
            (xd, cp.asarray(yc), cp.ascontiguousarray(cuts), np.int64(n), np.int32(min_seg), np.float64(floor_var), out),
        )
        return np.asarray(cp.asnumpy(out), dtype=np.float64)
    except Exception as exc:
        logger.debug("hinge precheck: the device form is unavailable (%s: %s)", type(exc).__name__, exc)
        return None


def _drops_host(xs: np.ndarray, y: np.ndarray, qs: Sequence[float], min_seg: int, floor_var: float) -> np.ndarray:
    """Host form: the parallel numba kernel."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    with parallel_kernel_entry():
        out: np.ndarray = _drops_njit(np.ascontiguousarray(xs, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64), np.asarray(qs, dtype=np.float64), int(min_seg), float(floor_var))
    return out


DEVICE_FALLBACK_ROWS = 20_000  # the hand-set mark used until a sweep has run on this host
_SWEEP_ROWS = [5_000, 20_000, 50_000, 200_000]
_SWEEP_COLS = 64
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(columns, y)``: a block of ``_SWEEP_COLS`` kinked float64 columns of ``dims["n_rows"]`` rows and a target for the sweep."""
    rng = np.random.default_rng(0)
    n_rows = int(dims["n_samples"])
    xs = rng.normal(size=(_SWEEP_COLS, n_rows))
    y = np.maximum(xs[0] - 0.3, 0.0) + 0.1 * rng.normal(size=n_rows)
    return (xs, y)


def _sweep_host(xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The host form on the sweep inputs."""
    return _drops_host(xs, y, (0.3, 0.5, 0.7), 30, VARIANCE_FLOOR)


def _sweep_device(xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The device form on the sweep inputs."""
    got = _drops_device(xs, y, (0.3, 0.5, 0.7), 30, VARIANCE_FLOOR)
    if got is None:
        raise RuntimeError("device hinge precheck declined")
    return got


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="hinge_precheck_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-9),
        axes={"n_rows": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="hinge_precheck_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for hinge_precheck_backend failed: %s", e)
    _SPEC = None


def hinge_precheck_on_device(n_rows: int) -> bool:
    """Whether a block of columns of ``n_rows`` rows is prechecked on the device: ``MLFRAME_HINGE_PRECHECK_DEVICE=0|1`` forces a side, else the per-host kernel-tuning cache, else the row mark."""
    forced = os.environ.get("MLFRAME_HINGE_PRECHECK_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_rows=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("hinge_precheck_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


#: Host bytes of one column block (the float64 values of its columns); the device holds the block, its sorted copy and the cut table, about three times that.
_BLOCK_BYTES = 256 * 1024 * 1024


def column_block_width(n_rows: int) -> int:
    """Columns per column block: as many as ``_BLOCK_BYTES`` holds at ``n_rows`` rows of float64 (at least one)."""
    return max(1, _BLOCK_BYTES // max(1, 8 * int(n_rows)))


def hinge_precheck_drops(xs: np.ndarray, y: np.ndarray, qs: Sequence[float], min_seg: int, floor_var: float = VARIANCE_FLOOR) -> np.ndarray:
    """The best fractional SSE drop of the coarse hinge over the line for every row of the finite block ``xs`` ``(columns, n)`` against ``y`` (``-1`` for a column the precheck rejects outright), from the
    backend the kernel-tuning cache picks for this row count. The columns go through in column blocks of :func:`column_block_width`, so the memory of a call is bounded whatever the width of the frame."""
    n = int(xs.shape[1])
    width = column_block_width(n)
    on_device = hinge_precheck_on_device(n)
    parts = []
    for a in range(0, int(xs.shape[0]), width):
        block = xs[a : a + width]
        got = _drops_device(block, y, qs, min_seg, floor_var) if on_device else None
        parts.append(got if got is not None else _drops_host(block, y, qs, min_seg, floor_var))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
