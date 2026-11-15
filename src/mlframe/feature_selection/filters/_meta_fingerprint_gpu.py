"""Column moments of the meta scorer's fingerprint: a numba kernel, a fused CUDA kernel and the per-host choice between them.

``_orthogonal_meta_scorer_fe.fingerprint_signal`` summarises a frame by ``mean_abs_skew``, ``mean_kurtosis`` and ``mean_abs_pearson``. The vectorised host form built three ``(n, K)`` temporaries
(``c``, ``c * c``, ``c * c * c``, ``c * c * c * c``) and a matrix product per statistic: seven passes over the matrix, two of them again for the correlation. Everything those statistics read is five sums
of a column about its mean, so one two-pass kernel per column gives them all:
``mean``, ``m2 = mean(c^2)``, ``m3 = mean(c^3)``, ``m4 = mean(c^4)`` (``c = x - mean``) and ``cxy = sum(c * yc)`` (``yc`` the centred target). The host keeps the cheap per-column formulas
(``G1`` / ``G2`` of pandas, ``r = cxy / sqrt(sum(c^2) * syy)``) on the ``K``-vectors.

Two backends: ``_moments_njit`` (``prange`` over columns) and ``column_moments`` of the device (``column_moments_blocks``: a CUDA block per column, warp-shuffle reductions, the column read twice coalesced).
``fingerprint_moments_on_device`` reads the per-host kernel-tuning cache (``meta_fingerprint_backend``, a sweep over the row count) with a row-count fallback; ``MLFRAME_META_FINGERPRINT_DEVICE=0|1`` forces a side.
The columns go through in column blocks of ``_BLOCK_BYTES``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
from numba import njit, prange

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

_SRC = render(
    r"""
extern "C" __global__ void column_moments_blocks(const {REAL}* __restrict__ x, const {ACC}* __restrict__ yc, const long long n, const int with_y, {ACC}* __restrict__ out) {
    // one block per column: pass 1 the mean, pass 2 the sums of c^2, c^3, c^4 and c * yc; out[col * 5 + {0: mean, 1: m2, 2: m3, 3: m4, 4: cxy}]
    const long long col = blockIdx.x;
    const {REAL}* xc = x + col * n;
    __shared__ {ACC} red[32][5];
    {ACC} s = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) s += ({ACC})xc[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][0] = s;
    __syncthreads();
    __shared__ {ACC} mean;
    if (threadIdx.x == 0) { {ACC} t = 0; for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w][0]; mean = t / ({ACC})n; }
    __syncthreads();
    {ACC} a2 = 0, a3 = 0, a4 = 0, ay = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const {ACC} c = ({ACC})xc[i] - mean;
        const {ACC} c2 = c * c;
        a2 += c2; a3 += c2 * c; a4 += c2 * c2;
        if (with_y) ay += c * yc[i];
    }
    for (int o = 16; o > 0; o >>= 1) {
        a2 += __shfl_down_sync(0xffffffffu, a2, o); a3 += __shfl_down_sync(0xffffffffu, a3, o);
        a4 += __shfl_down_sync(0xffffffffu, a4, o); ay += __shfl_down_sync(0xffffffffu, ay, o);
    }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5][1] = a2; red[threadIdx.x >> 5][2] = a3; red[threadIdx.x >> 5][3] = a4; red[threadIdx.x >> 5][4] = ay; }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} t2 = 0, t3 = 0, t4 = 0, ty = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) { t2 += red[w][1]; t3 += red[w][2]; t4 += red[w][3]; ty += red[w][4]; }
        out[col * 5 + 0] = mean; out[col * 5 + 1] = t2 / ({ACC})n; out[col * 5 + 2] = t3 / ({ACC})n; out[col * 5 + 3] = t4 / ({ACC})n; out[col * 5 + 4] = ty;
    }
}
"""
)
_MODULES = {"float": RawModuleCache(_SRC.replace("{REAL}", "float")), "double": RawModuleCache(_SRC.replace("{REAL}", "double"))}


def kernel(cp: Any, real: Any) -> Any:
    """``column_moments_blocks`` for columns of dtype ``real`` (float32 or float64); compile point of :func:`_moments_device` and of the register-pressure registry."""
    return _MODULES["float" if np.dtype(real) == np.float32 else "double"].get(cp).get_function("column_moments_blocks")


@njit(cache=True, nogil=True)
def _column_moments(row: np.ndarray, yc: np.ndarray, with_y: bool, out: np.ndarray) -> None:
    """The five sums of one column (see the module docstring) into ``out`` (length 5)."""
    n = row.shape[0]
    mean = 0.0
    for i in range(n):
        mean += row[i]
    mean /= n
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    ay = 0.0
    for i in range(n):
        c = row[i] - mean
        c2 = c * c
        a2 += c2
        a3 += c2 * c
        a4 += c2 * c2
        if with_y:
            ay += c * yc[i]
    out[0] = mean
    out[1] = a2 / n
    out[2] = a3 / n
    out[3] = a4 / n
    out[4] = ay


@njit(cache=True, parallel=True, nogil=True)
def _moments_njit(xs: np.ndarray, yc: np.ndarray, with_y: bool) -> np.ndarray:
    """``(K, 5)`` moments of the rows of ``xs`` ``(K, n)``: mean, m2, m3, m4 about the mean and the cross sum with the centred target."""
    k = xs.shape[0]
    out = np.empty((k, 5), dtype=np.float64)
    for j in prange(k):
        _column_moments(xs[j], yc, with_y, out[j])
    return out


def _moments_host(xs: np.ndarray, yc: Optional[np.ndarray]) -> np.ndarray:
    """Host form: the parallel numba kernel."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    y = np.zeros(xs.shape[1], dtype=np.float64) if yc is None else np.ascontiguousarray(yc, dtype=np.float64)
    with parallel_kernel_entry():
        out: np.ndarray = _moments_njit(np.ascontiguousarray(xs), y, yc is not None)
    return out


def _moments_device(xs: np.ndarray, yc: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Device form: one launch for the block, or ``None`` (no cupy, a device fault)."""
    try:
        import cupy as cp
        from pyutilz.performance.gpu_hw_launch import block_size_for

        k, n = xs.shape
        real = np.float32 if xs.dtype == np.float32 else np.float64
        ker = kernel(cp, real)
        y = np.zeros(n, dtype=np.float64) if yc is None else np.ascontiguousarray(yc, dtype=np.float64)
        out = cp.empty((k, 5), dtype=cp.float64)
        ker((k,), (block_size_for(ker, work_items=n),), (cp.asarray(np.ascontiguousarray(xs, dtype=real)), cp.asarray(y), np.int64(n), np.int32(yc is not None), out))
        return np.asarray(cp.asnumpy(out), dtype=np.float64)
    except Exception as exc:
        logger.debug("meta fingerprint: the device form is unavailable (%s: %s)", type(exc).__name__, exc)
        return None


DEVICE_FALLBACK_ROWS = 20_000
_SWEEP_ROWS = [5_000, 20_000, 50_000, 200_000]
_SWEEP_COLS = 100
_SALT = 1
#: Host bytes of one column block (the values of its columns in float64).
_BLOCK_BYTES = 256 * 1024 * 1024


def _sweep_inputs(dims: dict) -> tuple:
    """``(xs, yc)``: ``_SWEEP_COLS`` skewed columns of ``dims["n_samples"]`` rows and a centred target."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    xs = rng.standard_t(5, size=(_SWEEP_COLS, n))
    y = xs[0] + rng.normal(size=n)
    return (xs, y - y.mean())


def _sweep_host(xs: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """The host form on the sweep inputs."""
    return _moments_host(xs, yc)


def _sweep_device(xs: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """The device form on the sweep inputs."""
    got = _moments_device(xs, yc)
    if got is None:
        raise RuntimeError("device meta fingerprint declined")
    return got


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="meta_fingerprint_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-8, atol=1e-8),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="meta_fingerprint_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for meta_fingerprint_backend failed: %s", e)
    _SPEC = None


def fingerprint_moments_on_device(n_rows: int) -> bool:
    """Whether a block of columns of ``n_rows`` rows is summarised on the device: ``MLFRAME_META_FINGERPRINT_DEVICE=0|1`` forces a side, else the per-host kernel-tuning cache, else the row mark."""
    forced = os.environ.get("MLFRAME_META_FINGERPRINT_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("meta_fingerprint_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


def column_block_width(n_rows: int) -> int:
    """Columns per column block: as many as ``_BLOCK_BYTES`` holds at ``n_rows`` rows of float64 (at least one)."""
    return max(1, _BLOCK_BYTES // max(1, 8 * int(n_rows)))


def column_moments(xs: np.ndarray, yc: Optional[np.ndarray] = None) -> np.ndarray:
    """``(K, 5)`` moments (mean, m2, m3, m4 about the mean, cross sum with ``yc``; zero when ``yc`` is None) of the rows of the finite block ``xs`` ``(K, n)``, from the backend the kernel-tuning cache
    picks for this row count, in column blocks of :func:`column_block_width`."""
    n = int(xs.shape[1])
    width = column_block_width(n)
    on_device = fingerprint_moments_on_device(n)
    parts = []
    for a in range(0, int(xs.shape[0]), width):
        block = xs[a : a + width]
        got = _moments_device(block, yc) if on_device else None
        parts.append(got if got is not None else _moments_host(block, yc))
    return np.concatenate(parts) if parts else np.empty((0, 5), dtype=np.float64)
