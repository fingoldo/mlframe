"""Device twin of the rounding and digit-extraction columns of ``_numeric_decompose_fe`` (``apply_rounding``, ``apply_digit_extract``).

One launch makes every (source column, output) pair of a column block: output kind 0 is ``rint(x / precision) * precision`` (round half to even, the same as ``np.round``), kind 1 is the ``floor(x * 10^k) mod 10`` digit
with non-finite values mapped to 0. Both are per-element IEEE operations in double, so the columns equal the host's exactly. Source columns go to the device in column blocks sized from the room under the pool
(``_device_blocks``), halved when a block does not fit. The generators pick host or device per call from the kernel-tuning entry ``numeric_decompose_backend`` by the work ``n * outputs``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, cast

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

__all__ = ["decompose_columns_gpu", "decompose_columns_host", "decompose_device_chosen"]

_SRC_TEMPLATE = r"""
extern "C" __global__ __launch_bounds__(256)
void decompose_columns(const {REAL}* __restrict__ X, const long long n, const int n_out, const int* __restrict__ kinds, const {ACC}* __restrict__ params, {ACC}* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int o = blockIdx.y, col = blockIdx.z;
    const {REAL} xr = X[(long long)col * n + i];
    {ACC} r;
    if (kinds[o] == 0) {
        const {REAL} pr = ({REAL})params[o];
        r = ({ACC})(rint(xr / pr) * pr);                     // np.round(x / p) * p in the value type, NaN stays NaN
    } else {
        const {ACC} v = ({ACC})xr * params[o];               // the host loop multiplies in the accumulator type
        if (!isfinite(v)) { r = 0.0; }
        else {
            const {ACC} f = floor(v);
            if (fabs(f) < 1.0e15) {                              // an integer-valued value below 2^50: the integer floor-mod equals f - 10 * floor(f / 10) exactly, without the slow division of the wide type
                const long long fi = (long long)f;
                long long m = fi % 10;
                if (m < 0) m += 10;
                r = ({ACC})m;
            } else { r = f - 10.0 * floor(f / 10.0); }
        }
    }
    out[((long long)col * n_out + o) * n + i] = r;
}
"""
_SRC = render(_SRC_TEMPLATE, real=np.float32)  # the float32 build, the form the offline checks compile

_KERNELS: dict = {}
_THREADS = 256


def _kernel(cp: Any, real: Any) -> Any:
    """The compiled kernel for columns of value type ``real`` (the pipeline's criterion dtype)."""
    key = np.dtype(real)
    if key not in _KERNELS:
        _KERNELS[key] = cp.RawKernel(render(_SRC_TEMPLATE, real=real), "decompose_columns")
    return _KERNELS[key]


def decompose_columns_host(cols: Sequence[np.ndarray], kinds: Sequence[int], params: Sequence[float]) -> np.ndarray:
    """``(p, n_out, n)`` array: the host expressions of ``apply_rounding`` (kind 0, ``params`` a precision) and ``apply_digit_extract`` (kind 1, ``params`` a digit position)."""
    from ._numeric_decompose_fe import apply_digit_extract, apply_rounding

    out = np.empty((len(cols), len(kinds), len(cols[0]) if len(cols) else 0))
    for c, x in enumerate(cols):
        for o, (kind, par) in enumerate(zip(kinds, params)):
            out[c, o] = apply_rounding(x, par) if kind == 0 else apply_digit_extract(x, int(par))
    return out


def decompose_columns_gpu(cols: Sequence[np.ndarray], kinds: Sequence[int], params: Sequence[float]) -> np.ndarray:
    """The device twin of ``decompose_columns_host``; raises on a device fault so the caller can fall back to the host."""
    import cupy as cp

    from ._device_blocks import first_block, run_column_blocks

    p = len(cols)
    n_out = len(kinds)
    n = len(cols[0]) if p else 0
    result = np.empty((p, n_out, n))
    if p == 0 or n == 0:
        return result
    kinds_d = cp.asarray(np.asarray(kinds, dtype=np.int32))
    # digit kinds carry the scale 10^k, rounding kinds the precision
    par_d = cp.asarray(np.asarray([float(par) if kind == 0 else 10.0 ** int(par) for kind, par in zip(kinds, params)], dtype=np.float64))  # Intentional float64: parameters of a double computation
    from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

    real = _crit_np_dtype()
    kernel = _kernel(cp, real)

    def block_work(lo: int, hi: int) -> int:
        """Columns ``lo:hi`` of every output written into ``result``."""
        xb = cp.asarray(np.stack([np.ascontiguousarray(c, dtype=real) for c in cols[lo:hi]]))
        ob = cp.empty((hi - lo, n_out, n), dtype=cp.float64)  # Intentional float64: the output equals the double host column
        kernel(((n + _THREADS - 1) // _THREADS, n_out, hi - lo), (_THREADS,), (xb, np.int64(n), np.int32(n_out), kinds_d, par_d, ob))  # launch-ok: rows on x, outputs on y, columns on z
        result[lo:hi] = cp.asnumpy(ob)
        return hi - lo

    block = first_block(cp, n, p, 8, 2 + n_out)
    run_column_blocks(cp, p, block, block_work)
    return result


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 400_000_000  # column-rows x outputs from which the device makes the columns before a sweep has run (the host won at every size swept, see the profile record)
_SWEEP_N = 100_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (7, 28, 112, 448)]
_SWEEP_KINDS = (0, 0, 0, 0, 1, 1, 1)
_SWEEP_PARAMS = (1.0, 0.1, 0.01, 0.001, 0, 1, 2)
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(cols,)``: enough 100000-row columns that columns x 7 outputs reach ``dims["n_samples"]`` column-rows, some values non-finite."""
    rng = np.random.default_rng(0)
    p = max(1, int(dims["n_samples"]) // (_SWEEP_N * len(_SWEEP_KINDS)))
    cols = [rng.standard_normal(_SWEEP_N) * 100.0 for _ in range(p)]
    cols[0][::97] = np.nan
    return (cols,)


def _sweep_host(cols: list) -> np.ndarray:
    """The host columns, summarised for the comparison (sum per output of the finite values)."""
    out = decompose_columns_host(cols, _SWEEP_KINDS, _SWEEP_PARAMS)
    return cast(np.ndarray, np.nansum(out, axis=(0, 2)))


def _sweep_device(cols: list) -> np.ndarray:
    """The device kernel; a failed launch raises, so the sweep never records a host fallback as a device time."""
    out = decompose_columns_gpu(cols, _SWEEP_KINDS, _SWEEP_PARAMS)
    return cast(np.ndarray, np.nansum(out, axis=(0, 2)))


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="numeric_decompose_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-9),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="numeric_decompose_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for numeric_decompose_backend failed: %s", e)
    _SPEC = None


def decompose_device_chosen(n: int, p: int, n_out: int) -> bool:
    """Whether ``p`` columns over ``n`` rows make their ``n_out`` outputs on the device: the per-host cache by the work ``n * p * n_out``, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(p) * int(n_out)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        log_throttle(logger, "numeric_decompose.choose_failed", logging.WARNING, "numeric_decompose_backend: KTC choose() failed (%s: %s); the %d-column-row floor decides", type(exc).__name__, exc, DEVICE_FALLBACK_ROWS)
        return work >= DEVICE_FALLBACK_ROWS
