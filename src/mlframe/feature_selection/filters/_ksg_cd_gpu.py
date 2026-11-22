"""Device twin of ``_ksg_cd_fast._mi_cd_columns``: the Ross continuous-versus-discrete k-NN MI of many one-dimensional columns.

The host kernel walks, for every point of a column sorted by value, outward through the points of its own class to find the k-th neighbour distance ``r`` and then counts the points of
ALL classes strictly closer than ``r``. Here the columns are sorted on the device (one batched argsort), and one thread takes one point of one column: it walks the sorted column
skipping the points of other classes (the neighbours of a point sit next to it, so adjacent threads read adjacent memory), then runs the host's exact widen-then-trim count with the same
comparisons. Only ``sum psi(m)`` depends on the data; ``sum psi(k)`` and ``sum psi(class count)`` are class sums formed on the host. Each block writes its partial sum of ``psi(m)``
and the partials are added in a fixed order, so a run is reproducible; they differ from the host's sequential sum only by floating-point rounding (measured: 7e-12 absolute at 100000 rows).

Rows of classes with a single member are dropped before the sort (the host drops them from the sorted list, which leaves the order of the others unchanged). Columns go to the device in column blocks
that fit ``_BUDGET_FRACTION`` of the free memory; on any device fault the caller runs the host kernel.

The kernel-tuning cache decides by row count (fallback ``DEVICE_FALLBACK_ROWS``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np


from ._device_sort_safe import device_argsort, device_room, device_exclusive
from ._gpu_resident_select_kernels import _transpose_to_cm
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._kernel_types import render
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 20_000
_BLOCKS_PER_COLUMN = 64

_SRC = r"""
#define POS_INF __longlong_as_double(0x7ff0000000000000LL)
// {REAL} is the column type: 64 bits by default (the host kernel works on noised 64-bit columns and the distance comparisons must match it), 32 bits for a column block that is tie-free and finite after the cast (see mi_cd_columns_device); {ACC} is the accumulator type.
extern "C" __global__ void ksg_psi_m_sums(const {REAL}* __restrict__ xs, const {CT}* __restrict__ ds, const long long nk, const int* __restrict__ counts,
                                          const int nn, const {ACC}* __restrict__ psi, {ACC}* __restrict__ partial) {
    __shared__ {ACC} red[256];
    const int col = blockIdx.y;
    const {REAL}* x = xs + (long long)col * nk;
    const {CT}* d = ds + (long long)col * nk;
    {ACC} acc = 0.0;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < nk; j += (long long)gridDim.x * blockDim.x) {
        const int c = d[j];
        const int cnt = counts[c];
        const int k = min(nn, cnt - 1);
        const {REAL} v = x[j];
        long long lo = j - 1, hi = j + 1;
        {REAL} r = 0.0;
        for (int t = 0; t < k; ++t) {
            while (lo >= 0 && d[lo] != c) --lo;
            while (hi < nk && d[hi] != c) ++hi;
            const {REAL} dl = lo >= 0 ? v - x[lo] : POS_INF;
            const {REAL} dh = hi < nk ? x[hi] - v : POS_INF;
            if (dl <= dh) { r = dl; --lo; } else { r = dh; ++hi; }
        }
        const {REAL} lo_key = v - r, hi_key = v + r;
        long long lo_m = j;
        while (lo_m > 0 && x[lo_m - 1] >= lo_key) --lo_m;
        long long hi_m = j + 1;
        while (hi_m < nk && x[hi_m] <= hi_key) ++hi_m;
        while (lo_m < hi_m && v - x[lo_m] >= r) ++lo_m;
        while (hi_m > lo_m && x[hi_m - 1] - v >= r) --hi_m;
        acc += psi[hi_m - lo_m];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[(long long)col * gridDim.x + blockIdx.x] = red[0];
}
"""

# {CT} is the type of the class labels of the sorted rows: ``int`` (the original) or ``signed char`` (1 byte per row instead of 4: the neighbour walks compare a label per step, and the labels are the second
# stream the kernel reads; the classes of a screening target are a handful). Both are compiled for both column types; the ``int`` instances are the ones the kernel always had.
_C_TYPES: "dict[np.dtype, str]" = {np.dtype(np.int32): "int", np.dtype(np.int8): "signed char"}
_REAL_TYPES: "tuple[type[np.floating], ...]" = (np.float64, np.float32)  # Intentional float64: both column types are compiled
_MODULES = {
    (np.dtype(real), np.dtype(ct)): RawModuleCache(render(_SRC, real=real).replace("{CT}", c_name))  # Intentional float64: the default column type (see the comment in the kernel); {ACC} is the 64-bit accumulator
    for real in _REAL_TYPES
    for ct, c_name in _C_TYPES.items()
}


def kernel(cp: Any, real: Any = np.float64, classes: Any = np.int32) -> Any:  # Intentional float64: selectable double-precision variant of the kernel (default and environment switch)
    """``ksg_psi_m_sums`` for 64-bit (default) or 32-bit columns and ``int32`` (default) or ``int8`` class labels; compile point of :func:`mi_cd_columns_device` and of the register-pressure registry."""
    module = _MODULES.get((np.dtype(real), np.dtype(classes)))
    if module is None:
        raise ValueError("ksg_psi_m_sums is instantiated for float32 / double-precision columns and int32 / int8 class labels only")
    return module.get(cp).get_function("ksg_psi_m_sums")


def kernel_small_classes(cp: Any, real: Any = np.float64) -> Any:  # Intentional float64: selectable double-precision variant of the kernel
    """The ``int8``-label instance of :func:`kernel` (a factory of its own for the register-pressure registry, which compiles one entry per factory)."""
    return kernel(cp, real, np.int8)


def _i8_enabled() -> bool:
    """Whether the class labels of the sorted rows are read as one byte (``MLFRAME_KSG_I8=0`` keeps the four-byte ``int`` labels of the original kernel instance)."""
    import os

    return os.environ.get("MLFRAME_KSG_I8", "1").strip().lower() not in ("0", "false", "off", "no")

_THREADS = 256


def _f32_enabled() -> bool:
    """Whether tie-free, finite column blocks are scored in float32 (``MLFRAME_KSG_F32=0`` turns it off). Float32 halves the upload and the sort of a block; it cannot hold the 1e-10 Kraskov noise, so a column whose
    float32 values contain equal neighbours (or a non-finite value) is scored in double precision instead (see :func:`_block_sums`)."""
    import os

    return os.environ.get("MLFRAME_KSG_F32", "1").strip().lower() not in ("0", "false", "off", "no")


def _tie_pair_bound(n_rows: int) -> int:
    """Largest number of equal neighbours a column may have after the float32 cast and still be scored in float32. A continuous column has about ``1e-8 * n^2`` chance collisions of the 24-bit mantissa (measured:
    rate 1.1e-8 * n at 30000 rows, 0.85e-8 * n at 100000, at most 2 pairs at 5000) which move the MI by about 1e-4; a discretised column has a tie rate of 0.67 to 0.97. The bound is four times the chance level, with a floor of 8."""
    return max(8, int(4e-8 * float(n_rows) * float(n_rows)))


def _packed_f32_order(cp: Any, xd: Any) -> Any:
    """``(width, rows)`` int64 row order that sorts every row of the float32 matrix ``xd`` ascending, from ONE radix sort: each value becomes the 64-bit key ``column | order-preserving float bits | row``, so the
    columns stay in contiguous segments and the low bits of the sorted keys are the row ids. A per-column ``argsort`` of 100 columns x 30000 rows took 28 ms, this 10 ms (equal sorted values). ``None`` when the
    three fields do not fit 64 bits (rows beyond 2^22 with 1000 columns), and the caller sorts per column."""
    width, rows = int(xd.shape[0]), int(xd.shape[1])
    row_bits = max(1, (rows - 1).bit_length())
    col_bits = max(1, (width - 1).bit_length())
    if 32 + row_bits + col_bits > 64:
        return None
    bits = xd.view(cp.uint32)
    flip = cp.where((bits >> 31) != 0, cp.uint32(0xFFFFFFFF), cp.uint32(0x80000000))
    key = (bits ^ flip).astype(cp.uint64)
    cols = (cp.arange(width, dtype=cp.uint64) << cp.uint64(32 + row_bits))[:, None]
    packed = (cols | (key << cp.uint64(row_bits)) | cp.arange(rows, dtype=cp.uint64)[None, :]).ravel()
    packed = cp.sort(packed)
    return (packed & cp.uint64((1 << row_bits) - 1)).astype(cp.int64).reshape(width, rows)


def _block_sums(cp: Any, host_block: np.ndarray, real: Any, d_dev: Any, counts_dev: Any, psi_dev: Any, n_kept: int, n_neighbors: int, partial_buf: Any) -> "tuple[Any, Any]":
    """``(sum psi(m) per column, columns to redo)`` of one ``(rows, width)`` host block scored with columns of type ``real``: uploaded, transposed on the device, sorted and run through the kernel.

    For float32 the second value is a device boolean per column that is True where the cast column has more equal neighbours than :func:`_tie_pair_bound` or a non-finite entry (the noise that breaks ties is lost in
    float32), and such a column's sum is not valid; for double precision it is ``None``."""
    # one host copy of the block's columns (a plain row-wise copy, none when the block is the whole array), one upload, then the (rows, columns) to (columns, rows) transpose on the device with the coalesced tiled kernel;
    # the host transpose of this block took 180 ms of a 308 ms call at 100 columns x 100000 rows
    xd = _transpose_to_cm(cp.asarray(np.ascontiguousarray(host_block, dtype=real)))
    order = _packed_f32_order(cp, xd) if np.dtype(real) == np.float32 else None
    if order is None:
        order = device_argsort(xd, axis=1)
    xs = cp.take_along_axis(xd, order, axis=1)  # already C-contiguous (checked directly); the extra ascontiguousarray was a redundant copy
    ds = d_dev[order]
    del xd
    width = xs.shape[0]
    redo = None
    if np.dtype(real) == np.float32:
        redo = ((xs[:, 1:] == xs[:, :-1]).sum(axis=1) > _tie_pair_bound(n_kept)) | ~cp.isfinite(xs).all(axis=1)
    partial = partial_buf[:width]
    kernel(cp, real, d_dev.dtype)(  # launch-ok: _THREADS=256 is the compile-time size of the kernel's `__shared__ double red[256]` reduction buffer, not a tuned size
        (_BLOCKS_PER_COLUMN, width), (_THREADS,), (xs, ds, np.int64(n_kept), counts_dev, np.int32(n_neighbors), psi_dev, partial)
    )
    return partial.sum(axis=1), redo


@device_exclusive
def mi_cd_columns_device(X: np.ndarray, d: np.ndarray, n_classes: int, n_neighbors: int) -> Optional[np.ndarray]:
    """Ross MI (nats) per column of the noised, scaled ``X`` (rows, columns) with integer labels ``d``; ``None`` when the device cannot take it (the caller runs the host kernel)."""
    n, n_cols = X.shape
    if n == 0 or n_cols == 0:
        return None
    try:
        import cupy as cp

        counts = np.bincount(d, minlength=n_classes).astype(np.int64)
        keep = counts[d] > 1
        n_kept = int(keep.sum())
        if n_kept == 0:
            return np.zeros(n_cols)
        # psi at integers 1..n via psi(m+1) = psi(m) + 1/m, as the host table
        psi = np.zeros(n + 2)
        psi[1] = -0.5772156649015329
        psi[2:] = psi[1] + np.cumsum(1.0 / np.arange(1, n + 1))
        multi = counts > 1
        k_c = np.minimum(n_neighbors, counts - 1)
        sum_k = float((counts[multi] * psi[k_c[multi]]).sum())
        sum_cnt = float((counts[multi] * psi[counts[multi]]).sum())
        d_dev = cp.asarray(np.ascontiguousarray(d[keep], dtype=np.int8 if (_i8_enabled() and int(n_classes) <= 127) else np.int32))
        counts_dev = cp.asarray(counts.astype(np.int32))
        psi_dev = cp.asarray(psi)
        f32 = _f32_enabled()
        free = device_room(cp)
        per_col = n_kept * (8 + 8 + 8 + 8 + 4)  # row-major upload, transposed values, sort order, sorted values, sorted labels
        block = int(max(1, min(n_cols, (_BUDGET_FRACTION * free) // per_col)))
        X_kept = X if n_kept == n else X[keep]
        partial_buf = cp.empty((block, _BLOCKS_PER_COLUMN), dtype=cp.float64)  # one reusable kernel-output buffer for every chunk, not one per chunk  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
        # Intentional float64: the resident accumulator of the statistic, kept in double precision; one D2H after the loop instead of one per chunk
        sum_m_dev = cp.empty(n_cols, dtype=cp.float64)
        for c0 in range(0, n_cols, block):
            host_block = X_kept[:, c0 : c0 + block]
            width = host_block.shape[1]
            # Intentional float64: the double-precision fallback runs when float32 cannot hold the column (the f32 path is the default)
            sums, redo = _block_sums(cp, host_block, np.float32 if f32 else np.float64, d_dev, counts_dev, psi_dev, n_kept, n_neighbors, partial_buf)
            sum_m_dev[c0 : c0 + width] = sums
            if redo is not None:
                redo_idx = np.flatnonzero(cp.asnumpy(redo))  # one small read per block: the columns float32 cannot hold are scored again in double precision
                if redo_idx.size:
                    # Intentional float64: the redo of the columns float32 cannot hold, in double precision
                    sums64, _ = _block_sums(cp, host_block[:, redo_idx], np.float64, d_dev, counts_dev, psi_dev, n_kept, n_neighbors, partial_buf)
                    sum_m_dev[c0 + cp.asarray(redo_idx)] = sums64
        sum_m = cp.asnumpy(sum_m_dev)
        return np.asarray(np.maximum(psi[n_kept] + (sum_k - sum_cnt - sum_m) / n_kept, 0.0), dtype=np.float64)  # Intentional float64: statistic / reference vector in double precision (compared exactly with its host twin)
    except Exception as exc:
        log_throttle(logger, "ksg_cd_gpu.failed", logging.WARNING, "device KSG MI failed (%s: %s); the host kernel scores the columns", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A noised 60-column block of ``dims["n_samples"]`` rows with a 3-class label."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    X = rng.standard_normal((n, 60))
    d = rng.integers(0, 3, n).astype(np.int64)
    X[:, :10] += d[:, None] * 0.5
    return (X, d)


def _host(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The host kernel's MI per column."""
    from ._ksg_cd_fast import _mi_cd_columns

    return np.asarray(_mi_cd_columns(np.ascontiguousarray(X), d, int(d.max()) + 1, 3))


def _device(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The device kernel's MI per column; a failed launch raises so the sweep does not record a host fallback as a device time."""
    out = mi_cd_columns_device(X, d, int(d.max()) + 1, 3)
    if out is None:
        raise RuntimeError("device KSG unavailable")
    return out


_run_sweep = host_device_sweep(__name__, _host, _device, repeats=3, rtol=1e-3, atol=5e-4)  # the sweep block is tie-free, so the device runs float32 columns: the MI agrees to the rounding of the data (measured 1.3e-4)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


_fallback_choice = rows_fallback(__name__)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_ksg_cd_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_ksg_cd_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_ksg_cd_device failed: %s", e)
    _SPEC = None


from ._ktc_fallback import ktc_device_chooser as _ktc_device_chooser

use_device = _ktc_device_chooser(__name__, "fe_ksg_cd_device")


__all__ = ["mi_cd_columns_device", "use_device"]
