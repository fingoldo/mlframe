"""Device twin of ``_orthogonal_tail_dependence_fe._tail_dep_columns_njit``: the floored tail dependence of many columns against one rank-uniformised target.

The host kernel sorts a column, walks its runs of equal values and reads both tails off the average rank of each run. The statistic depends only on the runs and on how many rows of each
run fall in the target's tails, never on the order of rows inside a run.

Versions (all kept; every one is bit-identical to the host kernel, because everything compared or counted is an integer count and the final arithmetic is done on the host in double precision in the
host kernel's order):

* ``_block_counts_packed`` (float32 columns, the default): one kernel (``pack_keys``) packs the target's two tail flags are packed into the low bits of a 64-bit key together with the order-preserving bits of the value and
  the column number, one flat array of keys is sorted with no payload, and ONE kernel (``tail_counts``, a block per column segment) walks the sorted keys, finds each run, forms its average rank in
  double as the host does and adds the four counts in shared memory. Win axis: the sort is the only heavy step; there is no permutation of the target, no gather and no intermediate table.
* ``_block_counts_chain`` (double precision columns, and the reference the tests compare against): ``argsort`` of each column, then about twenty array operations with ``(b, n)`` int64 tables in
  global memory. Slower; kept for double precision input, which the 64-bit key cannot hold.

Columns go to the device in column blocks that fit ``_BUDGET_FRACTION`` of the room the allocator has left; on any device fault the caller runs the host kernel. The kernel-tuning cache decides by
row count (fallback ``DEVICE_FALLBACK_ROWS``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from pyutilz.performance.gpu_hw_launch import block_size_for

from ._device_sort_safe import device_argsort, device_room, sort_room, device_exclusive
from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 12_000  # measured on a GTX 1050 Ti, 100 float32 columns, host kernel against the packed device path with the per-call upload: 0.41x at 2000 rows, 0.73x at 5000, 0.92x at 10000, 1.29x at 20000, 2.61x at 100000, 3.96x at 1000000
_BYTES_PER_ELEMENT_CHAIN = 144  # per column element: values, sort order, permuted target, run numbers and bounds (int64), the rank and the masks
_BYTES_PER_ELEMENT_PACKED = 48  # per column element: the float32 value, the 64-bit key sorted in place and the sort's scratch

_SRC = r"""
// The average rank of a run is formed in the {ACC} type exactly as the host kernel forms it (0.5 * (first + 1 + end) / (n + 1)) because it is compared with the tail quantile; no pointer is of that type
// (keys are 64-bit integers, counts are int).
// A key is (column << 34) | (order-preserving bits of the {REAL} value << 2) | (target above the upper tail << 1) | (target below the lower tail).
// The packed path takes {REAL} columns only (the project's critical dtype): the 64-bit key has room for 32 value bits, 2 flag bits and the column number, not for a 64-bit value.
extern "C" __global__ void pack_keys(const {REAL}* __restrict__ x, const unsigned long long* __restrict__ vflag, const long long n, const long long total, unsigned long long* __restrict__ keys) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const long long c = t / n;
    const long long r = t - c * n;
    {REAL} f = x[t];
    if (f == 0.0f) f = 0.0f;  // -0.0 and +0.0 are one value for the host, so they must be one key
    const unsigned int bits = __float_as_uint(f);
    const unsigned int ordered = bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);  // order-preserving unsigned image of a finite {REAL}
    keys[t] = ((unsigned long long)c << 34) | ((unsigned long long)ordered << 2) | vflag[r];
}

extern "C" __global__ void tail_counts(const unsigned long long* __restrict__ keys, const long long n, const {ACC} hi, const {ACC} lo, const int b, const long long seg, int* __restrict__ out) {
    const int c = blockIdx.x;
    const long long begin = (long long)blockIdx.y * seg;
    long long stop = begin + seg;
    if (stop > n) stop = n;
    const unsigned long long* col = keys + (long long)c * n;
    int n_u_hi = 0, both_hi = 0, n_u_lo = 0, both_lo = 0;
    // a run belongs to its first element, wherever the segments cut the column; the walk to the end of a run may leave the segment, which only reads
    for (long long i = begin + threadIdx.x; i < stop; i += blockDim.x) {
        const unsigned long long k0 = col[i] >> 2;
        if (i > 0 && (col[i - 1] >> 2) == k0) continue;  // inside a run: its first element owns it
        long long e = i;
        int cnt_hi = 0, cnt_lo = 0;
        while (e < n && (col[e] >> 2) == k0) {
            cnt_hi += (int)((col[e] >> 1) & 1ULL);
            cnt_lo += (int)(col[e] & 1ULL);
            ++e;
        }
        const {ACC} u = 0.5 * ({ACC})(i + 1 + e) / (({ACC})n + 1.0);
        if (u > hi) { n_u_hi += (int)(e - i); both_hi += cnt_hi; }
        if (u < lo) { n_u_lo += (int)(e - i); both_lo += cnt_lo; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        n_u_hi += __shfl_down_sync(0xffffffffu, n_u_hi, o);
        both_hi += __shfl_down_sync(0xffffffffu, both_hi, o);
        n_u_lo += __shfl_down_sync(0xffffffffu, n_u_lo, o);
        both_lo += __shfl_down_sync(0xffffffffu, both_lo, o);
    }
    if ((threadIdx.x & 31) == 0) {                          // integer adds: the totals do not depend on the order of the blocks and warps
        atomicAdd(&out[0 * b + c], n_u_hi);
        atomicAdd(&out[1 * b + c], both_hi);
        atomicAdd(&out[2 * b + c], n_u_lo);
        atomicAdd(&out[3 * b + c], both_lo);
    }
}
"""
_SRC = render(_SRC, real=np.float32)  # the packed path takes float32 columns only; {ACC} is double (the rank is compared with the tail quantile)

_MODULE = RawModuleCache(_SRC)


def _block_counts_chain(cp, x, v_d, n: int, hi: float, lo: float) -> np.ndarray:
    """``(4, b)`` counts of one block ``x`` ``(b, n)`` (reference version): rows in the column's upper tail, of those also in the target's upper tail, then the same for the lower tail."""
    b = int(x.shape[0])
    order = device_argsort(x, axis=1)
    s = cp.take_along_axis(x, order, axis=1)
    vs = v_d[order]
    del order
    starts = cp.ones((b, n), dtype=cp.bool_)
    starts[:, 1:] = s[:, 1:] != s[:, :-1]
    del s
    flat = starts.ravel()
    run_of = cp.cumsum(flat, dtype=cp.int64) - 1
    pos = cp.flatnonzero(flat)
    pos_ext = cp.concatenate([pos, cp.asarray([b * n], dtype=cp.int64)])
    row_off = (cp.arange(b, dtype=cp.int64) * n)[:, None]
    first = pos[run_of].reshape(b, n) - row_off
    end = pos_ext[run_of + 1].reshape(b, n) - row_off
    del run_of, pos, pos_ext, flat, starts
    u = 0.5 * (first + 1 + end) / (n + 1.0)  # the host's order: 0.5 * (first + 1 + end), then the division
    u_hi = u > hi
    u_lo = u < lo
    return np.asarray(cp.stack([u_hi.sum(axis=1), (u_hi & (vs > hi)).sum(axis=1), u_lo.sum(axis=1), (u_lo & (vs < lo)).sum(axis=1)]).get(), dtype=np.int64)


_MIN_SEGMENT_ROWS = 4096  # a block is not worth fewer rows than this
_BLOCKS_PER_SM = 8


def _segments_for(cp, b: int, n: int) -> int:
    """Segments per column so that ``b * segments`` blocks fill the card (``_BLOCKS_PER_SM`` per multiprocessor), none shorter than ``_MIN_SEGMENT_ROWS`` rows."""
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    want = -(-_BLOCKS_PER_SM * sms // max(1, b))
    return int(max(1, min(want, n // _MIN_SEGMENT_ROWS)))


def _block_counts_packed(cp, x, vflag, n: int, hi: float, lo: float) -> np.ndarray:
    """``(4, b)`` counts of one float32 block ``x`` ``(b, n)``: pack, sort the flat keys without a payload, count in one kernel."""
    b = int(x.shape[0])
    module = _MODULE.get(cp)
    pack = module.get_function("pack_keys")
    total = b * n
    keys = cp.empty(total, dtype=cp.uint64)
    threads = block_size_for(pack, work_items=total)
    pack(((total + threads - 1) // threads,), (threads,), (cp.ascontiguousarray(x), vflag, np.int64(n), np.int64(total), keys))
    with sort_room(keys):
        keys.sort()  # in place: the unsorted keys are not needed again, so no copy of the array is made
    sorted_keys = keys
    kernel = module.get_function("tail_counts")
    out = cp.zeros((4, b), dtype=cp.int32)
    threads = block_size_for(kernel, work_items=n)
    segments = _segments_for(cp, b, n)
    seg = -(-n // segments)
    kernel((b, segments), (threads,), (sorted_keys, np.int64(n), np.float64(hi), np.float64(lo), np.int32(b), np.int64(seg), out))  # Intentional float64: tail quantiles are compared with a double rank
    return np.asarray(out.get(), dtype=np.int64)


def _slice_counts(cols_rows: Any, c0: int, block: int, resident: bool, packed: bool, vflag: Any, v_d: Any, hi: float, lo: float) -> np.ndarray:
    """``(4, b)`` counts of the columns ``c0 .. c0 + block`` of ``cols_rows``: uploaded unless already on the device, then the packed or the chain version."""
    import cupy as cp

    n = int(cols_rows.shape[1])
    part = cols_rows[c0 : c0 + block] if resident else cp.asarray(np.ascontiguousarray(cols_rows[c0 : c0 + block]))
    return _block_counts_packed(cp, part, vflag, n, hi, lo) if packed else _block_counts_chain(cp, part, v_d, n, hi, lo)


@device_exclusive
def tail_dependence_columns_device(cols_rows: Any, v: np.ndarray, q: float) -> Optional[np.ndarray]:
    """Floored tail dependence of every row of ``cols_rows`` ``(k, n)`` against the rank-uniformised target ``v``, as the host kernel gives it; ``None`` when the device cannot take it.

    ``cols_rows`` is a host array, which is uploaded block by block, or a cupy array already on the device, which is read in place with no upload."""
    k, n = cols_rows.shape
    if cols_rows.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or k == 0 or n < 2:  # Intentional float64: dtype check: float64 is a supported column type here, nothing is computed in it
        return None
    if int(np.shape(v)[0]) != int(n):
        log_throttle(logger, "tail_dependence_gpu.target_length", logging.WARNING, "device tail dependence skipped: the target has %d rows, the columns %d; the host kernel scores the columns", int(np.shape(v)[0]), int(n))
        return None
    try:
        import cupy as cp

        resident = isinstance(cols_rows, cp.ndarray)
        hi = float(q)
        lo = 1.0 - q
        denom = 1.0 - q
        v_host = np.ascontiguousarray(v, dtype=np.float64)  # Intentional float64: the rank-uniformised target is compared in double, as the host kernel does
        n_v_hi = int(np.count_nonzero(v_host > hi))
        n_v_lo = int(np.count_nonzero(v_host < lo))
        v_d = cp.asarray(v_host)
        vflag = (cp.asarray(v_host > hi).astype(cp.uint64) << cp.uint64(1)) | cp.asarray(v_host < lo).astype(cp.uint64)
        packed = cols_rows.dtype == np.dtype(np.float32)
        per_element = _BYTES_PER_ELEMENT_PACKED if packed else _BYTES_PER_ELEMENT_CHAIN
        block = int(max(1, min(k, (_BUDGET_FRACTION * device_room(cp)) // (n * per_element))))
        counts = np.empty((4, k), dtype=np.int64)
        for c0 in range(0, k, block):
            got = _slice_counts(cols_rows, c0, block, resident, packed, vflag, v_d, hi, lo)
            counts[:, c0 : c0 + got.shape[1]] = got
        n_u_hi, both_hi, n_u_lo, both_lo = counts
        up = both_hi / n / denom - (n_u_hi / n) * (n_v_hi / n) / denom
        low = both_lo / n / denom - (n_u_lo / n) * (n_v_lo / n) / denom
        best = np.where(up > low, up, low)
        return np.asarray(np.where(best > 0.0, best, 0.0), dtype=np.float64)  # Intentional float64: statistic output kept f64
    except Exception as exc:
        log_throttle(logger, "tail_dependence_gpu.failed", logging.WARNING, "device tail dependence failed (%s: %s); the host kernel scores the columns", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A 60-column float32 block of ``dims["n_samples"]`` rows with a binary target already rank-uniformised."""
    from ._orthogonal_copula_mi_fe import _rank_to_uniform

    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    cols = np.ascontiguousarray(rng.standard_normal((60, n)).astype(np.float32))
    y = rng.integers(0, 2, n)
    return (cols, np.ascontiguousarray(_rank_to_uniform(y), dtype=np.float64))  # Intentional float64: the host kernel takes a float64 target


def _host(cols: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The host kernel's tail dependence per column."""
    from ._orthogonal_tail_dependence_fe import _tail_dep_columns_njit

    out = np.zeros(cols.shape[0], dtype=np.float64)  # Intentional float64: statistic output kept f64
    _tail_dep_columns_njit(cols, v, 0.95, out, cols.dtype == np.float32)
    return out


def _device(cols: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The device result; a failed launch raises so the sweep does not record a host fallback as a device time."""
    out = tail_dependence_columns_device(cols, v, 0.95)
    if out is None:
        raise RuntimeError("device tail dependence unavailable")
    return out


_run_sweep = host_device_sweep(__name__, _host, _device, repeats=3, rtol=1e-12)


_SWEEP_ROWS = [5_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
_SALT = 3


_fallback_choice = rows_fallback(__name__)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_tail_dep_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_tail_dep_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_tail_dep_device failed: %s", e)
    _SPEC = None


from ._ktc_fallback import ktc_device_chooser as _ktc_device_chooser

use_device = _ktc_device_chooser(__name__, "fe_tail_dep_device")


__all__ = ["tail_dependence_columns_device", "use_device"]
