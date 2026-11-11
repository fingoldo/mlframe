"""Device twin of ``masked_abs_spearman_presorted``: the |Spearman rho| of one engineered candidate against the kept columns it shares NaN rows with.

One block per (candidate, kept column) pair. The rank of a row over the jointly finite rows is read off the cached sorted order of its column: a block-wide prefix sum of the
"row is jointly finite" flags along the sorted order gives, for every tie run ``[lo, hi]``, the number of kept rows before the run (``cs[lo - 1]``) and inside it, and the average
rank is ``before + (inside + 1) / 2``. The mean of the ranks is ``(m + 1) / 2`` exactly, so the three rank sums (``saa``, ``sbb``, ``sab``) are accumulated while the ranks are produced,
and no rank vector is written back except the candidate's, kept in row space for the kept column's pass to read.

The skip rules are the host ones (fewer than 8 jointly finite rows, a column constant to ``rel_tol`` of its largest magnitude, constant ranks give ``-1``). The sums are double precision;
only their order differs from the host loop, so ``rho`` agrees to the last digits and the ``>= 0.99`` verdict can only move for a pair within ~1e-15 of the threshold.

The kept columns, their sorted orders and tie runs live on the device across the candidates of a fit (``_COLUMN_CACHE``, least recently used first out within a share of the free memory),
because re-uploading ~100 columns per candidate would cost more than the host kernel. When the pairs do not fit the memory the call returns ``False`` and the caller runs the host kernel.

``MLFRAME_DEDUP_SPEARMAN_DEVICE=0|1`` forces the path; otherwise the kernel-tuning cache decides by row count (fallback ``DEVICE_FALLBACK_ROWS``).
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from typing import Any, Optional

import numpy as np

from .._ktc_fallback import device_from_rows
from .._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

THREADS = 1024  # 84 pairs give one block per pair and ~1 resident block per SM: wide blocks hide the dependent-gather latency (84 pairs x 200000 rows per launch: 128 threads 82 ms, 256 86 ms, 512 63 ms, 1024 53 ms)
_CACHE_FRACTION = 0.3  # share of the free device memory the cached kept columns may hold
_SCRATCH_FRACTION = 0.15  # share the per-pair prefix-sum and rank scratch of one launch may hold
DEVICE_FALLBACK_ROWS = 100_000  # until the kernel-tuning cache has a measured crossover

_SRC = r"""
#define T %THREADS%
// dtype-ok: Intentional float64: the rank sums are double on the host kernel as well; columns are widened exactly.
__device__ double block_sum(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) { double t = 0.0; for (int w = 0; w < T / 32; ++w) t += sh[w]; sh[32] = t; }
    __syncthreads();
    const double r = sh[32];
    __syncthreads();
    return r;
}
__device__ double block_max(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffff, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) { double t = 0.0; for (int w = 0; w < T / 32; ++w) t = fmax(t, sh[w]); sh[32] = t; }
    __syncthreads();
    const double r = sh[32];
    __syncthreads();
    return r;
}
// inclusive count of jointly finite rows along the sorted order `ord` (nf entries) into cs; four consecutive positions per thread, warp-shuffle scan, two barriers per 4*T positions
__device__ __forceinline__ int joint_bit(const unsigned int* jm, const int r) { return (int)((jm[r >> 5] >> (r & 31)) & 1u); }
__device__ void scan_side(const int* ord, const int nf, const unsigned int* jm, int* cs, int* wsum) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int carry = 0;
    for (int base = 0; base < nf; base += 4 * T) {
        const int t0 = base + threadIdx.x * 4;
        int f[4];
        int loc = 0;
        for (int j = 0; j < 4; ++j) {
            const int t = t0 + j;
            const int fj = t < nf ? joint_bit(jm, ord[t]) : 0;
            f[j] = fj;
            loc += fj;
        }
        int v = loc;
        for (int o = 1; o < 32; o <<= 1) { const int x = __shfl_up_sync(0xffffffff, v, o); if (lane >= o) v += x; }
        if (lane == 31) wsum[w] = v;
        __syncthreads();
        int woff = 0, tot = 0;
        for (int i = 0; i < T / 32; ++i) { if (i < w) woff += wsum[i]; tot += wsum[i]; }
        int run = carry + woff + v - loc;
        for (int j = 0; j < 4; ++j) { run += f[j]; const int t = t0 + j; if (t < nf) cs[t] = run; }
        carry += tot;
        __syncthreads();
    }
    __syncthreads();
}
// dtype-ok: Intentional float64: the Spearman rho accumulates n-element rank sums and products; a float32 accumulator loses the low bits the |rho| >= threshold decision reads on ties.
extern "C" __global__ void __launch_bounds__(%THREADS%) masked_spearman(
    const double* __restrict__ a, const unsigned int* __restrict__ fbits_a, const int* __restrict__ order_a, const int* __restrict__ lo_a, const int* __restrict__ hi_a, const int na,
    const unsigned long long* __restrict__ cols, const unsigned long long* __restrict__ fbits, const unsigned long long* __restrict__ ords, const unsigned long long* __restrict__ los,
    const unsigned long long* __restrict__ his, const int* __restrict__ nfs, const long long n, const int n_words, const double rel_tol,
    int* __restrict__ cs_all, float* __restrict__ ra_all, unsigned int* __restrict__ jm_all, double* __restrict__ out) {
    __shared__ double sh[33];
    __shared__ int scan[T / 32];
    const int k = blockIdx.x;
    const double* b = (const double*)cols[k];
    const unsigned int* fb = (const unsigned int*)fbits[k];
    const int* ob = (const int*)ords[k];
    const int* lob = (const int*)los[k];
    const int* hib = (const int*)his[k];
    const int nb = nfs[k];
    int* cs = cs_all + (long long)k * n;
    float* ra = ra_all + (long long)k * n;
    unsigned int* jm = jm_all + (long long)k * n_words;

    double cnt = 0.0;
    for (int w = threadIdx.x; w < n_words; w += T) { const unsigned int x = fbits_a[w] & fb[w]; jm[w] = x; cnt += (double)__popc(x); }
    cnt = block_sum(cnt, sh);
    const int m = (int)cnt;
    if (m < 8) { if (threadIdx.x == 0) out[k] = -1.0; return; }
    double sa = 0.0, sb = 0.0;
    for (long long i = threadIdx.x; i < n; i += T) {
        if (joint_bit(jm, (int)i)) { sa += a[i]; sb += b[i]; }
    }
    sa = block_sum(sa, sh);
    sb = block_sum(sb, sh);
    const double ma = sa / m, mb = sb / m;
    double va = 0.0, vb = 0.0, xa = 0.0, xb = 0.0;
    for (long long i = threadIdx.x; i < n; i += T) {
        if (joint_bit(jm, (int)i)) {
            const double da = a[i] - ma, db = b[i] - mb;
            va += da * da; vb += db * db;
            xa = fmax(xa, fabs(a[i])); xb = fmax(xb, fabs(b[i]));
        }
    }
    va = block_sum(va, sh);
    vb = block_sum(vb, sh);
    xa = block_max(xa, sh);
    xb = block_max(xb, sh);
    if (sqrt(va / m) <= rel_tol * xa || sqrt(vb / m) <= rel_tol * xb) { if (threadIdx.x == 0) out[k] = -1.0; return; }

    const double rmean = 0.5 * (m + 1.0);
    scan_side(order_a, na, jm, cs, scan);
    double saa = 0.0;
    for (int t = threadIdx.x; t < na; t += T) {
        const int r = order_a[t];
        if (joint_bit(jm, r)) {
            const int s = lo_a[t], e = hi_a[t];
            const int before = s > 0 ? cs[s - 1] : 0;
            const float rank = (float)before + 0.5f * (float)(cs[e] - before + 1);
            ra[r] = rank;
            saa += ((double)rank - rmean) * ((double)rank - rmean);
        }
    }
    __syncthreads();
    scan_side(ob, nb, jm, cs, scan);
    double sbb = 0.0, sab = 0.0;
    for (int t = threadIdx.x; t < nb; t += T) {
        const int r = ob[t];
        if (joint_bit(jm, r)) {
            const int s = lob[t], e = hib[t];
            const int before = s > 0 ? cs[s - 1] : 0;
            const double rank = (double)before + 0.5 * (double)(cs[e] - before + 1);
            sbb += (rank - rmean) * (rank - rmean);
            sab += ((double)ra[r] - rmean) * (rank - rmean);
        }
    }
    saa = block_sum(saa, sh);
    sbb = block_sum(sbb, sh);
    sab = block_sum(sab, sh);
    if (threadIdx.x == 0) {
        out[k] = (sqrt(saa / m) <= 1e-12 || sqrt(sbb / m) <= 1e-12) ? -1.0 : fabs(sab / sqrt(saa * sbb));
    }
}
"""

_MODULE = RawModuleCache(_SRC.replace("%THREADS%", str(THREADS)))


def _finite_bits(col: np.ndarray) -> np.ndarray:
    """The finite-row mask of ``col`` as little-endian bits in uint32 words (row ``r`` is bit ``r % 32`` of word ``r // 32``)."""
    packed = np.packbits(np.isfinite(col), bitorder="little")
    packed = np.concatenate((packed, np.zeros((-packed.size) % 4, dtype=np.uint8)))
    return packed.view(np.uint32)


class _DeviceColumn:
    """A kept column on the device: its values at float64 and its sorted finite-row order with the tie run ``[lo, hi]`` of every sorted position."""

    __slots__ = ("bits", "hi", "lo", "nbytes", "nf", "order", "ref", "values")

    def __init__(self, cp: Any, col: np.ndarray, order: np.ndarray, ref: Any) -> None:
        sv = col[order]
        nf = int(order.shape[0])
        idx = np.arange(nf, dtype=np.int64)
        if nf > 1:
            neq = sv[1:] != sv[:-1]
        else:
            neq = np.zeros(0, dtype=np.bool_)
        starts = np.maximum.accumulate(np.where(np.concatenate(([True], neq)), idx, 0)) if nf else idx
        ends = np.minimum.accumulate(np.where(np.concatenate((neq, [True])), idx, nf)[::-1])[::-1] if nf else idx
        self.values = cp.asarray(np.ascontiguousarray(col, dtype=np.float64))  # Intentional float64: exact widening of the column
        self.bits = cp.asarray(_finite_bits(col))
        self.order = cp.asarray(np.ascontiguousarray(order, dtype=np.int32))
        self.lo = cp.asarray(starts.astype(np.int32))
        self.hi = cp.asarray(ends.astype(np.int32))
        self.nf = nf
        self.nbytes = int(self.values.nbytes + self.bits.nbytes + 3 * self.order.nbytes)
        self.ref = ref


_CACHE_LOCK = threading.Lock()
_COLUMN_CACHE: "dict[int, _DeviceColumn]" = {}  # id(column) -> device copy, least recently used first; a dead weak reference never matches a reused id


def clear_device_column_cache() -> None:
    """Drop the device copies of the kept columns (a fit ended)."""
    with _CACHE_LOCK:
        _COLUMN_CACHE.clear()


def _free_device_bytes(cp: Any) -> int:
    """Free device memory plus what the pool holds unused."""
    pool = cp.get_default_memory_pool()
    return int(cp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())


def _device_columns(cp: Any, cols: "list[np.ndarray]", orders: "list[np.ndarray]") -> "Optional[list[_DeviceColumn]]":
    """Device copies of ``cols`` (uploaded when not cached), or ``None`` when all of them do not fit within the cache share of the free memory."""
    held = sum(c.nbytes for c in _COLUMN_CACHE.values())
    budget = int(_CACHE_FRACTION * (_free_device_bytes(cp) + held))
    need = sum(int(c.shape[0]) * 8 + 12 * int(o.shape[0]) for c, o in zip(cols, orders))
    if need > budget:
        return None
    out: list = []
    with _CACHE_LOCK:
        for col, order in zip(cols, orders):
            hit = _COLUMN_CACHE.pop(id(col), None)
            if hit is None or hit.ref() is not col:
                for dead in [k for k, v in _COLUMN_CACHE.items() if v.ref() is None]:
                    del _COLUMN_CACHE[dead]
                hit = _DeviceColumn(cp, col, order, weakref.ref(col))
            _COLUMN_CACHE[id(col)] = hit  # most recently used last
            out.append(hit)
        in_use = {id(col) for col in cols}
        total = sum(c.nbytes for c in _COLUMN_CACHE.values())
        for key in list(_COLUMN_CACHE):
            if total <= budget:
                break
            if key not in in_use:
                total -= _COLUMN_CACHE.pop(key).nbytes
    return out


def _pointer_table(cp: Any, part: "list[_DeviceColumn]") -> Any:
    """``ptr(attr)``: a device array of the addresses of attribute ``attr`` of the columns in ``part``, the kernel's per-pair column table."""

    def ptr(attr: str) -> Any:
        """The address table of one attribute."""
        return cp.asarray(np.array([int(getattr(c, attr).data.ptr) for c in part], dtype=np.uint64))

    return ptr


def masked_abs_spearman_device(a: np.ndarray, order_a: np.ndarray, others: "list[np.ndarray]", orders: "list[np.ndarray]", rel_tol: float, out: np.ndarray) -> bool:
    """Fill ``out[k]`` with |rho| of ``a`` (its finite rows are the candidate mask) and ``others[k]`` on the device (``-1`` for a skipped pair); ``False`` (``out`` untouched) when the device cannot take the pairs."""
    n = int(a.shape[0])
    if not others or n == 0 or n >= (1 << 23):  # float32 ranks are exact below 2**23 rows
        return False
    try:
        import cupy as cp

        cols = _device_columns(cp, others, orders)
        if cols is None:
            return False
        cand = _DeviceColumn(cp, a, order_a, weakref.ref(a))
        kernel = _MODULE.get(cp).get_function("masked_spearman")
        n_words = int(cand.bits.shape[0])
        per_pair = 8 * n + 4 * n_words
        chunk = int(max(1, min(len(cols), (_SCRATCH_FRACTION * _free_device_bytes(cp)) // per_pair)))
        cs = cp.empty((chunk, n), dtype=cp.int32)
        ra = cp.empty((chunk, n), dtype=cp.float32)  # ranks are half-integers below 2**23, exact at float32; the sums read them back as double
        jm = cp.empty((chunk, n_words), dtype=cp.uint32)
        res = cp.empty(chunk, dtype=cp.float64)  # Intentional float64: |rho| statistic vector
        for k0 in range(0, len(cols), chunk):
            part = cols[k0 : k0 + chunk]
            ptr = _pointer_table(cp, part)
            nfs = cp.asarray(np.array([c.nf for c in part], dtype=np.int32))
            kernel(  # launch-ok: THREADS=1024 is measured (see the constant's own comment: 128/256/512/1024 benchmarked, 1024 fastest) and baked into the C template at compile time (%THREADS%), not a runtime choice
                (len(part),),
                (THREADS,),
                (cand.values, cand.bits, cand.order, cand.lo, cand.hi, np.int32(cand.nf), ptr("values"), ptr("bits"), ptr("order"), ptr("lo"), ptr("hi"), nfs, np.int64(n), np.int32(n_words), np.float64(rel_tol), cs, ra, jm, res),
            )
            out[k0 : k0 + len(part)] = cp.asnumpy(res[: len(part)])
        return True
    except Exception as exc:
        log_throttle(logger, "eng_dedup_spearman_gpu.failed", logging.WARNING, "device masked Spearman failed (%s: %s); the host kernel scores the pairs", type(exc).__name__, exc)
        return False


def _sweep_inputs(dims: dict) -> tuple:
    """A candidate with ~5 % NaN rows and 40 kept columns (half of them NaN-bearing, with ties) at ``dims["n_samples"]`` rows."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    a = rng.normal(size=n).astype(np.float32)
    a[rng.random(n) < 0.05] = np.nan
    cols = []
    for j in range(40):
        c = (a * (1.0 if j % 3 == 0 else 0.0) + rng.normal(size=n)).round(2 if j % 2 else 4).astype(np.float32)
        if j % 2:
            c[rng.random(n) < 0.03] = np.nan
        cols.append(c)
    return (a, cols)


def _host_scores(a: np.ndarray, cols: "list[np.ndarray]") -> np.ndarray:
    """The host kernel's |rho| of ``a`` against ``cols``."""
    from numba.typed import List as _TypedList

    from ._eng_dedup_batch_corr import finite_sort_order, masked_abs_spearman_presorted

    others, orders = _TypedList(), _TypedList()
    for c in cols:
        others.append(c)
        orders.append(finite_sort_order(c, cache=False))
    out = np.empty(len(cols), dtype=np.float64)  # Intentional float64: |rho| statistic vector
    masked_abs_spearman_presorted(a, np.isfinite(a), finite_sort_order(a, cache=False), others, orders, 1e-12, out)
    return out


def _device_scores(a: np.ndarray, cols: "list[np.ndarray]") -> np.ndarray:
    """The device kernel's |rho|; a failed launch raises so the sweep does not record a host fallback as a device time."""
    from ._eng_dedup_batch_corr import finite_sort_order

    out = np.empty(len(cols), dtype=np.float64)  # Intentional float64: |rho| statistic vector
    if not masked_abs_spearman_device(a, finite_sort_order(a, cache=False), cols, [finite_sort_order(c, cache=False) for c in cols], 1e-12, out):
        raise RuntimeError("device masked Spearman unavailable")
    return out


def _run_sweep() -> list:
    """Time the host kernel and the device kernel per row count; the device must agree with the host."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return sweep_backend_grid(
        {"host": _host_scores, "device": _device_scores},
        {"n_samples": _SWEEP_ROWS},
        _sweep_inputs,
        reference="host",
        repeats=3,
        equiv_rtol=1e-9,
        equiv_atol=1e-9,
    )


_SWEEP_ROWS = [30_000, 100_000, 400_000]
_SALT = 1


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the device from ``DEVICE_FALLBACK_ROWS`` rows."""
    return device_from_rows(n_samples, DEVICE_FALLBACK_ROWS)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_dedup_spearman_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_dedup_spearman_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_dedup_spearman_device failed: %s", e)
    _SPEC = None


def use_device(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device kernel: ``MLFRAME_DEDUP_SPEARMAN_DEVICE=0|1`` forces a path, else the per-host tuner (fallback ``DEVICE_FALLBACK_ROWS``)."""
    from .._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    forced = os.environ.get("MLFRAME_DEDUP_SPEARMAN_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_dedup_spearman_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__ = ["masked_abs_spearman_device", "use_device", "clear_device_column_cache"]
