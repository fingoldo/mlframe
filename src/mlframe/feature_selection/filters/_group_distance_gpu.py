"""Device twin of the per-group statistics of ``_group_distance_fe``: for one group column and a block of numeric columns, the finite count, the sum, the global-bin histogram and the Wasserstein-1 distance to the global sample of every (column, group) cell.

The host builds the three distance features (z of the group mean, KL of the group histogram, Wasserstein-1) one (group column, numeric column) pair at a time with a Python loop over the groups
(162 pairs and 2.6 thread-seconds at 100000 rows x 100 columns). Here a block of columns of one group column is done in three launches:

1. ``gd_stats``: one pass over the rows; a block keeps the ``G x (1 + bins)`` count table and the ``G`` sums of its slice of rows in shared memory and adds them to the column's global tables
   (integer counts are exact; the double sums use atomic adds, so their last bit depends on the schedule, 1e-16 relative);
2. ``gd_sort_keys`` and a batched stable argsort: one 64-bit key per element, the group id in the high word (non-finite rows get the id ``G`` and sort last) and the order-preserving 32-bit code of the
   value in the low word, so each group's finite values come out contiguous and ascending;
3. ``gd_w1``: one thread per (column, group) runs the host's two-pointer merge of the group's sorted values against the column's sorted global sample (the arithmetic of
   ``_wasserstein1_sorted_kernel``, in double precision).

The caller forms z and KL from the tables exactly as the host does. A group count above ``MAX_GROUPS`` (the shared tables would not fit) or a non-float32 block hands the work back to the host.
Columns go in blocks that fit ``_BUDGET_FRACTION`` of the free device memory; any device fault returns ``None``. The host/device choice is the kernel-tuning entry ``fe_group_distance_stats_device``
(fallback: the host).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._device_sort_safe import device_argsort, device_room
from ._ktc_fallback import device_from_rows
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 10**9  # the tuner decides; until a sweep says the device wins the host loop runs
MAX_GROUPS = 256
_SHARED_BYTES = 40 * 1024
_W1_STEPS = 512  # must equal W1_STEPS in the kernel source

_SRC = r"""
// dtype-ok: Intentional float64: {ACC} holds the sums and the Wasserstein integral exactly as the host kernel does; {REAL} is the column type.
typedef unsigned long long u64;
typedef unsigned int u32;

extern "C" __global__ void gd_stats(const {REAL}* __restrict__ xs, const short* __restrict__ bins, const int* __restrict__ gid, int* __restrict__ cnt, {ACC}* __restrict__ sums,
                                    int* __restrict__ hist, const long long n, const int ngroups, const int nbins) {
    extern __shared__ int shared[];
    int* s_cnt = shared;                        // ngroups
    int* s_hist = shared + ngroups;             // ngroups * nbins
    {ACC}* s_sum = ({ACC}*)(shared + ngroups + ngroups * nbins + ((ngroups + ngroups * nbins) & 1));
    const int cells = ngroups * nbins;
    for (int c = threadIdx.x; c < cells; c += blockDim.x) s_hist[c] = 0;
    for (int c = threadIdx.x; c < ngroups; c += blockDim.x) { s_cnt[c] = 0; s_sum[c] = 0.0; }
    __syncthreads();
    const long long base = (long long)blockIdx.y * n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const {REAL} v = xs[base + i];
        if (!isfinite(v)) continue;
        const int g = gid[i];
        atomicAdd(&s_cnt[g], 1);
        atomicAdd(&s_sum[g], ({ACC})v);
        const int b = (int)bins[base + i];
        if (b >= 0) atomicAdd(&s_hist[g * nbins + b], 1);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < ngroups; c += blockDim.x) {
        if (s_cnt[c]) { atomicAdd(&cnt[(long long)blockIdx.y * ngroups + c], s_cnt[c]); atomicAdd(&sums[(long long)blockIdx.y * ngroups + c], s_sum[c]); }
    }
    for (int c = threadIdx.x; c < cells; c += blockDim.x) if (s_hist[c]) atomicAdd(&hist[(long long)blockIdx.y * cells + c], s_hist[c]);
}

extern "C" __global__ void gd_sort_keys(const {REAL}* __restrict__ xs, const int* __restrict__ gid, u64* __restrict__ keys, const long long n, const int ngroups) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = (long long)blockIdx.y * n + i;
    const {REAL} v = xs[at];
    u32 u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;
    const u32 ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    const u32 g = isfinite(v) ? (u32)gid[i] : (u32)ngroups;
    keys[at] = ((u64)g << 32) | (u64)ord;
}

#define W1_STEPS 512  // merged elements one thread walks
extern "C" __global__ void gd_w1(const {REAL}* __restrict__ xs_sorted, const {ACC}* __restrict__ gsorted, const long long* __restrict__ gn, const long long* __restrict__ offs,
                                 const int* __restrict__ cnt, {ACC}* __restrict__ out, const long long n, const long long gmax, const int ngroups, const int chunks) {
    const int col = blockIdx.y;
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int g = (int)(t / chunks);
    const long long c = t % chunks;
    if (g >= ngroups) return;
    const long long nu = cnt[col * ngroups + g];
    const long long nv = gn[col];
    if (nu == 0 || nv == 0) return;
    const long long p0 = c * W1_STEPS;
    if (p0 >= nu + nv) return;
    const {REAL}* u = xs_sorted + (long long)col * n + offs[col * ngroups + g];
    const {ACC}* v = gsorted + (long long)col * gmax;
    // merge-path co-rank of the diagonal p0: the number of u elements among the first p0 merged ones (u before v on ties, as the host merge takes u when u <= v)
    long long lo = p0 > nv ? p0 - nv : 0, hi = p0 < nu ? p0 : nu;
    while (lo < hi) {
        const long long mid = (lo + hi) >> 1;
        if (({ACC})u[mid] <= v[p0 - 1 - mid]) lo = mid + 1; else hi = mid;
    }
    long long i = lo, j = p0 - lo;
    {ACC} prev;
    if (p0 == 0) { const {ACC} u0 = ({ACC})u[0], v0 = v[0]; prev = u0 <= v0 ? u0 : v0; }
    else if (i > 0 && (j == 0 || ({ACC})u[i - 1] >= v[j - 1])) prev = ({ACC})u[i - 1];
    else prev = v[j - 1];
    const {ACC} nu_f = ({ACC})nu, nv_f = ({ACC})nv;
    {ACC} total = 0.0;
    long long end = p0 + W1_STEPS;
    if (end > nu + nv) end = nu + nv;
    for (long long p = p0; p < end; ++p) {
        const bool take_u = i < nu && (j >= nv || ({ACC})u[i] <= v[j]);
        const {ACC} cur = take_u ? ({ACC})u[i] : v[j];
        if (cur != prev) {
            {ACC} d = ({ACC})i / nu_f - ({ACC})j / nv_f;
            if (d < 0.0) d = -d;
            total += d * (cur - prev);
            prev = cur;
        }
        if (take_u) ++i; else ++j;
    }
    atomicAdd(&out[col * ngroups + g], total);
}
"""

_MODULE = RawModuleCache(_SRC.replace("{REAL}", "float").replace("{ACC}", "double"))


def group_stats_device(
    gid: np.ndarray, n_groups: int, xs: np.ndarray, bins: np.ndarray, global_sorted: Sequence[np.ndarray], n_bins: int
) -> "Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]":
    """``(count, sum, hist, w1)`` of every (column, group) cell, or ``None`` when the device cannot take the block (the caller runs the host loop).

    ``gid``: group id per row (``0 .. n_groups - 1``); ``xs``: float32 ``(columns, n)``; ``bins``: the global-histogram bin per element (``-1`` outside); ``global_sorted``: each column's finite values ascending
    (double precision). ``count`` is ``(columns, groups)`` int64, ``sum`` float64, ``hist`` ``(columns, groups, bins)`` int64, ``w1`` float64."""
    k, n = xs.shape
    if k == 0 or n < 2 or xs.dtype != np.float32 or n_groups > MAX_GROUPS or n_groups * (n_bins + 3) * 4 > _SHARED_BYTES:
        return None
    try:
        import cupy as cp

        module = _MODULE.get(cp)
        stats_k, keys_k, w1_k = module.get_function("gd_stats"), module.get_function("gd_sort_keys"), module.get_function("gd_w1")
        gid_d = cp.asarray(np.ascontiguousarray(gid, dtype=np.int32))
        free = device_room(cp)
        gmax = max(1, max(int(g.size) for g in global_sorted))
        per_col = n * (4 + 2 + 8 + 8 + 4 + 16) + gmax * 8
        block = int(max(1, min(k, (_BUDGET_FRACTION * free) // per_col)))
        cnt = np.zeros((k, n_groups), dtype=np.int64)
        sums = np.zeros((k, n_groups), dtype=np.float64)  # Intentional float64: group sums
        hist = np.zeros((k, n_groups, n_bins), dtype=np.int64)
        w1 = np.zeros((k, n_groups), dtype=np.float64)  # Intentional float64: Wasserstein integral
        shared = (n_groups * (1 + n_bins) + ((n_groups * (1 + n_bins)) & 1)) * 4 + n_groups * 8
        # the block width is a device/kernel occupancy choice, not a literal: each of the three kernels gets its own (different registers, stats_k's shared table changes with n_groups/n_bins)
        stats_threads = block_size_for(stats_k, dyn_smem=shared, work_items=n, pow2=True)
        keys_threads = block_size_for(keys_k, work_items=n, pow2=True)
        w1_threads = block_size_for(w1_k, work_items=n, pow2=True)
        grid_x = (n + stats_threads - 1) // stats_threads
        for c0 in range(0, k, block):
            w = min(block, k - c0)
            x_d = cp.asarray(np.ascontiguousarray(xs[c0 : c0 + w]))
            b_d = cp.asarray(np.ascontiguousarray(bins[c0 : c0 + w], dtype=np.int16))
            cnt_d = cp.zeros((w, n_groups), dtype=cp.int32)
            sum_d = cp.zeros((w, n_groups), dtype=cp.float64)
            hist_d = cp.zeros((w, n_groups, n_bins), dtype=cp.int32)
            stats_k((min(grid_x, 64), w), (stats_threads,), (x_d, b_d, gid_d, cnt_d, sum_d, hist_d, np.int64(n), np.int32(n_groups), np.int32(n_bins)), shared_mem=shared)
            del b_d
            keys = cp.empty((w, n), dtype=cp.uint64)
            keys_k(((n + keys_threads - 1) // keys_threads, w), (keys_threads,), (x_d, gid_d, keys, np.int64(n), np.int32(n_groups)))
            order = device_argsort(keys, axis=1)
            del keys
            xs_sorted = cp.take_along_axis(x_d, order, axis=1)
            del x_d
            g_d = cp.zeros((w, gmax), dtype=cp.float64)
            gn = np.zeros(w, dtype=np.int64)
            for j in range(w):
                arr = np.ascontiguousarray(global_sorted[c0 + j], dtype=np.float64)  # Intentional float64: sorted operand of the merge
                g_d[j, : arr.size] = cp.asarray(arr)
                gn[j] = arr.size
            out_d = cp.zeros((w, n_groups), dtype=cp.float64)
            cnt64 = cnt_d.astype(cp.int64)
            offs = cp.cumsum(cnt64, axis=1) - cnt64
            chunks = int((int(cnt_d.max()) + gmax + _W1_STEPS - 1) // _W1_STEPS)
            threads_total = n_groups * chunks
            w1_k(((threads_total + w1_threads - 1) // w1_threads, w), (w1_threads,), (xs_sorted, g_d, cp.asarray(gn), offs, cnt_d, out_d, np.int64(n), np.int64(gmax), np.int32(n_groups), np.int32(chunks)))
            cnt[c0 : c0 + w] = cp.asnumpy(cnt_d)
            sums[c0 : c0 + w] = cp.asnumpy(sum_d)
            hist[c0 : c0 + w] = cp.asnumpy(hist_d)
            w1[c0 : c0 + w] = cp.asnumpy(out_d)
            del order, xs_sorted, g_d, out_d
        return cnt, sums, hist, w1
    except Exception as exc:
        log_throttle(logger, "group_distance_gpu.failed", logging.WARNING, "device group statistics failed (%s: %s); the host loop computes them", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A 40-column float32 block of ``dims["n_samples"]`` rows, 12 groups, the global bins on 20 edges and the sorted global samples."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    xs = rng.standard_normal((40, n)).astype(np.float32)
    gid = rng.integers(0, 12, n).astype(np.int32)
    xs[:5] += gid[None, :] * 0.1
    bins = np.empty(xs.shape, dtype=np.int16)
    for j in range(xs.shape[0]):
        edges = np.linspace(float(xs[j].min()), float(xs[j].max()), 21)
        b = np.searchsorted(edges, xs[j], side="right") - 1
        b[xs[j] == edges[-1]] = 19
        bins[j] = b
    return (gid, 12, xs, bins, [np.sort(r.astype(np.float64)) for r in xs], 20)  # Intentional float64: sorted global samples


def _host(gid: np.ndarray, n_groups: int, xs: np.ndarray, bins: np.ndarray, global_sorted: Sequence[np.ndarray], n_bins: int) -> np.ndarray:
    """The host loop's Wasserstein-1 per (column, group), the statistic the tuner compares."""
    from ._group_distance_fe import _wasserstein1_sorted_kernel

    out = np.zeros((xs.shape[0], n_groups), dtype=np.float64)  # Intentional float64: Wasserstein integral
    for j in range(xs.shape[0]):
        for g in range(n_groups):
            vals = np.sort(xs[j][gid == g].astype(np.float64))  # Intentional float64: sorted operand of the merge
            out[j, g] = _wasserstein1_sorted_kernel(vals, global_sorted[j])
    return out


def _device(gid: np.ndarray, n_groups: int, xs: np.ndarray, bins: np.ndarray, global_sorted: Sequence[np.ndarray], n_bins: int) -> np.ndarray:
    """The device kernel's Wasserstein-1 per (column, group); a failed launch raises so the sweep does not record a host fallback as a device time."""
    got = group_stats_device(gid, n_groups, xs, bins, global_sorted, n_bins)
    if got is None:
        raise RuntimeError("device group statistics unavailable")
    return got[3]


def _run_sweep() -> list:
    """Time the host loop and the device kernels per row count; the device must agree with the host."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return sweep_backend_grid({"host": _host, "device": _device}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="host", repeats=2, equiv_rtol=1e-9, equiv_atol=1e-9)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the host unless ``DEVICE_FALLBACK_ROWS`` is lowered."""
    return device_from_rows(n_samples, DEVICE_FALLBACK_ROWS)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_group_distance_stats_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_group_distance_stats_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_group_distance_stats_device failed: %s", e)
    _SPEC = None


def use_device(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device kernels: the per-host kernel-tuning cache decides (fallback ``DEVICE_FALLBACK_ROWS``, the host until a sweep says otherwise)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_group_distance_stats_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__ = ["group_stats_device", "use_device", "MAX_GROUPS"]
