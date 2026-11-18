"""Device twin of the sorted walk of ``_bootstrap_rank_mi_njit`` for columns whose values are all distinct.

Win axis: the host kernel sorts each column once and walks the sorted rows for all resamples (about 3 to 4 s for 105 columns, 10 resamples of 80000 rows). Here ``cupy.argsort`` orders all columns in one call and one
thread per (column, resample) walks the sorted rows, emitting each row ``counts[row, resample]`` times into the equi-frequency slices of that resample and ending with the plug-in MI, the host walk step for step.
It handles only fully distinct finite columns: a block of equal values of DIFFERENT rows is emitted in draw order by the host (``rowptr`` / ``poslist``), and that tie rule stays there. For a distinct column the
repeats of one row are adjacent and of one class, so the draw order inside them cannot matter and the MI equals the host's (summed in double in the same order).

The kernel reads only integers (row ids, multiplicities, class codes, slice ends), so there is no ``{REAL}``; the MI sums are double.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import cast

import numpy as np

from ._ktc_fallback import device_from_rows


from ._raw_module_cache import RawModuleCache
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # noqa: F401  (thrust aborts the process on a short pool)
from ._fe_gpu_vram import pool_column_block
from ._gpu_resident_select_kernels import _transpose_to_cm
from ._kernel_types import render

logger = logging.getLogger(__name__)

MAX_CLASSES = 16
#: Columns with up to this share of rows equal to the next sorted row are scored on the device; above it nearly every resample has a bin end inside a run and the host walk does the work anyway.
MAX_TIE_FRACTION = 0.01
MAX_BINS = 32
_COLUMN_BLOCK = 256

_SRC = render(r"""
#define KMAX 16
#define BMAX 32
extern "C" __global__ void bootstrap_rank_walk(const int* __restrict__ ord, const unsigned char* __restrict__ tie, const int* __restrict__ counts, const int* __restrict__ ycode,
                                               const int* __restrict__ cum, const int* __restrict__ lens, const int n, const int kc, const int B, const int nbins, const int ncls,
                                               {ACC}* __restrict__ out, unsigned char* __restrict__ bad) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= (long long)kc * B) return;
    const int j = (int)(t / B);
    const int b = (int)(t - (long long)j * B);
    const int* oj = ord + (long long)j * n;
    long long hist[BMAX * KMAX];
    for (int u = 0; u < nbins * ncls; ++u) hist[u] = 0;
    const unsigned char* tj = tie + (long long)j * n;
    int pos = 0, cur = 0;
    for (int i = 0; i < n; ++i) {
        // rows i .. last hold one value (``tie`` marks a row equal to the next sorted row); all their draws, repeats of a row included, take the bin of the run's
        // first sorted position: the tie rule of the rank binners, so no run is split across bins and the draw order inside it does not matter
        int last = i;
        while (tj[last]) ++last;
        int total = 0;
        for (int r = i; r <= last; ++r) total += counts[(long long)oj[r] * B + b];
        if (total > 0) {
            while (pos >= cum[b * nbins + cur] && cur < nbins - 1) ++cur;
            for (int r = i; r <= last; ++r) {
                const int row = oj[r];
                hist[cur * ncls + ycode[row]] += counts[(long long)row * B + b];
            }
            pos += total;
        }
        i = last;
    }
    bad[(long long)b * kc + j] = 0;
    long long hx[BMAX], hy[KMAX];
    for (int u = 0; u < nbins; ++u) hx[u] = 0;
    for (int c = 0; c < ncls; ++c) hy[c] = 0;
    for (int u = 0; u < nbins; ++u) for (int c = 0; c < ncls; ++c) { hx[u] += hist[u * ncls + c]; hy[c] += hist[u * ncls + c]; }
    const int m = pos;
    const {ACC} log_n = log(({ACC})m);
    {ACC} mi = 0.0;
    for (int u = 0; u < nbins; ++u) {
        if (hx[u] == 0) continue;
        const {ACC} log_hx = log(({ACC})hx[u]);
        for (int c = 0; c < ncls; ++c) {
            const long long nxy = hist[u * ncls + c];
            if (nxy == 0 || hy[c] == 0) continue;
            mi += (({ACC})nxy / ({ACC})m) * (log(({ACC})nxy) + log_n - log_hx - log(({ACC})hy[c]));
        }
    }
    out[(long long)b * kc + j] = mi < 0.0 ? 0.0 : mi;
}

extern "C" __global__ void bootstrap_rank_walk_blocks(const int* __restrict__ ord, const unsigned char* __restrict__ tie, const int* __restrict__ counts_bt, const int* __restrict__ ycode,
                                                      const int* __restrict__ cum, const int n, const int kc, const int B, const int nbins, const int ncls, {ACC}* __restrict__ out) {
    // One block per (column, resample); the block walks the sorted rows in chunks of blockDim.x. The draws before a row are the exclusive prefix sum of the multiplicities along the sorted order, a run of
    // equal values takes the bin of the draws before its FIRST row (a running maximum of those prefixes over the run starts, which only grow), the (bin, class) table is a shared histogram.
    const int j = blockIdx.x / B, b = blockIdx.x - (blockIdx.x / B) * B;
    __shared__ int hist[BMAX * KMAX];
    __shared__ int cums[BMAX];
    __shared__ int wsum[32], wmax[32];
    __shared__ int carry_p, carry_f, block_total, next_f;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = (blockDim.x + 31) >> 5;
    for (int u = threadIdx.x; u < nbins * ncls; u += blockDim.x) hist[u] = 0;
    for (int u = threadIdx.x; u < nbins; u += blockDim.x) cums[u] = cum[b * nbins + u];
    if (threadIdx.x == 0) { carry_p = 0; carry_f = 0; }
    __syncthreads();
    const int* oj = ord + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    const int* cb = counts_bt + (long long)b * n;
    for (int base = 0; base < n; base += blockDim.x) {
        const int i = base + threadIdx.x;
        int c = 0, row = 0, start = 0;
        if (i < n) { row = oj[i]; c = cb[row]; start = (i == 0) || !tj[i - 1]; }
        int v = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += t; }
        if (lane == 31) wsum[warp] = v;
        __syncthreads();
        if (threadIdx.x == 0) { int run = 0; for (int w = 0; w < nwarps; ++w) { const int t = wsum[w]; wsum[w] = run; run += t; } block_total = run; }
        __syncthreads();
        const int pos = carry_p + wsum[warp] + v - c;       // draws before row i
        int q = (i < n && start) ? pos : -1;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up_sync(0xffffffffu, q, o); if (lane >= o && t > q) q = t; }
        if (lane == 31) wmax[warp] = q;
        __syncthreads();
        if (threadIdx.x == 0) { int run = carry_f; for (int w = 0; w < nwarps; ++w) { const int t = wmax[w]; wmax[w] = run; if (t > run) run = t; } next_f = run; }
        __syncthreads();
        const int first = q > wmax[warp] ? q : wmax[warp];   // draws before the first row of this row's run
        if (i < n && c > 0) {
            int cur = 0;
            while (first >= cums[cur] && cur < nbins - 1) ++cur;
            atomicAdd(&hist[cur * ncls + ycode[row]], c);
        }
        __syncthreads();
        if (threadIdx.x == 0) { carry_p += block_total; carry_f = next_f; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        long long hx[BMAX], hy[KMAX];
        for (int u = 0; u < nbins; ++u) hx[u] = 0;
        for (int c = 0; c < ncls; ++c) hy[c] = 0;
        for (int u = 0; u < nbins; ++u) for (int c = 0; c < ncls; ++c) { hx[u] += hist[u * ncls + c]; hy[c] += hist[u * ncls + c]; }
        const int m = carry_p;
        const {ACC} log_n = log(({ACC})m);
        {ACC} mi = 0.0;
        for (int u = 0; u < nbins; ++u) {
            if (hx[u] == 0) continue;
            const {ACC} log_hx = log(({ACC})hx[u]);
            for (int c = 0; c < ncls; ++c) {
                const long long nxy = hist[u * ncls + c];
                if (nxy == 0 || hy[c] == 0) continue;
                mi += (({ACC})nxy / ({ACC})m) * (log(({ACC})nxy) + log_n - log_hx - log(({ACC})hy[c]));
            }
        }
        out[(long long)b * kc + j] = mi < 0.0 ? 0.0 : mi;
    }
}
""")

_MODULE = RawModuleCache(_SRC)


#: Device bytes one column of a block holds at its peak: the float32 block, the int64 argsort, the sorted copy, the int32 order and the walk's int32 table, per row.
_BYTES_PER_COLUMN_ROW = 64
_BUDGET_FRACTION = 0.3


def _bootstrap_walk_column_block(cp, n_rows: int) -> int:
    """Columns per launch such that a block's temporaries stay within ``_BUDGET_FRACTION`` of what the memory pool can still hand out, at least 8 and at most ``_COLUMN_BLOCK``."""
    return pool_column_block(
        cp, n_rows, bytes_per_column_row=_BYTES_PER_COLUMN_ROW, budget_fraction=_BUDGET_FRACTION, min_block=8, max_block=_COLUMN_BLOCK, fallback=32, what="device bootstrap walk"
    )


def _release_pool_cache(cp) -> None:
    """Return the memory pool's unused cached blocks to the device (blocks in use are kept)."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        logger.debug("device bootstrap walk: could not release the pool cache (%s)", exc)


def _ties_on_device() -> bool:
    """Whether nearly distinct columns are walked on the device (``MLFRAME_BOOTSTRAP_RANK_TIES_DEVICE=0`` turns it off; default on).

    A run of equal values takes the bin of the draws before its first row whatever the draw order inside it, so the walk needs no host rule for ties; continuous float32 columns of 100000 rows nearly
    always hold a few equal neighbours (birthday collisions of 24-bit mantissas), and with the fully-distinct restriction almost no column ever reached the device.

    History: bench-attempt-rejected (2026-11-07, GTX 1050 Ti, 10 resamples): with the per-column sorts and one thread per (column, resample) the opt-in version lost to the host (200000 x 24 host 0.15 s / device
    0.35 s). 2026-11-16: the block is sorted by ONE 64-bit-key sort and walked by ``bootstrap_rank_walk_blocks`` (a block per (column, resample), the draws before a row by a block-wide prefix sum): 100000 x 100,
    10 resamples, host 0.62 s / device 0.23 s, MI equal to 1.8e-15; the kernel-tuning entry decides per host."""
    return os.environ.get("MLFRAME_BOOTSTRAP_RANK_TIES_DEVICE", "1").strip().lower() in ("1", "true", "on", "yes")


def distinct_walk_gpu(arr: np.ndarray, prep: dict, n_bins: int) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(mi (B, k), done (k,) bool)``: the MI of every fully distinct finite column of ``arr`` ``(n, k)`` for the resamples of ``prep`` (from ``_resample_tables`` with the walk tables), the other columns left to the host.

    ``None`` when the device path does not apply (too many classes or bins, no cupy).
    """
    n, k = int(arr.shape[0]), int(arr.shape[1])
    n_classes, n_rep = int(prep["n_classes"]), int(prep["counts_t"].shape[1])
    if n_classes > MAX_CLASSES or int(n_bins) > MAX_BINS:
        return None
    try:
        import cupy as cp
    except Exception as exc:
        logger.debug("device bootstrap walk unavailable (%s)", exc)
        return None
    _release_pool_cache(cp)  # start from a pool without cached blocks of other sizes: the argsort asks thrust for large temporaries that must fit contiguously
    try:
        counts = cp.asarray(np.ascontiguousarray(prep["counts_t"], dtype=np.int32))
        ycode = cp.asarray(np.ascontiguousarray(prep["y_codes"], dtype=np.int32))
        cum = cp.asarray(np.ascontiguousarray(prep["cum"], dtype=np.int32))
        mi = np.zeros((n_rep, k), dtype=np.float64)  # Intentional float64: MI statistic matrix
        done = np.zeros(k, dtype=bool)
        block_cols = _bootstrap_walk_column_block(cp, n)
        counts_bt = cp.ascontiguousarray(counts.T)  # (resample, row): one resample's multiplicities contiguous for the block that walks it
        walk = _MODULE.get(cp).get_function("bootstrap_rank_walk_blocks")
        for c0 in range(0, k, block_cols):
            c1 = min(k, c0 + block_cols)
            width = c1 - c0
            # (columns, rows) block built on the device by the coalesced tiled transpose; ONE sort of the whole block: the 64-bit key is the column index above the order-preserving key of the value
            # (-0.0 made +0.0 so equal values have equal keys), so the sorted array holds every column's sorted rows one after the other and the per-column sort launches are gone
            xd = _transpose_to_cm(cp.asarray(np.ascontiguousarray(arr[:, c0:c1]))) + np.float32(0.0)
            finite = cp.isfinite(xd).all(axis=1)
            bits = xd.view(cp.uint32)
            mono = cp.where(bits >> 31 == 1, ~bits, bits | np.uint32(0x80000000)).astype(cp.uint64)
            keys = ((cp.arange(width, dtype=cp.uint64)[:, None] << np.uint64(32)) | mono).ravel()
            del xd, bits, mono
            order = device_argsort(keys)
            tie = cp.zeros(width * n, dtype=cp.uint8)
            tie[:-1] = (keys[order][1:] == keys[order][:-1]).astype(cp.uint8)  # a row equal to the next sorted row (the high word keeps columns apart)
            tie = tie.reshape(width, n)
            n_equal = tie.sum(axis=1)
            ord32 = (order.reshape(width, n) - (cp.arange(width, dtype=cp.int64) * n)[:, None]).astype(cp.int32)
            del keys, order
            take = finite & (n_equal <= int(MAX_TIE_FRACTION * n if _ties_on_device() else 0))
            ok = np.flatnonzero(cp.asnumpy(take))
            if ok.size == 0:
                continue
            # Intentional float64: device MI statistic matrix, same element type as the host array
            out = cp.zeros((n_rep, width), dtype=cp.float64)
            walk((width * n_rep,), (256,), (ord32, tie, counts_bt, ycode, cum, np.int32(n), np.int32(width), np.int32(n_rep), np.int32(n_bins), np.int32(n_classes), out))  # launch-ok: the block-wide scan needs a warp multiple, 256 threads is the chunk of sorted rows a block walks per step
            mi[:, c0 + ok] = cp.asnumpy(out)[:, ok]
            done[c0 + ok] = True
        return mi, done
    except Exception as exc:  # out of device memory or a driver fault: the host walk scores every column
        logger.warning("device bootstrap rank walk failed (%s: %s); using the host walk", type(exc).__name__, exc)
        try:
            import cupy as _cp

            if isinstance(exc, _cp.cuda.memory.OutOfMemoryError):
                from ._fe_gpu_vram import note_gpu_oom

                note_gpu_oom()  # this call site adaptively shrinks its own block (pool_column_block) but can still OOM; arm the shared cooldown for other call sites.
        except Exception:  # nosec B110 - best-effort cooldown note, never let it mask the original failure
            logger.debug("GPU out-of-memory cooldown note failed", exc_info=True)
        return None
    finally:
        # The argsort / take temporaries come in sizes no later request reuses; handing the unused pool blocks back keeps the pool from filling up with them (a 525 MB request of the CMI scorer failed against them).
        _release_pool_cache(cp)


def use_device_walk(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device walk: ``MLFRAME_BOOTSTRAP_RANK_DEVICE=0|1`` forces a path, else the per-host tuner (fallback: the host walk)."""
    forced = os.environ.get("MLFRAME_BOOTSTRAP_RANK_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_WALK_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_bootstrap_rank_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_WALK_FALLBACK_ROWS


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


def _make_inputs(dims: dict) -> tuple:
    """A 60-column continuous frame of ``dims["n_samples"]`` rows, a binary target and 10 resamples of 80% of the rows."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    arr = rng.normal(size=(n, 60)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int64)
    m = max(2, round(0.8 * n))
    reps = [np.sort(rng.integers(0, n, m)).astype(np.int64) for _ in range(10)]
    return (arr, y, reps)


@contextmanager
def _forced_env(name: str, value: str):
    """Set ``name`` to ``value`` for the block, then put back the value it had before (or remove it when it was unset).

    The sweep runs in the middle of a fit, so popping the variable would drop an override the user set for the rest of the process.
    """
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _run_sweep() -> list:
    """Time the host walk and the device walk per row count; the fastest variant whose MI agrees with the host's wins each region."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    from . import _bootstrap_rank_mi_njit as host

    def host_variant(arr, y, reps):
        """Host walk with the device path switched off."""
        with _forced_env("MLFRAME_BOOTSTRAP_RANK_DEVICE", "0"):
            return host.bootstrap_rank_mi(arr, y, reps, 10)[0]

    def device_variant(arr, y, reps):
        """Device walk forced on."""
        with _forced_env("MLFRAME_BOOTSTRAP_RANK_DEVICE", "1"):
            return host.bootstrap_rank_mi(arr, y, reps, 10)[0]

    return cast(
        list,
        sweep_backend_grid(
            {"host": host_variant, "device": device_variant},
            {"n_samples": _SWEEP_ROWS},
            _make_inputs,
            reference="host",
            repeats=3,
            equiv_rtol=1e-9,
            equiv_atol=1e-9,
        ),
    )


#: Rows from which the device walk is the fallback choice when the kernel-tuning cache has no entry.
DEVICE_WALK_FALLBACK_ROWS = 30_000


def _bootstrap_walk_fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the device walk from ``DEVICE_WALK_FALLBACK_ROWS`` rows. Re-measured 2026-11-01 on the development GPU (105 columns, 10 resamples, min of 5): 30000 rows host 0.05 s / device 0.04 s,
    100000 rows host 0.29 s / device 0.15 s. (The earlier note that the host walk won, 0.24 against 0.56 s, was taken under contention and does not reproduce.)"""
    return device_from_rows(n_samples, DEVICE_WALK_FALLBACK_ROWS)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_bootstrap_rank_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_bootstrap_walk_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_bootstrap_rank_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_bootstrap_rank_device failed: %s", e)
    _SPEC = None

__all__ = ["distinct_walk_gpu", "use_device_walk"]
