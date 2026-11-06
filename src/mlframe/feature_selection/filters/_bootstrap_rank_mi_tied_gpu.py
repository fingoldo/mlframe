"""Device twin of the counting-sort replicate MI of ``_bootstrap_rank_mi_njit._replicate_mis`` (the path of columns with repeated values).

Win axis: the host sorts each column once and then, per resample, counting-sorts the draws by the dense rank of their row (two passes over the 80000 draws of every
resample, random reads of the rank and level tables): about 3.3 ms per 100000-row column for 10 resamples, 2.9 s for the 800 columns of the numeric-decomposition
family. The dense ranks stay a host job (``dense_ranks_f32``, the same radix sort); the resample passes run here, one block per (column, resample).

What the host computes: the position of a draw in the stable counting sort is ``start[level] + (draws of that level before it)``, and its bin is ``binpos[b, position]``.
A level whose whole position range ``[start, start + count)`` lies in one bin sends every draw of it to that bin, whatever their order, so those draws are histogrammed
in parallel. Only the levels that straddle a bin end (at most ``n_bins - 1`` per resample) need the draw order: one warp walks the draws in order and gives each
draw of a straddling level its ordinal by warp ballots. The ``(bin, class)`` table is then the host's, and the MI is summed in the host's order in double precision.

Columns with more levels than the shared-memory table holds stay on the host. ``MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE=0|1`` forces a path; otherwise the kernel tuning
cache (``fe_bootstrap_rank_tied_device``) picks host or device per row count.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ._kernel_types import render
from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

MAX_CLASSES = 16
MAX_BINS = 32
#: Levels per column the shared table holds (int32 start + one byte of straddle flag each); a column with more stays on the host.
MAX_LEVELS = 6144
_COLUMN_BLOCK = 128

_SRC = r"""
#define KMAX 16
#define BMAX 32
#define PRIV_LEVELS 64
extern "C" __global__ void tied_replicate_mi(const short* __restrict__ rk, const int* __restrict__ idx, const short* __restrict__ binpos, const int* __restrict__ ycode,
                                             const int* __restrict__ lens, const int* __restrict__ nlev, const long long n, const int m_max, const int B,
                                             const int nbins, const int ncls, {ACC}* __restrict__ out) {
    extern __shared__ int sh[];
    const int j = blockIdx.x / B;
    const int b = blockIdx.x - j * B;
    const int nl = nlev[j];
    int* start = sh;                                          // nl + 1 counts, then exclusive starts
    unsigned char* sid = (unsigned char*)(sh + nl + 1);       // index of the level among the straddling ones, 255 when it lies inside one bin
    const int nw = (int)(blockDim.x >> 5);
    const int cells = nbins * ncls;
    int* whist = sh + nl + 1 + (nl + 3) / 4;                  // one (bin, class) table per warp: a digit column has 20 cells, one shared table serialised every atomic
    int* wcnt = whist + nw * cells;                           // per-warp level counts when the levels are few (priv_levels)
    const bool priv_levels = nl <= PRIV_LEVELS;
    __shared__ int warp_tot[32];
    __shared__ int n_strad;
    __shared__ int strad_level[BMAX];
    __shared__ int wstrad[32 * BMAX];                         // per warp and straddling level: the count, then the position of the warp's first draw
    const int tid = threadIdx.x;
    const int m = lens[b];
    const short* rj = rk + (long long)j * n;
    const int* ib = idx + (long long)b * m_max;
    const short* bp = binpos + (long long)b * m_max;
    for (int u = tid; u <= nl; u += blockDim.x) start[u] = 0;
    for (int u = tid; u < nw * cells; u += blockDim.x) whist[u] = 0;
    if (priv_levels) for (int u = tid; u < nw * nl; u += blockDim.x) wcnt[u] = 0;
    __syncthreads();
    int* myhist = whist + (tid >> 5) * cells;
    if (priv_levels) {
        int* mycnt = wcnt + (tid >> 5) * nl;
        for (int t = tid; t < m; t += blockDim.x) atomicAdd(&mycnt[rj[ib[t]]], 1);
        __syncthreads();
        for (int u = tid; u < nl; u += blockDim.x) { int c = 0; for (int w = 0; w < nw; ++w) c += wcnt[w * nl + u]; start[u] = c; }
    } else {
        for (int t = tid; t < m; t += blockDim.x) atomicAdd(&start[rj[ib[t]]], 1);
    }
    __syncthreads();
    // exclusive scan of the nl counts: a contiguous chunk per thread, warp shuffles over the chunk sums, then the warp totals
    const int cw = (nl + blockDim.x - 1) / blockDim.x;
    const int u0 = min(nl, tid * cw), u1 = min(nl, u0 + cw);
    int s = 0;
    for (int u = u0; u < u1; ++u) s += start[u];
    int v = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int x = __shfl_up_sync(0xffffffffu, v, o); if ((tid & 31) >= o) v += x; }
    if ((tid & 31) == 31) warp_tot[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) { int run = 0; for (int w = 0; w < (int)(blockDim.x >> 5); ++w) { const int x = warp_tot[w]; warp_tot[w] = run; run += x; } }
    __syncthreads();
    if (tid == 0) n_strad = 0;
    __syncthreads();
    int run = warp_tot[tid >> 5] + v - s;
    for (int u = u0; u < u1; ++u) {
        const int c = start[u];
        start[u] = run;
        if (c > 0 && bp[run] != bp[run + c - 1]) { const int q = atomicAdd(&n_strad, 1); sid[u] = (unsigned char)q; strad_level[q] = u; } else sid[u] = 255;
        run += c;
    }
    __syncthreads();
    const int ns = n_strad;  // at most nbins - 1: every straddling level holds a distinct bin end
    // Each warp takes one contiguous range of draws and walks it 32 at a time (coalesced, lane order = draw order). A draw of a level inside one bin goes to that bin.
    // A draw of a straddling level needs its ordinal among that level's draws: the level's count over the earlier warps' ranges (a scan over the warps) plus the
    // count of earlier draws in the warp's range, from warp ballots.
    const int lane = tid & 31, w = tid >> 5;
    const unsigned int lt = (1u << lane) - 1u;
    const int seg = ((m + nw - 1) / nw + 31) & ~31;
    const int t0 = min(m, w * seg), t1 = min(m, t0 + seg);
    for (int u = tid; u < nw * BMAX; u += blockDim.x) wstrad[u] = 0;
    __syncthreads();
    for (int t = t0 + lane; t - lane < t1; t += 32) {
        int q = 255, r = 0;
        if (t < t1) {
            r = ib[t];
            const int L = rj[r];
            q = sid[L];
            if (q == 255) atomicAdd(&myhist[bp[start[L]] * ncls + ycode[r]], 1);
        }
        unsigned int mask = __ballot_sync(0xffffffffu, q != 255);
        while (mask) {
            const int ql = __shfl_sync(0xffffffffu, q, __ffs(mask) - 1);
            const unsigned int same = __ballot_sync(0xffffffffu, q == ql);
            if (lane == 0) wstrad[w * BMAX + ql] += __popc(same);
            mask &= ~same;
        }
    }
    __syncthreads();
    if (tid < ns) { int acc = start[strad_level[tid]]; for (int v2 = 0; v2 < nw; ++v2) { const int z = wstrad[v2 * BMAX + tid]; wstrad[v2 * BMAX + tid] = acc; acc += z; } }
    __syncthreads();
    if (ns > 0) {
        for (int t = t0 + lane; t - lane < t1; t += 32) {
            int q = 255, r = 0;
            if (t < t1) { r = ib[t]; q = sid[rj[r]]; }
            unsigned int mask = __ballot_sync(0xffffffffu, q != 255);
            while (mask) {
                const int ql = __shfl_sync(0xffffffffu, q, __ffs(mask) - 1);
                const unsigned int same = __ballot_sync(0xffffffffu, q == ql);
                const int base = wstrad[w * BMAX + ql];
                if (q == ql) atomicAdd(&myhist[bp[base + __popc(same & lt)] * ncls + ycode[r]], 1);
                __syncwarp();
                if (lane == 0) wstrad[w * BMAX + ql] = base + __popc(same);
                __syncwarp();
                mask &= ~same;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        long long hx[BMAX], hy[KMAX];
        for (int bb = 0; bb < nbins; ++bb) hx[bb] = 0;
        for (int c = 0; c < ncls; ++c) hy[c] = 0;
        for (int u = 0; u < cells; ++u) { int c = 0; for (int w = 0; w < nw; ++w) c += whist[w * cells + u]; whist[u] = c; }
        for (int bb = 0; bb < nbins; ++bb) for (int c = 0; c < ncls; ++c) { hx[bb] += whist[bb * ncls + c]; hy[c] += whist[bb * ncls + c]; }
        const {ACC} log_n = log(({ACC})m);
        {ACC} mi = 0.0;
        for (int bb = 0; bb < nbins; ++bb) {
            if (hx[bb] == 0) continue;
            const {ACC} log_hx = log(({ACC})hx[bb]);
            for (int c = 0; c < ncls; ++c) {
                const long long nxy = whist[bb * ncls + c];
                if (nxy == 0 || hy[c] == 0) continue;
                mi += (({ACC})nxy / ({ACC})m) * (log(({ACC})nxy) + log_n - log_hx - log(({ACC})hy[c]));
            }
        }
        out[(long long)b * gridDim.x / B + j] = mi < 0.0 ? 0.0 : mi;
    }
}
"""

_MODULE = RawModuleCache(render(_SRC))


def use_device_tied(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device counting sort: ``MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE=0|1`` forces a path, else the per-host tuner (fallback: the device
    from ``DEVICE_TIED_MIN_ROWS`` rows)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False  # a run without a GPU (MLFRAME_DISABLE_GPU=1, CUDA_VISIBLE_DEVICES=-1 / "") never tries the device
    forced = os.environ.get("MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return _fallback_choice(int(n_rows)) == "device"
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_bootstrap_rank_tied_device: KTC choose() failed, using the fallback: %s", exc)
        return _fallback_choice(int(n_rows)) == "device"


#: Rows from which the device counting sort is the fallback choice. Measured on the GTX 1050 Ti (856 rounding / digit / raw columns, 10 resamples of 80%):
#: 100000 rows host 2.98 s, device 1.75 s.
DEVICE_TIED_MIN_ROWS = 30_000
_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the device from ``DEVICE_TIED_MIN_ROWS`` rows."""
    return "device" if int(n_samples) >= DEVICE_TIED_MIN_ROWS else "host"


def _make_inputs(dims: dict) -> tuple:
    """A 64-column frame of rounded and digit-like columns (repeated values) of ``dims["n_samples"]`` rows, a binary target and 10 resamples of 80% of the rows."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    base = rng.normal(size=(n, 16)) * 30
    arr = np.concatenate([np.round(base), np.round(base * 10) / 10, np.floor(np.abs(base)) % 10, np.round(base / 7)], axis=1).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int64)
    m = max(2, round(0.8 * n))
    reps = [rng.integers(0, n, m).astype(np.int64) for _ in range(10)]
    return (np.ascontiguousarray(arr), y, reps)


def _run_sweep() -> list:
    """Time the host counting sort and the device one per row count; the fastest variant whose MI agrees with the host's wins each region."""
    from typing import cast

    from pyutilz.dev.benchmarking import sweep_backend_grid

    from . import _bootstrap_rank_mi_njit as host

    def _variant(flag: str):
        """``bootstrap_rank_mi`` with the device counting sort forced to ``flag``."""

        def run(arr, y, reps):
            """One scoring of the frame."""
            from ._bootstrap_rank_mi_gpu import _forced_env

            with _forced_env("MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE", flag):
                return host.bootstrap_rank_mi(arr, y, reps, 10)[0]

        return run

    return cast(
        list,
        sweep_backend_grid({"host": _variant("0"), "device": _variant("1")}, {"n_samples": _SWEEP_ROWS}, _make_inputs, reference="host", repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12),
    )


_PRIV_LEVELS = 64  # PRIV_LEVELS of the kernel


def _shared_bytes(n_levels: int, n_warps: int, n_bins: int, n_classes: int) -> int:
    """Dynamic shared bytes of one block: the level starts, the straddle ids, a (bin, class) table per warp and, for few levels, a level table per warp."""
    words = n_levels + 1 + (n_levels + 3) // 4 + n_warps * n_bins * n_classes + (n_warps * n_levels if n_levels <= _PRIV_LEVELS else 0)
    return 4 * words


def _block_width(kernel, shared_bytes: int, m: int) -> int:
    """Threads per block from the device limits (``fixed_shared_block_size``: the level table is fixed per block), 256 without device information."""
    from pyutilz.performance.gpu_hw_launch import fixed_shared_block_size

    try:
        width = fixed_shared_block_size(shared_bytes, regs_per_thread=int(kernel.num_regs), static_smem=int(kernel.shared_size_bytes), min_width=64, max_width=256, work_items=m)  # the shared size is computed for 8 warps
    except Exception as exc:  # best-effort: only the launch width changes, never a result
        logger.debug("tied replicate MI block width: device limits unavailable (%s)", exc)
        width = 0
    return width or 256


def tied_replicate_mi_stream(bits: np.ndarray, prep: dict, n_bins: int, rank_fn) -> "Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]":
    """``(mi (B, k), done (k,), ok (k,))`` for the float32 matrix given as raw bits ``(n, k)``: the columns with repeated values and at most ``MAX_LEVELS`` levels are
    scored by the device counting sort (``done``), ``ok`` marks the finite columns. ``rank_fn(bits_block) -> (rk, n_levels, ok)`` gives the host dense ranks of a block of
    columns; a block is ranked on the host while the device scores the previous one. ``None`` when the path does not apply or the device fails (the host scores all)."""
    n_classes = int(prep["n_classes"])
    if n_classes > MAX_CLASSES or int(n_bins) > MAX_BINS:
        return None
    n, k = int(bits.shape[0]), int(bits.shape[1])
    mi = np.zeros((len(prep["lens"]), k), dtype=np.float64)  # Intentional float64: MI statistic matrix, the host's element type
    done = np.zeros(k, dtype=bool)
    ok = np.zeros(k, dtype=bool)
    try:
        import cupy as cp

        kernel = _MODULE.get(cp).get_function("tied_replicate_mi")
        idx = cp.asarray(np.ascontiguousarray(prep["idx"], dtype=np.int32))
        binpos = cp.asarray(np.ascontiguousarray(prep["binpos"], dtype=np.int16))
        ycode = cp.asarray(np.ascontiguousarray(prep["y_codes"], dtype=np.int32))
        lens = cp.asarray(np.ascontiguousarray(prep["lens"], dtype=np.int32))
        n_rep, m_max = int(idx.shape[0]), int(idx.shape[1])
        pending = None
        for c0 in range(0, k, _COLUMN_BLOCK):
            c1 = min(k, c0 + _COLUMN_BLOCK)
            rk, n_levels, ok_b = rank_fn(np.ascontiguousarray(bits[:, c0:c1]))  # host work while the device runs the previous block
            n_levels, ok_b = np.asarray(n_levels), np.asarray(ok_b)
            ok[c0:c1] = ok_b
            if pending is not None:
                res, cols = pending
                mi[:, cols] = cp.asnumpy(res)
                done[cols] = True
                pending = None
            local = np.flatnonzero(ok_b & (n_levels < n) & (n_levels <= MAX_LEVELS))
            if local.size == 0:
                continue
            nl = n_levels[local].astype(np.int32)
            width = _block_width(kernel, _shared_bytes(int(nl.max()), 8, int(n_bins), n_classes), m_max)
            shared = _shared_bytes(int(nl.max()), width // 32, int(n_bins), n_classes)
            rk_dev = cp.asarray(np.ascontiguousarray(rk[local]).astype(np.int16))
            res = cp.zeros((n_rep, local.size), dtype=cp.float64)  # Intentional float64: MI statistic matrix
            kernel(
                (int(local.size) * n_rep,), (width,),
                (rk_dev, idx, binpos, ycode, lens, cp.asarray(nl), np.int64(n), np.int32(m_max), np.int32(n_rep), np.int32(n_bins), np.int32(n_classes), res),
                shared_mem=shared,
            )
            pending = (res, c0 + local)
        if pending is not None:
            res, cols = pending
            mi[:, cols] = cp.asnumpy(res)
            done[cols] = True
    except Exception as exc:
        logger.warning("device tied replicate MI failed (%s: %s); using the host walk", type(exc).__name__, exc)
        return None
    if not done.any():
        return None
    return mi, done, ok


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_bootstrap_rank_tied_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_bootstrap_rank_tied_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_bootstrap_rank_tied_device failed: %s", e)
    _SPEC = None

__all__ = ["tied_replicate_mi_stream", "use_device_tied"]
