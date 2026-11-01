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
from typing import cast

import numpy as np

from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

MAX_CLASSES = 16
MAX_BINS = 32
_COLUMN_BLOCK = 256

_SRC = r"""
#define KMAX 16
#define BMAX 32
extern "C" __global__ void bootstrap_rank_walk(const int* __restrict__ ord, const int* __restrict__ counts, const int* __restrict__ ycode, const int* __restrict__ cum,
                                               const int* __restrict__ lens, const int n, const int kc, const int B, const int nbins, const int ncls, double* __restrict__ out) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= (long long)kc * B) return;
    const int j = (int)(t / B);
    const int b = (int)(t - (long long)j * B);
    const int* oj = ord + (long long)j * n;
    long long hist[BMAX * KMAX];
    for (int u = 0; u < nbins * ncls; ++u) hist[u] = 0;
    int pos = 0, cur = 0;
    for (int i = 0; i < n; ++i) {
        const int row = oj[i];
        int rem = counts[(long long)row * B + b];
        if (rem == 0) continue;
        const int c = ycode[row];
        while (rem > 0) {
            while (pos >= cum[b * nbins + cur] && cur < nbins - 1) ++cur;
            const int room = cum[b * nbins + cur] - pos;
            const int take = room > 0 ? (rem < room ? rem : room) : rem;
            hist[cur * ncls + c] += take;
            pos += take;
            rem -= take;
        }
    }
    long long hx[BMAX], hy[KMAX];
    for (int u = 0; u < nbins; ++u) hx[u] = 0;
    for (int c = 0; c < ncls; ++c) hy[c] = 0;
    for (int u = 0; u < nbins; ++u) for (int c = 0; c < ncls; ++c) { hx[u] += hist[u * ncls + c]; hy[c] += hist[u * ncls + c]; }
    const int m = pos;
    const double log_n = log((double)m);
    double mi = 0.0;
    for (int u = 0; u < nbins; ++u) {
        if (hx[u] == 0) continue;
        const double log_hx = log((double)hx[u]);
        for (int c = 0; c < ncls; ++c) {
            const long long nxy = hist[u * ncls + c];
            if (nxy == 0 || hy[c] == 0) continue;
            mi += ((double)nxy / (double)m) * (log((double)nxy) + log_n - log_hx - log((double)hy[c]));
        }
    }
    out[(long long)b * kc + j] = mi < 0.0 ? 0.0 : mi;
}
"""

_MODULE = RawModuleCache(_SRC)


#: Device bytes one column of a block holds at its peak: the float32 block, the int64 argsort, the sorted copy, the int32 order and the walk's int32 table, per row.
_BYTES_PER_COLUMN_ROW = 32
_BUDGET_FRACTION = 0.3


def _column_block(cp, n_rows: int) -> int:
    """Columns per launch such that a block's temporaries stay within ``_BUDGET_FRACTION`` of what the memory pool can still hand out (the smaller of the device's free memory and the pool limit minus its used bytes), at least 8 and at most ``_COLUMN_BLOCK``."""
    try:
        pool = cp.get_default_memory_pool()
        free_device = int(cp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        limit = int(pool.get_limit())
        room = free_device if limit <= 0 else min(free_device, max(0, limit - int(pool.used_bytes())))
    except Exception as exc:
        logger.debug("device bootstrap walk: free memory unknown (%s); small blocks", exc)
        return 32
    return int(max(8, min(_COLUMN_BLOCK, (_BUDGET_FRACTION * room) // (_BYTES_PER_COLUMN_ROW * max(1, n_rows)))))


def _release_pool_cache(cp) -> None:
    """Return the memory pool's unused cached blocks to the device (blocks in use are kept)."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        logger.debug("device bootstrap walk: could not release the pool cache (%s)", exc)


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
        kernel = _MODULE.get(cp).get_function("bootstrap_rank_walk")
        counts = cp.asarray(np.ascontiguousarray(prep["counts_t"], dtype=np.int32))
        ycode = cp.asarray(np.ascontiguousarray(prep["y_codes"], dtype=np.int32))
        cum = cp.asarray(np.ascontiguousarray(prep["cum"], dtype=np.int32))
        lens = cp.asarray(np.ascontiguousarray(prep["lens"], dtype=np.int32))
        mi = np.zeros((n_rep, k), dtype=np.float64)  # Intentional float64: MI statistic matrix
        done = np.zeros(k, dtype=bool)
        block_cols = _column_block(cp, n)
        for c0 in range(0, k, block_cols):
            c1 = min(k, c0 + block_cols)
            block = cp.asarray(np.ascontiguousarray(arr[:, c0:c1]))
            finite = cp.isfinite(block).all(axis=0)
            order = cp.argsort(block, axis=0)
            sv = cp.take_along_axis(block, order, axis=0)
            distinct = finite & (sv[1:] != sv[:-1]).all(axis=0)
            ok = np.flatnonzero(cp.asnumpy(distinct))
            if ok.size == 0:
                continue
            ord_t = cp.ascontiguousarray(order[:, cp.asarray(ok)].T.astype(cp.int32))
            out = cp.zeros((n_rep, ok.size), dtype=cp.float64)
            threads = 128
            grid = (int(ok.size) * n_rep + threads - 1) // threads
            kernel((grid,), (threads,), (ord_t, counts, ycode, cum, lens, np.int32(n), np.int32(ok.size), np.int32(n_rep), np.int32(n_bins), np.int32(n_classes), out))
            mi[:, c0 + ok] = cp.asnumpy(out)
            done[c0 + ok] = True
        return mi, done
    except Exception as exc:  # out of device memory or a driver fault: the host walk scores every column
        logger.warning("device bootstrap rank walk failed (%s: %s); using the host walk", type(exc).__name__, exc)
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


def _run_sweep() -> list:
    """Time the host walk and the device walk per row count; the fastest variant whose MI agrees with the host's wins each region."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    from . import _bootstrap_rank_mi_njit as host

    def host_variant(arr, y, reps):
        """Host walk with the device path switched off."""
        os.environ["MLFRAME_BOOTSTRAP_RANK_DEVICE"] = "0"
        try:
            return host.bootstrap_rank_mi(arr, y, reps, 10)[0]
        finally:
            os.environ.pop("MLFRAME_BOOTSTRAP_RANK_DEVICE", None)

    def device_variant(arr, y, reps):
        """Device walk forced on."""
        os.environ["MLFRAME_BOOTSTRAP_RANK_DEVICE"] = "1"
        try:
            return host.bootstrap_rank_mi(arr, y, reps, 10)[0]
        finally:
            os.environ.pop("MLFRAME_BOOTSTRAP_RANK_DEVICE", None)

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


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the device walk from ``DEVICE_WALK_FALLBACK_ROWS`` rows. Re-measured 2026-11-01 on the development GPU (105 columns, 10 resamples, min of 5): 30000 rows host 0.05 s / device 0.04 s,
    100000 rows host 0.29 s / device 0.15 s. (The earlier note that the host walk won, 0.24 against 0.56 s, was taken under contention and does not reproduce.)"""
    return "device" if int(n_samples) >= DEVICE_WALK_FALLBACK_ROWS else "host"


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_bootstrap_rank_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_bootstrap_rank_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_bootstrap_rank_device failed: %s", e)
    _SPEC = None

__all__ = ["distinct_walk_gpu", "use_device_walk"]
