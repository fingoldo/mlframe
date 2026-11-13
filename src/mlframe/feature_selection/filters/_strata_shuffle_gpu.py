"""Within-stratum shuffles of a z-sorted column, one thread per (permutation, stratum), instead of one flat radix sort of packed 64-bit keys.

The conditional permutation null of the CMI redundancy gate shuffles a candidate inside every stratum of the support. With many short strata each shuffle is a few-element Fisher-Yates
that one thread finishes alone, while the flat sort moves ``nperm * n`` 64-bit keys through cub's radix passes whatever the strata look like.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
from pyutilz.performance.gpu_hw_launch import launch_1d

from ._kernel_types import render

#: Opt-in (``MLFRAME_STRATA_SHUFFLE_KERNEL=1``): the whole-fit ABBA at 100000 x 100 measured 72.3 / 74.8 s with it against 72.1 / 70.4 s without (noise, no fit-level win).
#: Strata from which the kernel runs. Measured at n=100000, nperm=25 (flat sort 9.2 ms at every strata count): 300 strata 12.5 ms, 1000 strata 11.8 ms, 5000 strata 4.5 ms,
#: 20000 strata 1.4 ms, 43000 strata 1.0 ms; the serial per-thread walk is the cost, so the kernel only wins when strata are short and many.
MIN_STRATA = 3000

_SRC = r"""
// coalescing-ok: one thread owns a whole stratum run (short by the MIN_STRATA gate), so its reads and swaps are scattered by design; 1.0-4.5 ms against the 9.2 ms flat sort at 5000-43000 strata (nvprof-timed micro-benchmark in the tracker, FC11-28)
extern "C" __global__
void within_stratum_shuffle(const long long* __restrict__ x_sorted, const int* __restrict__ starts, const int n_strata, const long long n, const int nperm,
                            const unsigned long long* __restrict__ seed_p, long long* __restrict__ out) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)nperm * n_strata) return;
    const int p = (int)(tid / n_strata);
    const int s = (int)(tid - (long long)p * n_strata);
    const int lo = starts[s], hi = starts[s + 1];
    long long* dst = out + (long long)p * n;
    for (int i = lo; i < hi; ++i) dst[i] = x_sorted[i];
    unsigned long long state = seed_p[0] ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(p + 1)) ^ (0xBF58476D1CE4E5B9ULL * (unsigned long long)(s + 1));
    for (int i = hi - lo - 1; i > 0; --i) {
        state += 0x9E3779B97F4A7C15ULL;
        unsigned long long z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        const int j = (int)(((z >> 32) * (unsigned long long)(i + 1)) >> 32);
        const long long t = dst[lo + i];
        dst[lo + i] = dst[lo + j];
        dst[lo + j] = t;
    }
}
"""
_MODULES: dict = {}
_LOCK = threading.Lock()


def kernel(cp: Any, real: Any = None) -> Any:
    """``within_stratum_shuffle`` compiled once per process (integer data, so ``real`` does not change the module)."""
    with _LOCK:
        mod = _MODULES.get("m")
        if mod is None:
            mod = _MODULES["m"] = cp.RawModule(code=render(_SRC))
        return mod.get_function("within_stratum_shuffle")


def strata_shuffle(cp: Any, rs: Any, x_sorted_d: Any, z_rank_1d: Any, nperm: int, nrank: int) -> Any:
    """``(nperm, n)`` int64 within-stratum shuffles of the z-sorted ``x_sorted_d``; ``z_rank_1d`` is the monotone dense stratum rank of the sorted rows, ``nrank`` the stratum count.

    The generator seed is drawn on the device from ``rs``, so the call needs no host sync.
    """
    n = int(x_sorted_d.shape[0])
    starts = cp.searchsorted(z_rank_1d, cp.arange(nrank + 1, dtype=z_rank_1d.dtype)).astype(cp.int32)
    seed = rs.randint(0, 1 << 62, size=1, dtype=cp.int64).astype(cp.uint64)
    out = cp.empty((nperm, n), dtype=cp.int64)
    fn = kernel(cp)
    grid, block = launch_1d(fn, nperm * nrank)
    fn(grid, block, (x_sorted_d.astype(cp.int64, copy=False), starts, np.int32(nrank), np.int64(n), np.int32(nperm), seed, out))
    return out
