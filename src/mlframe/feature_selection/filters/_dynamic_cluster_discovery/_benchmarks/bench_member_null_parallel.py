"""Bench: DCD member permutation-null -- serial mutate-restore loop vs the prange-parallel kernel.

The member null draws B shuffles of ONE cluster-member column and recomputes I(member; y | Selected-anchor)
under each. Pre-carve it ran serially (~1 core). ``_dcd_swap_null.run_member_null`` pre-generates all B shuffles
(SAME rng -> bit-identical p-value) then ``prange``s the per-draw conditional MI over a thread-local shuffled
column (no frame copy, no mutate-restore). This bench measures wall + confirms the p-value is unchanged.

Run:  python -m mlframe.feature_selection.filters._dynamic_cluster_discovery._benchmarks.bench_member_null_parallel

Measured (2026-07-06, 22 numba threads, CPU-only): p-value bit-identical across all configs; speedup is
memory-bandwidth-gated by the joint (X,Y,Z) frequency-array size (grows with |Z| and nb), so it is largest at
small |Z| / large n:
    n=30k  |z|=2  B=199 : serial 1.11s -> parallel 0.24s  = 4.56x
    n=100k |z|=4  B=199 : serial 7.71s -> parallel 1.91s  = 4.04x
    n=100k |z|=2  B=199 : serial 3.23s -> parallel 1.90s  = 1.70x
    n=300k |z|=3  B=199 : serial 14.70s -> parallel 6.49s = 2.26x
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from ...info_theory import conditional_mi, entropy, merge_vars
from ..._numba_utils import unpack_and_sort
from .. import _dcd_swap_null as K


def _serial(data, nbins, m, y, z, mr, B, seed):
    rng = np.random.default_rng(seed)
    _, fz, _ = merge_vars(data, z, None, nbins); hz = float(entropy(fz))
    _, fyz, _ = merge_vars(data, unpack_and_sort(y, z), None, nbins); hyz = float(entropy(fyz))
    col = data[:, m].copy(); ne = 0
    for _ in range(B):
        sh = col.copy(); rng.shuffle(sh); data[:, m] = sh
        v = float(conditional_mi(factors_data=data, x=np.array([m]), y=y, z=z, var_is_nominal=None,
                                 factors_nbins=nbins, entropy_z=hz, entropy_yz=hyz, entropy_cache=None,
                                 can_use_x_cache=False, can_use_y_cache=False))
        if v >= mr: ne += 1
    data[:, m] = col
    return (ne + 1) / (B + 1)


def _parallel(data, nbins, m, y, z, mr, B, seed):
    rng = np.random.default_rng(seed)
    zc, fz, znc = merge_vars(data, z, None, nbins); hz = float(entropy(fz))
    yzc, fyz, yznc = merge_vars(data, unpack_and_sort(y, z), None, nbins); hyz = float(entropy(fyz))
    n = data.shape[0]; col = data[:, m].astype(np.int64)
    sh = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        s = col.copy(); rng.shuffle(s); sh[b] = s
    ne = K._member_null_cmi_prange(sh, int(nbins[m]), zc.astype(np.int64), int(znc),
                                   yzc.astype(np.int64), int(yznc), hz, hyz, float(mr))
    return (ne + 1) / (B + 1)


def main():
    import numba
    print("numba threads:", numba.get_num_threads())
    for (n, nz, nb, B) in [(30000, 2, 8, 199), (100000, 4, 8, 199), (100000, 2, 8, 199), (300000, 3, 8, 199)]:
        ncols = 20
        rng = np.random.default_rng(3)
        data = rng.integers(0, nb, size=(n, ncols)).astype(np.int32)
        nbins = np.full(ncols, nb, dtype=np.int64)
        m, yi = 0, ncols - 1
        z = np.sort(np.arange(1, 1 + nz, dtype=np.int64)); y = np.array([yi])
        mr = float(conditional_mi(factors_data=data, x=np.array([m]), y=y, z=z, var_is_nominal=None,
                                  factors_nbins=nbins, entropy_cache=None, can_use_x_cache=False,
                                  can_use_y_cache=False))
        _serial(data.copy(), nbins, m, y, z, mr, 4, 1)   # warm
        _parallel(data.copy(), nbins, m, y, z, mr, 10, 1)
        ts = []; tp = []; ps = pp = 0.0
        for _ in range(3):
            t = time.perf_counter(); ps = _serial(data.copy(), nbins, m, y, z, mr, B, 9); ts.append(time.perf_counter() - t)
        for _ in range(3):
            t = time.perf_counter(); pp = _parallel(data.copy(), nbins, m, y, z, mr, B, 9); tp.append(time.perf_counter() - t)
        print(f"n={n:>6} |z|={nz} B={B}: serial={min(ts):6.3f}s parallel={min(tp):6.3f}s "
              f"speedup={min(ts)/min(tp):.2f}x  p_serial={ps:.4f} p_parallel={pp:.4f} equal={ps == pp}")


if __name__ == "__main__":
    main()
