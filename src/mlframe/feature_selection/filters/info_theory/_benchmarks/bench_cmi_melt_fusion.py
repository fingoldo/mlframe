"""Profile / A-B bench for the two per-candidate melts inside ``conditional_mi``.

After the (Y,Z)-entropy hoist (``_cmi_cuda.py``), each redundancy candidate still pays TWO melts:
  * H(X,Z)  -- ``merge_vars(unpack_and_sort(x, z))``           (X-on-Z melt)
  * H(X,Y,Z) -- ``merge_vars(x, final_classes=classes_yz, ...)`` (X-on-YZ-classes melt)

Both callers immediately reduce the returned ``freqs`` to a single ``entropy`` scalar and DISCARD the
``final_classes`` relabel array + the lookup-table remap that ``merge_vars`` builds. This bench measures how
much of the per-candidate cost is that discarded work, and A/Bs the fused freqs-only kernels against
``entropy(merge_vars(...)[1])`` for wall speed AND byte-identity.

Run:  python -m mlframe.feature_selection.filters.info_theory._benchmarks.bench_cmi_melt_fusion
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.feature_selection.filters._numba_utils import unpack_and_sort
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars
from mlframe.feature_selection.filters.info_theory._entropy_kernels import (
    entropy,
    _entropy_xz_fused,
    _entropy_x_onto_classes,
)


def _best_of(fn, n_iter=15, reps=30):
    fn()  # warm
    best = 1e18
    for _ in range(n_iter):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t) / reps)
    return best


def bench(n=1_000_000, nbins=16, z_ncols=1, seed=0):
    rng = np.random.default_rng(seed)
    # candidate x, target y, conditioning z (z_ncols columns), all ordinal-encoded
    ncols = 3 + z_ncols
    data = np.empty((n, ncols), dtype=np.int32)
    for c in range(ncols):
        data[:, c] = rng.integers(0, nbins, size=n, dtype=np.int32)
    factors_nbins = np.full(ncols, nbins, dtype=np.int64)
    dtype = np.int32

    x = np.array([0], dtype=np.int64)
    y = np.array([1], dtype=np.int64)
    z = np.arange(2, 2 + z_ncols, dtype=np.int64)

    # ---- melt A: H(X,Z) ----
    xz = unpack_and_sort(x, z)

    def old_xz():
        _, f, _ = merge_vars(data, xz, None, factors_nbins, dtype=dtype)
        return entropy(f)

    def new_xz():
        return _entropy_xz_fused(data, xz, factors_nbins, dtype)

    # ---- melt B: H(X,Y,Z) melting x onto precomputed classes_yz ----
    yz = unpack_and_sort(y, z)
    classes_yz, _, ncls_yz = merge_vars(data, yz, None, factors_nbins, dtype=dtype)

    def old_xyz():
        cyz = classes_yz.copy()
        _, f, _ = merge_vars(
            data, x, None, factors_nbins,
            current_nclasses=ncls_yz, final_classes=cyz, dtype=dtype,
        )
        return entropy(f)

    def new_xyz():
        return _entropy_x_onto_classes(data, int(x[0]), classes_yz, ncls_yz, int(factors_nbins[x[0]]))

    # identity
    d_xz = abs(old_xz() - new_xz())
    d_xyz = abs(old_xyz() - new_xyz())

    t_old_xz = _best_of(old_xz)
    t_new_xz = _best_of(new_xz)
    t_old_xyz = _best_of(old_xyz)
    t_new_xyz = _best_of(new_xyz)

    print(f"n={n} nbins={nbins} z_ncols={z_ncols}")
    print(f"  H(X,Z)  old={t_old_xz*1e3:8.3f}ms new={t_new_xz*1e3:8.3f}ms  speedup={t_old_xz/t_new_xz:5.2f}x  maxabsdiff={d_xz:.3e}")
    print(f"  H(XYZ)  old={t_old_xyz*1e3:8.3f}ms new={t_new_xyz*1e3:8.3f}ms  speedup={t_old_xyz/t_new_xyz:5.2f}x  maxabsdiff={d_xyz:.3e}")
    tot_old = t_old_xz + t_old_xyz
    tot_new = t_new_xz + t_new_xyz
    print(f"  per-candidate two-melt total: old={tot_old*1e3:8.3f}ms new={tot_new*1e3:8.3f}ms speedup={tot_old/tot_new:5.2f}x")
    return d_xz, d_xyz


if __name__ == "__main__":
    for nbins in (10, 16, 32):
        for z_ncols in (1, 2):
            bench(n=1_000_000, nbins=nbins, z_ncols=z_ncols)
    # smaller representative
    bench(n=100_000, nbins=16, z_ncols=1)
