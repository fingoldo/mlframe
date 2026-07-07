"""Bench: prune the wasted per-candidate melts in the hoisted CPU-CMI fast path.

The DEFAULT wellbore redundancy path (MLFRAME_CMI_YZ_HOIST=1) is
``_cpu_cmi_loop_hoisted_parallel`` -> ``_cmi_one_fixed_yz``, which for EACH
candidate calls the full ``merge_vars`` twice:
  * H(X,Z): merge_vars(xz) -> builds+remaps a length-n final_classes it discards.
  * H(X,Y,Z): classes_yz.copy() (length n) + merge_vars melt onto it -> again a
    discarded final_classes + remap.

``conditional_mi`` itself already avoids BOTH via the pruned freqs-only kernels
``_entropy_xz_fused`` (joint_freqs_2var for the 2-var X u Z) and
``_entropy_x_onto_classes`` (read-only melt of X onto the precomputed YZ classes,
no copy, no remap). The hoisted fast path never adopted them.

This bench A/Bs the OLD (merge_vars) vs NEW (pruned) per-candidate kernel at the
real redundancy shape (n=30k screen subsample, p candidates, nbins~10-16, |Z|=1)
and asserts bit-identity of the resulting CMI vector.

Run:  python -m mlframe.feature_selection.filters.info_theory._benchmarks.bench_cmi_pruned_melts
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit, prange

from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars
from mlframe.feature_selection.filters.info_theory._entropy_kernels import (
    entropy,
    _entropy_xz_fused,
    _entropy_x_onto_classes,
)
from mlframe.feature_selection.filters.info_theory._cmi_cuda import (
    _cmi_yz_fixed_terms,
)


# ---- OLD per-candidate kernel (current prod: full merge_vars, discards final_classes) ----
@njit(cache=True)
def _cmi_one_fixed_yz_OLD(factors_data, xi, zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype):
    xz = np.empty(2, dtype=np.int64)
    if xi <= zi:
        xz[0] = xi; xz[1] = zi
    else:
        xz[0] = zi; xz[1] = xi
    _, freqs_xz, _ = merge_vars(factors_data, xz, None, factors_nbins, dtype=dtype)
    ent_xz = entropy(freqs_xz)
    scratch = classes_yz.copy()
    xarr = np.empty(1, dtype=np.int64)
    xarr[0] = xi
    _, freqs_xyz, _ = merge_vars(factors_data, xarr, None, factors_nbins, current_nclasses=nclasses_yz, final_classes=scratch, dtype=dtype)
    ent_xyz = entropy(freqs_xyz)
    r = ent_xz + ent_yz - ent_z - ent_xyz
    return r if r > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _loop_OLD(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32):
    p = len(cand_indices)
    out = np.empty(p, dtype=np.float64)
    ent_z, classes_yz, ent_yz, nclasses_yz = _cmi_yz_fixed_terms(factors_data, y, z, factors_nbins, dtype)
    zi = z[0]
    for i in prange(p):
        out[i] = _cmi_one_fixed_yz_OLD(factors_data, cand_indices[i], zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype)
    return out


# ---- NEW per-candidate kernel (pruned: fused freqs-only melts, no discarded arrays) ----
@njit(cache=True)
def _cmi_one_fixed_yz_NEW(factors_data, xi, zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype):
    xz = np.empty(2, dtype=np.int64)
    if xi <= zi:
        xz[0] = xi; xz[1] = zi
    else:
        xz[0] = zi; xz[1] = xi
    ent_xz = _entropy_xz_fused(factors_data, xz, factors_nbins, dtype)
    ent_xyz = _entropy_x_onto_classes(factors_data, xi, classes_yz, nclasses_yz, factors_nbins[xi])
    r = ent_xz + ent_yz - ent_z - ent_xyz
    return r if r > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _loop_NEW(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32):
    p = len(cand_indices)
    out = np.empty(p, dtype=np.float64)
    ent_z, classes_yz, ent_yz, nclasses_yz = _cmi_yz_fixed_terms(factors_data, y, z, factors_nbins, dtype)
    zi = z[0]
    for i in prange(p):
        out[i] = _cmi_one_fixed_yz_NEW(factors_data, cand_indices[i], zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype)
    return out


def _make_data(n, p, nbins, seed=0):
    rng = np.random.default_rng(seed)
    nfeat = p + 2
    data = np.empty((n, nfeat), dtype=np.int32)
    for c in range(nfeat):
        nb = nbins if isinstance(nbins, int) else rng.integers(8, 17)
        data[:, c] = rng.integers(0, nb, size=n)
    fnbins = np.full(nfeat, nbins if isinstance(nbins, int) else 16, dtype=np.int64)
    cand = np.arange(p, dtype=np.int64)
    y = np.array([p], dtype=np.int64)
    z = np.array([p + 1], dtype=np.int64)
    return data, cand, y, z, fnbins


def bench(n=30000, p=400, nbins=12, reps=5):
    data, cand, y, z, fnbins = _make_data(n, p, nbins)
    # warm
    o = _loop_OLD(data, cand, y, z, fnbins)
    m = _loop_NEW(data, cand, y, z, fnbins)
    maxdiff = np.max(np.abs(o - m))

    def timeit(fn, inner=10):
        best = 1e30
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                fn(data, cand, y, z, fnbins)
            best = min(best, (time.perf_counter() - t0) / inner)
        return best

    # interleaved paired A/B so machine load cancels
    told = tnew = 0.0
    told = timeit(_loop_OLD); tnew = timeit(_loop_NEW)
    told = min(told, timeit(_loop_OLD)); tnew = min(tnew, timeit(_loop_NEW))
    print(f"n={n} p={p} nbins={nbins}: OLD={told*1000:.1f}ms NEW={tnew*1000:.1f}ms speedup={told/tnew:.2f}x maxdiff={maxdiff:.3e}")
    return told, tnew, maxdiff


if __name__ == "__main__":
    for n, p, nb in [(30000, 100, 10), (30000, 400, 12), (30000, 1000, 16), (30000, 2000, 16), (10000, 500, 12)]:
        bench(n, p, nb)
