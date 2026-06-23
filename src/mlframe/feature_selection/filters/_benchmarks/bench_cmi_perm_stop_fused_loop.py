"""Bench: fused within-stratum CMI permutation loop for ``cmi_permutation_stop`` /
``conditional_permutation_test``.

OLD: the Python ``for p in range(B)`` loop copies X, shuffles each stratum with
``rng.permutation``, then calls ``_cmi_plugin_njit`` -- which rebuilds the full
``(K_x,K_y,K_z)`` joint AND recomputes the ``Pz`` / ``Pyz`` marginals on every
permutation. But Y and Z are FIXED across all B permutations (only X is permuted
within Z-strata), so ``Pz`` and ``Pyz`` never change -- recomputing them B times
is wasted work, on top of B numpy copies + B*|strata| Python-level shuffles.

Two NEW variants benched:

* ``new_null_dist`` (v1) -- BIT-IDENTICAL: keeps OLD's exact ``rng.permutation``
  draws (builds the permuted-X matrix in Python with the identical rng), hoists
  ``Pz`` / ``Pyz`` out of the per-permutation loop, runs histogram+CMI in one njit
  kernel. Bit-identical to OLD (``max_abs=0``) but MEASURED 0.93-0.97x -- a LOSS:
  the marginal-hoist saving (one O(n) pass of ~three) is dwarfed by materialising
  the ``(B,n)`` int64 perm matrix; the dominant per-permutation O(n) joint rebuild
  is unavoidable (X changes every permutation) and already njit in OLD.

* ``new2_null_dist`` (v2) -- fully fused in njit (in-kernel Fisher-Yates from a
  precomputed uniform buffer, no perm matrix, ``Pxz`` folded from ``joint``).
  MEASURED 2.5-2.9x faster and the null-distribution MEANS match OLD to ~1e-4, but
  it consumes a DIFFERENT RNG stream than OLD's ``rng.permutation`` so the null is
  NOT bit-identical; per-input p-values diverge up to ~0.13 near the boundary
  (0 decision flips over 200 trials, but the boundary risk is real). A near-alpha
  decision could flip on some input -> selection-altering -> not shippable as the
  default per CLAUDE.md "Gate a big win on its safe condition".

VERDICT: REJECTED. v1 is bit-identical but slower; v2 is 2.5x faster but not
bit-identical and the divergence can move a near-alpha significance decision, with
no cheap predicate to gate the safe case. Kept here per "REJECTED != DELETED" so
a future agent on different HW / with a bit-exact in-njit RNG (e.g. a numba PCG64
matching numpy's stream) can revisit -- that would unlock the 2.5x bit-identically.

Run: CUDA_VISIBLE_DEVICES="" python bench_cmi_perm_stop_fused_loop.py        # v1 (bit-identical, LOSS)
     CUDA_VISIBLE_DEVICES="" python bench_cmi_perm_stop_fused_loop.py --v2   # v2 (2.5x, RNG-divergent)
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from mlframe.feature_selection.filters._cmi_perm_stop import _cmi_plugin_njit


# ----- OLD path (verbatim inner loop from cmi_permutation_stop) -----
def old_null_dist(x_int, y_int, z_comp, K_x, K_y, K_z, n_permutations, seed):
    rng = np.random.default_rng(int(seed))
    strata = {}
    for zv in np.unique(z_comp):
        strata[int(zv)] = np.flatnonzero(z_comp == zv)
    null_dist = np.empty(int(n_permutations), dtype=np.float64)
    for p in range(int(n_permutations)):
        x_perm = x_int.copy()
        for idx_arr in strata.values():
            if idx_arr.size <= 1:
                continue
            x_perm[idx_arr] = x_int[rng.permutation(idx_arr)]
        null_dist[p] = _cmi_plugin_njit(x_perm, y_int, z_comp, K_x, K_y, K_z)
    return null_dist


# ----- NEW fused kernel -----
@njit(nogil=True, cache=True)
def _cmi_perm_loop_njit(x_int, y_int, z_comp, perm_x_mat, K_x, K_y, K_z):
    """perm_x_mat[p] is the already-permuted X for permutation p (built in Python
    with the matched rng). Pz / Pyz computed once; joint+Pxz per permutation."""
    n = x_int.shape[0]
    B = perm_x_mat.shape[0]
    Pz = np.zeros(K_z, dtype=np.float64)
    Pyz = np.zeros((K_y, K_z), dtype=np.float64)
    for i in range(n):
        zc = z_comp[i]
        Pz[zc] += 1.0
        Pyz[y_int[i], zc] += 1.0
    n_f = float(n)
    out = np.empty(B, dtype=np.float64)
    joint = np.empty((K_x, K_y, K_z), dtype=np.float64)
    Pxz = np.empty((K_x, K_z), dtype=np.float64)
    for p in range(B):
        joint[:, :, :] = 0.0
        Pxz[:, :] = 0.0
        xp = perm_x_mat[p]
        for i in range(n):
            xi = xp[i]; yi = y_int[i]; zi = z_comp[i]
            joint[xi, yi, zi] += 1.0
            Pxz[xi, zi] += 1.0
        cmi = 0.0
        for i in range(K_x):
            for j in range(K_y):
                for k in range(K_z):
                    v = joint[i, j, k]
                    if v <= 0.0 or Pxz[i, k] <= 0.0 or Pyz[j, k] <= 0.0 or Pz[k] <= 0.0:
                        continue
                    p_xyz = v / n_f
                    p_z = Pz[k] / n_f
                    p_xz = Pxz[i, k] / n_f
                    p_yz = Pyz[j, k] / n_f
                    cmi += p_xyz * np.log((p_xyz * p_z) / (p_xz * p_yz))
        out[p] = max(0.0, cmi)
    return out


def new_null_dist(x_int, y_int, z_comp, K_x, K_y, K_z, n_permutations, seed):
    rng = np.random.default_rng(int(seed))
    strata = {}
    for zv in np.unique(z_comp):
        strata[int(zv)] = np.flatnonzero(z_comp == zv)
    B = int(n_permutations)
    perm_x_mat = np.empty((B, x_int.shape[0]), dtype=np.int64)
    for p in range(B):
        x_perm = x_int.copy()
        for idx_arr in strata.values():
            if idx_arr.size <= 1:
                continue
            x_perm[idx_arr] = x_int[rng.permutation(idx_arr)]
        perm_x_mat[p] = x_perm
    return _cmi_perm_loop_njit(x_int, y_int, z_comp, perm_x_mat, K_x, K_y, K_z)


def make_data(n, K_x, K_y, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, K_x, n).astype(np.int64)
    y = rng.integers(0, K_y, n).astype(np.int64)
    s1 = rng.integers(0, 5, n).astype(np.int64)
    s2 = rng.integers(0, 4, n).astype(np.int64)
    z = s1 * 4 + s2
    return x, y, z, 20


def bench():
    for n in (5000, 20000, 100000):
        K_x = K_y = 8
        x, y, z, K_z = make_data(n, K_x, K_y)
        B = 100
        # warm
        old_null_dist(x, y, z, K_x, K_y, K_z, 3, 1)
        new_null_dist(x, y, z, K_x, K_y, K_z, 3, 1)
        # identity
        a = old_null_dist(x, y, z, K_x, K_y, K_z, B, 7)
        b = new_null_dist(x, y, z, K_x, K_y, K_z, B, 7)
        max_abs = float(np.max(np.abs(a - b)))
        bit_id = bool(np.array_equal(a, b))
        # timing best-of-5
        def t(fn):
            best = 1e9
            for _ in range(5):
                s = time.perf_counter()
                fn(x, y, z, K_x, K_y, K_z, B, 1)
                best = min(best, time.perf_counter() - s)
            return best
        to = t(old_null_dist); tn = t(new_null_dist)
        print(f"n={n:>7} B={B}: OLD={to*1e3:8.2f}ms NEW={tn*1e3:8.2f}ms "
              f"speedup={to/tn:5.2f}x  bit_identical={bit_id} max_abs={max_abs:.2e}")


# ----- NEW2: shuffle + reduce fully inside njit, no perm matrix, Pxz folded from joint -----
@njit(nogil=True, cache=True)
def _cmi_perm_loop_v2(x_int, y_int, z_comp, strata_idx, strata_ptr, rand_u, B, K_x, K_y, K_z):
    n = x_int.shape[0]
    n_strata = strata_ptr.shape[0] - 1
    Pz = np.zeros(K_z, dtype=np.float64)
    Pyz = np.zeros((K_y, K_z), dtype=np.float64)
    for i in range(n):
        zc = z_comp[i]
        Pz[zc] += 1.0
        Pyz[y_int[i], zc] += 1.0
    n_f = float(n)
    out = np.empty(B, dtype=np.float64)
    joint = np.empty((K_x, K_y, K_z), dtype=np.float64)
    xperm = x_int.copy()
    ridx = 0
    for p in range(B):
        # Fisher-Yates within each stratum using precomputed uniforms.
        for s in range(n_strata):
            a = strata_ptr[s]; b = strata_ptr[s + 1]
            m = b - a
            if m <= 1:
                continue
            for t2 in range(m - 1, 0, -1):
                j = int(rand_u[ridx] * (t2 + 1)); ridx += 1
                if j > t2:
                    j = t2
                ia = strata_idx[a + t2]; ib = strata_idx[a + j]
                tmp = xperm[ia]; xperm[ia] = xperm[ib]; xperm[ib] = tmp
        joint[:, :, :] = 0.0
        for i in range(n):
            joint[xperm[i], y_int[i], z_comp[i]] += 1.0
        cmi = 0.0
        for k in range(K_z):
            pz = Pz[k]
            if pz <= 0.0:
                continue
            for i in range(K_x):
                pxz = 0.0
                for j in range(K_y):
                    pxz += joint[i, j, k]
                if pxz <= 0.0:
                    continue
                for j in range(K_y):
                    v = joint[i, j, k]
                    if v <= 0.0 or Pyz[j, k] <= 0.0:
                        continue
                    p_xyz = v / n_f
                    cmi += p_xyz * np.log((p_xyz * (pz / n_f)) / ((pxz / n_f) * (Pyz[j, k] / n_f)))
        out[p] = max(0.0, cmi)
    return out


def new2_null_dist(x_int, y_int, z_comp, K_x, K_y, K_z, n_permutations, seed):
    rng = np.random.default_rng(int(seed))
    order = np.argsort(z_comp, kind="stable")
    zs = z_comp[order]
    uniq, starts = np.unique(zs, return_index=True)
    strata_ptr = np.append(starts, len(zs)).astype(np.int64)
    strata_idx = order.astype(np.int64)
    B = int(n_permutations)
    # upper bound on random draws: sum(m-1) per perm <= n per perm
    rand_u = rng.random(B * x_int.shape[0]).astype(np.float64)
    return _cmi_perm_loop_v2(x_int, y_int, z_comp, strata_idx, strata_ptr, rand_u, B, K_x, K_y, K_z)


def bench2():
    for n in (5000, 20000, 100000):
        K_x = K_y = 8
        x, y, z, K_z = make_data(n, K_x, K_y)
        B = 100
        old_null_dist(x, y, z, K_x, K_y, K_z, 3, 1)
        new2_null_dist(x, y, z, K_x, K_y, K_z, 3, 1)
        def t(fn):
            best = 1e9
            for _ in range(5):
                s = time.perf_counter(); fn(x, y, z, K_x, K_y, K_z, B, 1); best = min(best, time.perf_counter() - s)
            return best
        to = t(old_null_dist); tn = t(new2_null_dist)
        # v2 won't be bit-identical (different RNG scheme) -> check distribution stats instead
        a = old_null_dist(x, y, z, K_x, K_y, K_z, 500, 7)
        b = new2_null_dist(x, y, z, K_x, K_y, K_z, 500, 7)
        print(f"V2 n={n:>7}: OLD={to*1e3:8.2f}ms NEW2={tn*1e3:8.2f}ms speedup={to/tn:5.2f}x  "
              f"mean OLD={a.mean():.5f} NEW2={b.mean():.5f}")


if __name__ == "__main__":
    import sys as _s
    if "--v2" in _s.argv:
        bench2()
    else:
        bench()
