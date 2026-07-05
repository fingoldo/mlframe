"""Bench: hoist the fixed Y,Z entropies/classes out of the per-candidate conditional_mi loop.

In an MRMR greedy round the target Y and the just-selected feature Z are FIXED across the whole
candidate pool, yet ``conditional_mi`` rebuilds H(Z), the (Y,Z) joint melt (``classes_yz``) and
H(Y,Z) on EVERY candidate. Its existing ``entropy_z``/``entropy_yz`` params cannot remove that work
because the H(X,Y,Z) term is built by melting X on top of ``classes_yz`` -- so ``classes_yz`` is
needed per candidate and gets recomputed each time.

This bench prototypes ``_cpu_cmi_loop_hoisted``: compute ``classes_yz`` / ``entropy_yz`` / H(Z) ONCE
per round, then in the prange over candidates reuse a per-thread COPY of ``classes_yz`` (cheap O(n)
copy) instead of re-melting (Y,Z) + Z from scratch. Bit-identical by construction (same merge order,
same entropy reduction). A/B vs the current ``_cpu_cmi_loop_parallel`` at the wellbore shape.

Run: python -u bench_cmi_yz_hoist.py
"""
from __future__ import annotations

import sys, time
import numpy as np
from numba import njit, prange

sys.path.insert(0, r"C:\Users\Admin\Machine learning\mlframe\src")

from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi, entropy
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars
from mlframe.feature_selection.filters.info_theory._cmi_cuda import _cpu_cmi_loop_parallel


@njit(cache=True)
def _entropy_z_once(factors_data, z, factors_nbins, dtype=np.int32):
    _, freqs_z, _ = merge_vars(factors_data, z, None, factors_nbins, dtype=dtype)
    return entropy(freqs_z)


@njit(cache=True)
def _yz_once(factors_data, y, z, factors_nbins, dtype=np.int32):
    """Melt (y,z) in unpack_and_sort order -> (classes_yz, entropy_yz, nclasses_yz)."""
    idx = np.empty(2, dtype=np.int64)
    a = z[0]; b = y[0]
    if a <= b:
        idx[0] = a; idx[1] = b
    else:
        idx[0] = b; idx[1] = a
    classes_yz, freqs_yz, nclasses_yz = merge_vars(factors_data, idx, None, factors_nbins, dtype=dtype)
    return classes_yz, entropy(freqs_yz), nclasses_yz


@njit(cache=True)
def _cmi_one_hoisted(factors_data, xi, zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype=np.int32):
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


@njit(cache=True)
def _cpu_cmi_loop_hoisted_serial(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32):
    p = len(cand_indices)
    out = np.empty(p, dtype=np.float64)
    ent_z = _entropy_z_once(factors_data, z, factors_nbins, dtype)
    classes_yz, ent_yz, nclasses_yz = _yz_once(factors_data, y, z, factors_nbins, dtype)
    zi = z[0]
    for i in range(p):
        out[i] = _cmi_one_hoisted(factors_data, cand_indices[i], zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype)
    return out


@njit(parallel=True, cache=True)
def _cpu_cmi_loop_hoisted(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32):
    p = len(cand_indices)
    out = np.empty(p, dtype=np.float64)
    ent_z = _entropy_z_once(factors_data, z, factors_nbins, dtype)
    classes_yz, ent_yz, nclasses_yz = _yz_once(factors_data, y, z, factors_nbins, dtype)
    zi = z[0]
    for i in prange(p):
        xi = cand_indices[i]
        # H(X,Z): merge (x,z) in ascending index order (matches unpack_and_sort)
        xz = np.empty(2, dtype=np.int64)
        if xi <= zi:
            xz[0] = xi; xz[1] = zi
        else:
            xz[0] = zi; xz[1] = xi
        _, freqs_xz, _ = merge_vars(factors_data, xz, None, factors_nbins, dtype=dtype)
        ent_xz = entropy(freqs_xz)
        # H(X,Y,Z): melt X on top of a PRIVATE copy of classes_yz (never mutate the shared one)
        scratch = classes_yz.copy()
        xarr = np.empty(1, dtype=np.int64)
        xarr[0] = xi
        _, freqs_xyz, _ = merge_vars(factors_data, xarr, None, factors_nbins, current_nclasses=nclasses_yz, final_classes=scratch, dtype=dtype)
        ent_xyz = entropy(freqs_xyz)
        r = ent_xz + ent_yz - ent_z - ent_xyz
        out[i] = r if r > 0.0 else 0.0
    return out


def bench(n, p, nb, reps=5, serial=False):
    rng = np.random.default_rng(42)
    ncols = p + 2
    fd = rng.integers(0, nb, size=(n, ncols)).astype(np.int32)
    fnb = np.full(ncols, nb, dtype=np.int64)
    cand = np.arange(p, dtype=np.int64)
    y = np.array([p], dtype=np.int64)
    z = np.array([p + 1], dtype=np.int64)
    _vin = np.empty(0, dtype=np.int64)

    def base_fn():
        if serial:
            out = np.empty(p, dtype=np.float64)
            for i in range(p):
                out[i] = conditional_mi(fd, np.array([cand[i]], dtype=np.int64), y, z, _vin, fnb)
            return out
        return _cpu_cmi_loop_parallel(fd, cand, y, z, fnb, _vin)

    hoist_fn = (lambda: _cpu_cmi_loop_hoisted_serial(fd, cand, y, z, fnb)) if serial else (lambda: _cpu_cmi_loop_hoisted(fd, cand, y, z, fnb))

    a = base_fn(); b = hoist_fn()
    maxdiff = float(np.max(np.abs(a - b)))

    def timeit(fn):
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return min(ts), sorted(ts)[len(ts)//2]

    b_min, b_med = timeit(base_fn)
    h_min, h_med = timeit(hoist_fn)
    tag = "SERIAL" if serial else "PARLL "
    print(f"[{tag}] n={n} p={p} nb={nb}: baseline min={b_min*1000:.1f}ms | "
          f"hoisted min={h_min*1000:.1f}ms | speedup={b_min/h_min:.2f}x | maxabsdiff={maxdiff:.2e}")


if __name__ == "__main__":
    # serial regime (p<32) -- the wellbore's actual branch (profile attributes tottime to _cpu_cmi_loop:407)
    for n, p, nb in [(998327, 10, 10), (998327, 20, 10)]:
        bench(n, p, nb, serial=True)
    # parallel regime (p>=32)
    for n, p, nb in [(998327, 100, 10), (998327, 300, 10), (200000, 100, 10), (998327, 64, 16)]:
        bench(n, p, nb)
