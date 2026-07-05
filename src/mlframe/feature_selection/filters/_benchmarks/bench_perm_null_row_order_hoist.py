"""Bench: hoist per-permutation wasted host work out of ``_conditional_perm_null``.

Two bit-identical hoists (row-order invariance of the histogram-plug-in CMI/MI):

1. CONDITIONAL loop -- the OLD code allocated a fresh ``np.empty_like(x)`` and SCATTERED
   ``x_perm[order] = x_sorted[within]`` every permutation, only to hand ``x_perm`` (original
   row order, paired with original-order ``y_i``/``z_i``) to ``cmi_from_binned_fixed_yz``.
   Permuting ``y_i``/``z_i`` by ``order`` ONCE (hoisted) lets the loop pass ``x_sorted[within]``
   (sorted row order) DIRECTLY: the multiset of ``(x,y,z)`` rows is identical, so the joint
   histograms -- hence the CMI -- are bit-identical, and the per-perm ``empty_like`` alloc +
   scatter write are eliminated.

2. MARGINAL loop -- the OLD code called ``_cmi_from_binned(x_perm, y, None)`` every permutation,
   which re-bins ``y`` + recomputes ``H(Y)`` each time. ``precompute_marginal_y_terms`` +
   ``marginal_mi_binned_fixed_y`` (already in the repo) hoist the y-only block once.

Run:  python -m mlframe.feature_selection.filters._benchmarks.bench_perm_null_row_order_hoist
"""
from __future__ import annotations

import time
import numpy as np

from .._mi_greedy_cmi_fe import (
    _cmi_from_binned,
    cmi_from_binned_fixed_yz,
    precompute_cmi_yz_terms,
    precompute_marginal_y_terms,
    marginal_mi_binned_fixed_y,
)

NPERM = 25
QUANTILE = 0.95


def _cond_old(x, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f, order, z_rank, rng):
    x_sorted = x[order]
    nulls = np.empty(NPERM, dtype=np.float64)
    for i in range(NPERM):
        keys = rng.random(x.size)
        within = np.argsort(z_rank + keys, kind="stable")
        x_perm = np.empty_like(x)
        x_perm[order] = x_sorted[within]
        nulls[i] = float(cmi_from_binned_fixed_yz(x_perm, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f))
    return float(np.quantile(nulls, QUANTILE)), float(np.mean(nulls))


def _cond_new(x, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f, order, z_rank, rng):
    x_sorted = x[order]
    y_s = y_i[order]
    z_s = z_i[order]
    nulls = np.empty(NPERM, dtype=np.float64)
    for i in range(NPERM):
        keys = rng.random(x.size)
        within = np.argsort(z_rank + keys, kind="stable")
        nulls[i] = float(cmi_from_binned_fixed_yz(x_sorted[within], y_s, z_s, h_yz, h_z, k_yz, k_z, n_f))
    return float(np.quantile(nulls, QUANTILE)), float(np.mean(nulls))


def _marg_old(x, y, rng):
    nulls = np.empty(NPERM, dtype=np.float64)
    for i in range(NPERM):
        x_perm = x[rng.permutation(x.size)]
        nulls[i] = float(_cmi_from_binned(x_perm, y, None))
    return float(np.quantile(nulls, QUANTILE)), float(np.mean(nulls))


def _marg_new(x, y, rng):
    y_i, h_y, k_y = precompute_marginal_y_terms(y)
    nulls = np.empty(NPERM, dtype=np.float64)
    for i in range(NPERM):
        x_perm = x[rng.permutation(x.size)]
        nulls[i] = float(marginal_mi_binned_fixed_y(x_perm, y_i, h_y, k_y))
    return float(np.quantile(nulls, QUANTILE)), float(np.mean(nulls))


def _make_cond(n, n_strata, kx, ky, seed=1):
    r = np.random.default_rng(seed)
    x = r.integers(0, kx, n).astype(np.int64)
    y = r.integers(0, ky, n).astype(np.int64)
    z = r.integers(0, n_strata, n).astype(np.int64)
    y_i, z_i, h_yz, h_z, k_yz, k_z, n_f = precompute_cmi_yz_terms(y, z)
    order = np.argsort(z_i, kind="stable")
    sorted_z = z_i[order]
    z_rank = np.zeros(n, dtype=np.float64)
    if n > 1:
        z_rank[1:] = np.cumsum(sorted_z[1:] != sorted_z[:-1])
    return x, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f, order, z_rank


def _best_of(fn, args, reps=15):
    fn(*args)  # warm
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        r = fn(*args)
        best = min(best, time.perf_counter() - t)
    return best, r


def main():
    print("=== CONDITIONAL loop (empty_like+scatter removed) ===")
    for n, ns, kx, ky in [(2000, 200, 8, 4), (10000, 1500, 12, 4), (30000, 3000, 16, 6)]:
        base = _make_cond(n, ns, kx, ky)
        seed = 12345
        to, ro = _best_of(lambda: _cond_old(*base, np.random.default_rng(seed)), ())
        tn, rn = _best_of(lambda: _cond_new(*base, np.random.default_rng(seed)), ())
        ident = ro == rn
        print(f"n={n:>6} strata={ns:>5}: old={to*1e3:7.2f}ms new={tn*1e3:7.2f}ms " f"speedup={to/tn:5.2f}x identical={ident}  old={ro} new={rn}")

    print("=== MARGINAL loop (H(Y) hoisted) ===")
    for n, kx, ky in [(2000, 8, 4), (10000, 16, 6), (30000, 24, 8)]:
        r = np.random.default_rng(7)
        x = r.integers(0, kx, n).astype(np.int64)
        y = r.integers(0, ky, n).astype(np.int64)
        seed = 999
        to, ro = _best_of(lambda: _marg_old(x, y, np.random.default_rng(seed)), ())
        tn, rn = _best_of(lambda: _marg_new(x, y, np.random.default_rng(seed)), ())
        ident = ro == rn
        print(f"n={n:>6} kx={kx:>3}: old={to*1e3:7.2f}ms new={tn*1e3:7.2f}ms " f"speedup={to/tn:5.2f}x identical={ident}  old={ro} new={rn}")


if __name__ == "__main__":
    main()
