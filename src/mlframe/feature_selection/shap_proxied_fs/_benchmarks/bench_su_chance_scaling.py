"""Scaling of the chance-correction gate (``veto_chance_edges``) with the number of high-cardinality columns.

Worst case is modelled: every pair of the ``f_hc`` ID-like columns is flagged by the plug-in scan (~f_hc*(f_hc-1)/2 rescored edges).
Compares the shipped gate against the frozen pre-optimisation implementation (``su_chance_legacy.py``); decisions must be identical.
Times are wall and process_time (all threads); the njit sections are timed directly (cProfile cannot see inside njit).

Run: python bench_su_chance_scaling.py [--quick] [--n=..] [--fhc=..] [--k=..]  (K is clipped to 0.45n)

Levers and verdicts (see numbers in the table printed by this script; recorded in the commit/PR description):
  SHIPPED  per-column stable counting-sort permutation computed once (prange over columns) and reused by every pair that uses the column as
           the major key; the pair MI then sorts only b-values inside each a-group (O(n log(n/Ka))) instead of an O(n log n) sort of n int64
           keys per MI evaluation (3 per edge). Bit-identical: identical (a,b) key order, identical summands, identical accumulation order.
  SHIPPED  prange work is scheduled per (edge) with O(n) int32 per-thread scratch instead of an int64 n-key alloc per edge.
  ALREADY  early exit for both K > n/2 (no MI at all) and per-column entropies / dense relabel (computed once per column).
  REJECTED sharing permutation nulls across pairs with a common (Ki,Kj,n) bucket: the null is a random draw per edge, so decisions are not
           provably identical (edges near the threshold flip); bench-attempt-rejected, option not added.
  REJECTED skipping the 2-perm rescore via analytic bounds: null SU >= 0 gives only "su < threshold => unlink", which the plug-in scan already
           guarantees never happens (flagged edges have su >= threshold); no analytic upper bound of the null is tight enough to be decision-identical.
  REJECTED capping the number of rescored pairs (fallback unlink): would change decisions for true duplicates (recoded IDs) beyond the cap.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

import numpy as np

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster_su_chance as new_mod

_HERE = pathlib.Path(__file__).parent


def load_legacy():
    spec = importlib.util.spec_from_file_location("su_chance_legacy", _HERE / "su_chance_legacy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["su_chance_legacy"] = mod
    spec.loader.exec_module(mod)
    return mod


def build(n, f_hc, K, seed, f_lo=10):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, K, n).astype(np.int64) for _ in range(f_hc)]
    if f_hc >= 2:
        cols[1] = rng.permutation(K)[cols[0]]  # true recoded duplicate: must stay linked
    for _ in range(f_lo):
        cols.append(rng.integers(0, 10, n).astype(np.int64))
    return cols


def timed(fn, *a, reps=1):
    best_w = best_c = 1e18
    out = None
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        out = fn(*a)
        best_w = min(best_w, time.perf_counter() - w0)
        best_c = min(best_c, time.process_time() - c0)
    return out, best_w, best_c


def _arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return [int(v) for v in a.split("=")[1].split(",")]
    return default


def main(quick=False):
    legacy = load_legacy()
    ns = _arg("n", [50_000] if quick else [50_000, 555_921])
    fhcs = _arg("fhc", [5, 20] if quick else [5, 20, 50, 100])
    ks = _arg("k", [10_000, 50_000, 10**9])
    print(f"{'n':>8} {'f_hc':>5} {'K':>7} {'edges':>6} | {'old wall':>9} {'old cpu':>9} | {'new wall':>9} {'new cpu':>9} | speedup(wall) identical")
    for n in ns:
        for K in sorted({min(k, int(0.45 * n)) for k in ks}):
            for f_hc in fhcs:
                cols = build(n, f_hc, K, 0)
                ii, jj = np.triu_indices(f_hc, 1)
                ei, ej = ii.astype(np.int64), jj.astype(np.int64)
                new_mod.veto_chance_edges(cols[:2] + cols[:0], ei[:1], ej[:1], 0.5)  # warm
                legacy.veto_chance_edges(cols[:2], ei[:1], ej[:1], 0.5)
                ro, wo, co = timed(legacy.veto_chance_edges, cols, ei, ej, 0.5)
                rn, wn, cn = timed(new_mod.veto_chance_edges, cols, ei, ej, 0.5, reps=2)
                same = np.array_equal(ro[0], rn[0]) and np.array_equal(ro[1], rn[1])
                print(f"{n:>8} {f_hc:>5} {K:>7} {ei.size:>6} | {wo:>9.2f} {co:>9.2f} | {wn:>9.2f} {cn:>9.2f} | {wo / wn:>6.2f}x {same}", flush=True)


if __name__ == "__main__":
    main("--quick" in sys.argv)
