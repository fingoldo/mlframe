"""A/B bench for two conformal kernels (iter118 @10M).

1. ``_fit_sigma_model`` per-bin grouped pass.
   OLD: ``for b in range(nb): m = idx==b; if m.any(): sigma[b]=ar[m].mean()`` --
   nb full-array masked sweeps. NEW: two np.bincount passes (sum + count) over the
   bin index, one O(n) grouped pass. ISOLATED 2.2x @10M; identity ~1e-14 (reduction
   order on a sigma width-scale), empty bins keep the global-mean default. e2e on
   _fit_sigma_model itself is NEUTRAL (np.quantile sort dominates) -- kept as a clean
   isolated win + cleaner code, not the headline.

2. ``predict_interval_mondrian`` per-row radius lookup (the HEADLINE e2e win).
   OLD: ``for i in range(n): radii[i] = per_group.get(g[i], global_r)`` -- pure-Python
   per-row loop with a dict lookup each row. NEW: pd.factorize(sort=False,
   use_na_sentinel=False) maps rows to unique-label codes (hash-based, C-level), dict
   lookup runs once per UNIQUE label, single code-gather assigns radii. 2.9x @10M,
   radii + missing-set BIT-IDENTICAL incl. NaN labels. (np.unique was tried: 4x SLOWER
   on object labels due to object sort -- factorize's hashing wins.)

Run:
  python src/mlframe/training/composite/_benchmarks/bench_conformal_sigma_bincount.py
"""
from __future__ import annotations

import sys
sys.modules.setdefault("cupy", None)  # avoid py3.14 cold-import segfault under contention

import time
import numpy as np
import pandas as pd

_NB = 20


def _mondrian_old(g, per_group, global_r, n):
    radii = np.empty(n, dtype=np.float64)
    missing = set()
    for i in range(n):
        lab = g[i]
        if lab in per_group:
            radii[i] = per_group[lab]
        else:
            radii[i] = global_r
            missing.add(lab)
    return radii, missing


def _mondrian_new(g, per_group, global_r, n):
    codes, uniq = pd.factorize(g, sort=False, use_na_sentinel=False)
    rpu = np.empty(uniq.shape[0], dtype=np.float64)
    missing = set()
    for j, lab in enumerate(uniq):
        if lab in per_group:
            rpu[j] = per_group[lab]
        else:
            rpu[j] = global_r
            missing.add(lab)
    return rpu[codes], missing


def _sigma_old(yp, ar, edges, nb):
    floor = max(1e-9, 0.05 * float(ar.mean()))
    sigma = np.full(nb, ar.mean() if ar.size else 1.0)
    idx = np.clip(np.searchsorted(edges, yp, side="right") - 1, 0, nb - 1)
    for b in range(nb):
        m = idx == b
        if m.any():
            sigma[b] = ar[m].mean()
    return np.maximum(sigma, floor)


def _sigma_new(yp, ar, edges, nb):
    floor = max(1e-9, 0.05 * float(ar.mean()))
    sigma = np.full(nb, ar.mean() if ar.size else 1.0)
    idx = np.clip(np.searchsorted(edges, yp, side="right") - 1, 0, nb - 1)
    counts = np.bincount(idx, minlength=nb)
    sums = np.bincount(idx, weights=ar, minlength=nb)
    nonempty = counts > 0
    sigma[nonempty] = sums[nonempty] / counts[nonempty]
    return np.maximum(sigma, floor)


def _make(n, seed=0):
    rng = np.random.default_rng(seed)
    yp = rng.normal(0, 1, n)
    ar = np.abs(rng.normal(0, 1 + 0.5 * np.abs(yp), n))  # heteroscedastic
    edges = np.quantile(yp, np.linspace(0.0, 1.0, _NB + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    nb = edges.size - 1
    return yp, ar, edges, nb


def _timeit(fn, args, reps=5):
    fn(*args)  # warm
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    for n in (1_000_000, 10_000_000):
        yp, ar, edges, nb = _make(n)
        s_old = _sigma_old(yp, ar, edges, nb)
        s_new = _sigma_new(yp, ar, edges, nb)
        max_abs = float(np.max(np.abs(s_old - s_new)))
        max_rel = float(np.max(np.abs(s_old - s_new) / np.maximum(np.abs(s_old), 1e-30)))
        t_old = _timeit(_sigma_old, (yp, ar, edges, nb))
        t_new = _timeit(_sigma_new, (yp, ar, edges, nb))
        print(f"sigma   n={n:>11,d} nb={nb}  OLD={t_old*1e3:8.3f}ms  NEW={t_new*1e3:8.3f}ms  "
              f"speedup={t_old/t_new:5.2f}x  max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

    for n in (1_000_000, 10_000_000):
        rng = np.random.default_rng(2)
        g = rng.integers(0, 50, n).astype(object)
        per_group = {i: float(rng.random()) for i in range(40)}  # 40 known, 10 missing
        global_r = 9.9
        ro, mo = _mondrian_old(g, per_group, global_r, n)
        rn, mn = _mondrian_new(g, per_group, global_r, n)
        ident = bool(np.array_equal(ro, rn)) and (mo == mn)
        t_o = _timeit(_mondrian_old, (g, per_group, global_r, n), reps=3)
        t_n = _timeit(_mondrian_new, (g, per_group, global_r, n), reps=3)
        print(f"mondrian n={n:>11,d}        OLD={t_o*1e3:8.3f}ms  NEW={t_n*1e3:8.3f}ms  "
              f"speedup={t_o/t_n:5.2f}x  identical={ident}")
