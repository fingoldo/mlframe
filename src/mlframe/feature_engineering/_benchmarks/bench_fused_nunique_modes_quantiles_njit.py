"""Bench: njit-fused nunique/modes/quantiles core for compute_nunique_modes_quantiles_numpy.

Target: feature_engineering/numerical.py::_fused_nunique_modes_quantiles (the all-finite fast
path used per row by numaggs_over_matrix_rows). The pre-fix body built the sorted-unique
boundaries, counts, mode pick and median_unbiased quantile interp out of a stack of numpy
temporaries (boundary / nonzero / diff / append / clip / floor x2 / astype / minimum), each
paying per-call dispatch + allocation overhead on the small per-row arrays. The fix fuses all of
it into one @njit pass (`_fused_nunique_modes_quantiles_kernel`); np.sort and compute_ncrossings
are unchanged.

Identity: bit-identical by construction. The mode pick replicates np.lexsort((vals, -counts))
exactly (global-max run-length defines the modes, ties broken ascending value, capped at
max_modes; the sorted array yields ascending uniques so the lowest values are kept first). The
quantile interp uses the same Hyndman-Fan type-8 virtual index. Measured 0.0 relative diff across
3000 diverse trials (continuous / low-cardinality / all-identical / rounded-tied).

Measured (python 3.14, numpy 2.3.5, numba 0.63.1, CUDA_VISIBLE_DEVICES=""; best-of-7,
3000 rows per size):
    n=  50  OLD=710.0ms  NEW=329.9ms  2.15x
    n= 200  OLD=775.5ms  NEW=294.2ms  2.64x
    n=1000  OLD=878.9ms  NEW=372.3ms  2.36x

Run:
    CUDA_VISIBLE_DEVICES="" python bench_fused_nunique_modes_quantiles_njit.py
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mlframe.feature_engineering.numerical as N


def old_fused(arr: np.ndarray, q: np.ndarray, max_modes: int) -> tuple:
    """Pre-fix numpy-only reference (the body that lived in _fused_nunique_modes_quantiles)."""
    s = np.sort(arr)
    n = s.size
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    np.not_equal(s[1:], s[:-1], out=boundary[1:])
    idx = np.nonzero(boundary)[0]
    vals = s[idx]
    counts = np.diff(np.append(idx, n))
    mm = min(max_modes, len(counts))
    modes_indices = np.lexsort((vals, -counts))[:mm]
    fmc = counts[modes_indices[0]]
    if fmc == 1:
        modes_min = modes_max = modes_mean = modes_qty = np.nan
    else:
        best = [vals[modes_indices[0]]]
        for i in range(1, mm):
            ni = modes_indices[i]
            if counts[ni] < fmc:
                break
            best.append(vals[ni])
        best = np.asarray(best)
        modes_min, modes_max, modes_mean, modes_qty = best.min(), best.max(), best.mean(), len(best)
    res = (len(vals), modes_min, modes_max, modes_mean, modes_qty)
    h = (n + 1.0 / 3.0) * q + 1.0 / 3.0
    np.clip(h, 1.0, float(n), out=h)
    lo = np.floor(h).astype(np.intp) - 1
    hi = np.minimum(lo + 1, n - 1)
    g = h - np.floor(h)
    quantiles = s[lo] * (1.0 - g) + s[hi] * g
    res = res + tuple(quantiles)
    res = res + tuple(N.compute_ncrossings(arr=arr, marks=quantiles))
    return res


def main() -> None:
    rng = np.random.default_rng(7)
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)

    # Identity sweep.
    N._fused_nunique_modes_quantiles_kernel(np.sort(rng.standard_normal(50)), q, 10)
    max_rel = 0.0
    for trial in range(3000):
        n = int(rng.integers(2, 300))
        kind = trial % 4
        if kind == 0:
            arr = rng.standard_normal(n)
        elif kind == 1:
            arr = rng.integers(0, 5, n).astype(np.float64)
        elif kind == 2:
            arr = np.full(n, rng.standard_normal())
        else:
            arr = np.round(rng.standard_normal(n), 1)
        o = np.array(old_fused(arr, q, 10), dtype=np.float64)
        w = np.array(N._fused_nunique_modes_quantiles(arr, q, "median_unbiased", 10), dtype=np.float64)
        rel = np.abs(o - w) / (np.abs(o) + 1e-300)
        max_rel = max(max_rel, float(np.nanmax(rel)))
    print(f"identity: max relative diff over 3000 trials = {max_rel:.2e}")

    for n in (50, 200, 1000):
        rows = [rng.standard_normal(n) for _ in range(3000)]
        old_fused(rows[0], q, 10)
        N._fused_nunique_modes_quantiles(rows[0], q, "median_unbiased", 10)
        bo = bn = 1e9
        for _ in range(7):
            t = time.perf_counter()
            for r in rows:
                old_fused(r, q, 10)
            bo = min(bo, time.perf_counter() - t)
            t = time.perf_counter()
            for r in rows:
                N._fused_nunique_modes_quantiles(r, q, "median_unbiased", 10)
            bn = min(bn, time.perf_counter() - t)
        print(f"n={n:5d} rows=3000  OLD={bo * 1e3:7.1f}ms  NEW={bn * 1e3:7.1f}ms  speedup={bo / bn:.2f}x")


if __name__ == "__main__":
    main()
