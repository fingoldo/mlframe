"""Isolated microbench: split discretize_2d_quantile_batch wall into the SORT
(``_quantile_edges_2d_njit``) vs the SEARCHSORTED (``_searchsorted_2d_right_njit*``)
phases, on scene-like FE buffer shapes. Authoritative per-kernel signal for Q1
(whether fusing searchsorted into the sort is worth the argsort+scatter cost).

Run with ONE python process at a time (RAM-tight box).
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlframe.feature_selection.filters.discretization import (
    _quantile_edges_2d_njit,
    _searchsorted_2d_right_njit,
    _searchsorted_2d_right_njit_parallel,
    discretize_2d_quantile_batch,
)


def _time(fn, *a, repeats=5):
    fn(*a)  # warm
    best = 1e30
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t)
    return best


def bench(n_rows, n_cols, n_bins=10):
    rng = np.random.default_rng(0)
    arr = np.ascontiguousarray(rng.standard_normal((n_rows, n_cols)).astype(np.float32))
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.empty((quantiles.shape[0], n_cols), dtype=np.float64)

    t_edges = _time(lambda: _quantile_edges_2d_njit(arr, quantiles, edges))
    edges_inner = np.ascontiguousarray(edges[1:-1], dtype=np.float64)
    out = np.empty((n_rows, n_cols), dtype=np.int8)
    t_ss_serial = _time(lambda: _searchsorted_2d_right_njit(edges_inner, arr, out))
    t_ss_par = _time(lambda: _searchsorted_2d_right_njit_parallel(edges_inner, arr, out))

    t_full_serial = _time(lambda: discretize_2d_quantile_batch(arr, n_bins=n_bins, dtype=np.int8, parallel=False))
    t_full_par = _time(lambda: discretize_2d_quantile_batch(arr, n_bins=n_bins, dtype=np.int8, parallel=True))

    print(f"\n=== shape {n_rows}x{n_cols} nbins={n_bins} ===")
    print(f"  edges(sort)            : {t_edges*1e3:8.2f} ms")
    print(f"  searchsorted serial    : {t_ss_serial*1e3:8.2f} ms")
    print(f"  searchsorted parallel  : {t_ss_par*1e3:8.2f} ms")
    print(f"  full batch  serial     : {t_full_serial*1e3:8.2f} ms")
    print(f"  full batch  parallel   : {t_full_par*1e3:8.2f} ms")
    print(f"  -> sort/full(par)      : {100*t_edges/t_full_par:5.1f}%   "
          f"ss(par)/full(par): {100*t_ss_par/t_full_par:5.1f}%")


if __name__ == "__main__":
    # scene FE chunk shapes: n=2407, K spans a few hundred .. several thousand.
    for nc in (300, 1000, 4000, 8000):
        bench(2407, nc)
