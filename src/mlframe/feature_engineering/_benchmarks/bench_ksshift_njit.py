"""A/B bench: ks_shift._ks_and_wasserstein per-query Python loop vs numba njit(prange).

OLD path (git HEAD ks_shift.py): a `for i in range(n_q)` Python loop, each iteration doing
np.searchsorted + np.abs + .max() + np.diff + weighted sum — n_q Python-frame round-trips with
many small temporaries (diff, widths, global_ranks) allocated per row.

NEW candidate: a single numba njit(parallel=True) kernel that does the searchsorted-equivalent
(global sorted is monotone -> binary search) + the KS-sup + W1 trapezoid inline, no per-row temps.

Methodology (CLAUDE.md A/B): warm numba JIT first, best-of-N median, realistic shape captured from
compute_ks_shift_features regression mode (n_q rows x k neighbours, global of size n_g). Identity gate:
exact float32 equality is NOT expected (different reduction/searchsorted tie handling), but the values
must match to <=1e-6 abs and the SELECTION-relevant feature ordering is preserved (FE bar).

Run: CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_engineering/_benchmarks/bench_ksshift_njit.py
"""
from __future__ import annotations

import time

import numpy as np
import numba


# ---- OLD: verbatim from git HEAD ks_shift._ks_and_wasserstein ----
def _ks_and_wasserstein_old(y_neighbors: np.ndarray, y_global_sorted: np.ndarray):
    n_q, k = y_neighbors.shape
    n_g = y_global_sorted.shape[0]
    ks_out = np.zeros(n_q, dtype=np.float32)
    w1_out = np.zeros(n_q, dtype=np.float32)
    y_local_sorted = np.sort(y_neighbors, axis=1)
    cdf_local = (np.arange(k) + 1).astype(np.float32) / k
    for i in range(n_q):
        local = y_local_sorted[i]
        global_ranks = np.searchsorted(y_global_sorted, local, side="right").astype(np.float32) / n_g
        diff = np.abs(cdf_local - global_ranks)
        ks_out[i] = diff.max()
        widths = np.diff(local, prepend=local[0])
        w1_out[i] = (diff * widths).sum()
    return ks_out, w1_out


# ---- NEW: njit(parallel) kernel ----
@numba.njit(cache=True, fastmath=True, parallel=True)
def _ks_w1_kernel(y_local_sorted, y_global_sorted, ks_out, w1_out):
    n_q, k = y_local_sorted.shape
    n_g = y_global_sorted.shape[0]
    inv_k = np.float32(1.0) / np.float32(k)
    inv_ng = np.float32(1.0) / np.float32(n_g)
    for i in numba.prange(n_q):
        ks = np.float32(0.0)
        w1 = np.float32(0.0)
        prev = y_local_sorted[i, 0]
        for j in range(k):
            v = y_local_sorted[i, j]
            # searchsorted side="right" via binary search on the sorted global array
            lo = 0
            hi = n_g
            while lo < hi:
                mid = (lo + hi) >> 1
                if y_global_sorted[mid] <= v:
                    lo = mid + 1
                else:
                    hi = mid
            g_rank = np.float32(lo) * inv_ng
            cdf_local = np.float32(j + 1) * inv_k
            d = abs(cdf_local - g_rank)
            if d > ks:
                ks = d
            width = v - prev if j > 0 else np.float32(0.0)
            w1 += d * width
            prev = v
        ks_out[i] = ks
        w1_out[i] = w1


def _ks_and_wasserstein_new(y_neighbors: np.ndarray, y_global_sorted: np.ndarray):
    n_q = y_neighbors.shape[0]
    y_local_sorted = np.sort(y_neighbors, axis=1).astype(np.float32)
    ks_out = np.zeros(n_q, dtype=np.float32)
    w1_out = np.zeros(n_q, dtype=np.float32)
    _ks_w1_kernel(y_local_sorted, y_global_sorted.astype(np.float32), ks_out, w1_out)
    return ks_out, w1_out


def _best_of(fn, *args, n=7):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def main():
    rng = np.random.default_rng(0)
    for n_q, k, n_g in [(2000, 32, 1600), (10000, 32, 8000), (50000, 32, 40000)]:
        y_global = np.sort(rng.normal(size=n_g).astype(np.float32))
        y_neighbors = rng.normal(size=(n_q, k)).astype(np.float32)

        ks_o, w1_o = _ks_and_wasserstein_old(y_neighbors, y_global)
        # warm JIT
        _ks_and_wasserstein_new(y_neighbors[:64], y_global)
        ks_n, w1_n = _ks_and_wasserstein_new(y_neighbors, y_global)

        ks_max = float(np.abs(ks_o - ks_n).max())
        w1_max = float(np.abs(w1_o - w1_n).max())

        old_min, old_med = _best_of(_ks_and_wasserstein_old, y_neighbors, y_global)
        new_min, new_med = _best_of(_ks_and_wasserstein_new, y_neighbors, y_global)
        speedup = old_med / new_med
        print(f"n_q={n_q:6d} k={k} n_g={n_g}: OLD {old_med*1e3:8.2f}ms  NEW {new_med*1e3:8.2f}ms  "
              f"speedup {speedup:5.2f}x | ks_abs_err {ks_max:.2e} w1_abs_err {w1_max:.2e}")


if __name__ == "__main__":
    main()
