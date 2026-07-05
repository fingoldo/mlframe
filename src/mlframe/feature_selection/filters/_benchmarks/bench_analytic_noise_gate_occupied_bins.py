"""Bench: per-column occupied-bin count in ``analytic_batch_noise_gate``.

OLD: ``np.unique(disc_2d[:, k]).size`` per column -> O(K * n log n) sort just to
count distinct occupied bin codes. NEW: one njit pass over the (n, K) matrix
counting occupied bins per column via a per-column presence array -> O(n * K).

Realistic shape from ``test_analytic_mi_null.py`` (n=80k, K=30, low-card codes)
and the FE-pair dispatch (analytic gate engages at n >= 50k). Best-of-N, numba
warmed. Identity: occupied-bin counts are exact == (same chi2 df, same gate).

Run:
    CUDA_VISIBLE_DEVICES="" python bench_analytic_noise_gate_occupied_bins.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))

from mlframe.feature_selection.filters._analytic_mi_null import (  # noqa: E402
    _occupied_bins_per_col,
)


def _old_counts(disc_2d):
    K = disc_2d.shape[1]
    return np.array([int(np.unique(disc_2d[:, k]).size) for k in range(K)], dtype=np.int64)


def _new_counts(disc_2d):
    return _occupied_bins_per_col(disc_2d)


def _best_of(fn, arg, reps=7):
    fn(arg)  # warm
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    for n, K, nbins in [(50_000, 30, 8), (80_000, 30, 8), (200_000, 60, 10)]:
        rng = np.random.default_rng(0)
        disc = rng.integers(0, nbins, size=(n, K)).astype(np.int16)
        old = _old_counts(disc)
        new = _new_counts(disc)
        assert np.array_equal(old, new), f"MISMATCH n={n} K={K}: {old} vs {new}"
        t_old = _best_of(_old_counts, disc)
        t_new = _best_of(_new_counts, disc)
        print(f"n={n:>7} K={K:>3} nbins={nbins:>2} | OLD {t_old*1e3:8.3f} ms | "
              f"NEW {t_new*1e3:8.3f} ms | speedup {t_old/t_new:6.2f}x | identical={np.array_equal(old,new)}")


if __name__ == "__main__":
    main()
