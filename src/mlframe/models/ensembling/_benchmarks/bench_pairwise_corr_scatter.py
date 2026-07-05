"""Bench: NaN-padded corr-matrix scatter in ``base._pairwise_corr_or_nan``.

The multiclass diversity path scatters a dense (K_use, K_use) correlation submatrix back into
a NaN-padded (original_k, original_k) matrix, once per class. The historical code did this with
an O(K_use^2) Python double loop; the optimization replaces it with a single ``np.ix_`` block
assignment -- bit-identical (a pure index reshuffle, no numerics change), measurably faster as
member count K grows.

Run: ``python -m mlframe.models.ensembling._benchmarks.bench_pairwise_corr_scatter``

Measured (2026-06-23, Python 3.14.3, numpy): K=5 1.52x, K=10 5.17x, K=20 16.73x. The absolute
saving is ~5-185 us/class/ensemble-call -- small but free (the new code is shorter and exactly
bit-identical), so it ships per the "don't dismiss small clean wins" rule.
"""
from __future__ import annotations

import time

import numpy as np


def _old(corr_used: np.ndarray, idx_use: np.ndarray, original_k: int, K_use: int) -> np.ndarray:
    out = np.full((original_k, original_k), np.nan, dtype=np.float64)
    for ii in range(K_use):
        for jj in range(K_use):
            out[idx_use[ii], idx_use[jj]] = corr_used[ii, jj]
    return out


def _new(corr_used: np.ndarray, idx_use: np.ndarray, original_k: int, K_use: int) -> np.ndarray:
    out = np.full((original_k, original_k), np.nan, dtype=np.float64)
    out[np.ix_(idx_use, idx_use)] = corr_used
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    for K in (5, 10, 20):
        corr_used = rng.random((K, K))
        idx_use = np.arange(K)
        a = _old(corr_used, idx_use, K, K)
        b = _new(corr_used, idx_use, K, K)
        assert np.array_equal(a, b, equal_nan=True), f"identity broken at K={K}"
        best_old = best_new = float("inf")
        for _ in range(7):
            t = time.perf_counter()
            for _ in range(2000):
                _old(corr_used, idx_use, K, K)
            best_old = min(best_old, time.perf_counter() - t)
            t = time.perf_counter()
            for _ in range(2000):
                _new(corr_used, idx_use, K, K)
            best_new = min(best_new, time.perf_counter() - t)
        print(f"K={K}: old={best_old * 1e6 / 2000:.2f}us new={best_new * 1e6 / 2000:.2f}us " f"speedup={best_old / best_new:.2f}x (bit-identical)")


if __name__ == "__main__":
    main()
