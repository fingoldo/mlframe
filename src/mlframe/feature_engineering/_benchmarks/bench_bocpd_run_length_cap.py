"""Bench: BOCPD run-length cap bounds per-step work on long stable streams.

Uncapped BOCPD grows the run-length vector to ~T -> O(T^2) total. The cap holds it to O(cap).

Run: python -m mlframe.feature_engineering._benchmarks.bench_bocpd_run_length_cap
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_engineering.bayesian import bocpd_features


def bestof(fn, n=3):
    fn()  # warm njit
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    for T in (2000, 8000, 20000):
        x = rng.normal(0.0, 1.0, size=T).astype(np.float64)  # stable -> run length grows toward T
        unc = bestof(lambda: bocpd_features(x, max_run_length=0))
        cap = bestof(lambda: bocpd_features(x, max_run_length=1000))
        print(f"T={T:6d}  uncapped {unc*1e3:9.3f}ms  cap=1000 {cap*1e3:9.3f}ms  ({unc/cap:.2f}x)")


if __name__ == "__main__":
    main()
