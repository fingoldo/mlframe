"""Bench: fused single-pass per-column min/max vs two-pass np.nanmin + np.nanmax for compute_naive_outlier_score.

Baseline mins=np.nanmin(X,axis=0); maxs=np.nanmax(X,axis=0) walks the (N,d) array twice (two full memory sweeps, serial C reductions).
A fused njit(parallel=True) kernel computes both bounds in ONE pass with prange over row chunks, halving memory traffic and adding multicore.
Run: python -m mlframe.preprocessing._benchmarks.bench_naive_outlier_minmax
"""
from __future__ import annotations
import sys, time
sys.modules.setdefault("cupy", None)
import numpy as np
from mlframe.preprocessing.outliers import _nanminmax_cols


def main():
    rng = np.random.default_rng(0)
    for d in (4, 8, 30):
        X = rng.standard_normal((10_000_000, d)).astype(np.float64)
        X[rng.integers(0, X.shape[0], 1000), 0] = np.nan
        mn, mx = _nanminmax_cols(X)  # warm
        assert np.array_equal(mn, np.nanmin(X, axis=0)), "min mismatch"  # nosec B101 - internal invariant check in src/mlframe/preprocessing/_benchmarks, not reachable with untrusted input
        assert np.array_equal(mx, np.nanmax(X, axis=0)), "max mismatch"  # nosec B101 - internal invariant check in src/mlframe/preprocessing/_benchmarks, not reachable with untrusted input
        old, new = [], []
        for _ in range(7):
            t = time.perf_counter(); np.nanmin(X, axis=0); np.nanmax(X, axis=0); old.append(time.perf_counter() - t)
            t = time.perf_counter(); _nanminmax_cols(X); new.append(time.perf_counter() - t)
        om, nm = min(old) * 1000, min(new) * 1000
        print(f"d={d:2d}: two-pass {om:7.1f}ms  fused {nm:7.1f}ms  speedup {om / nm:.2f}x")


if __name__ == "__main__":
    main()
