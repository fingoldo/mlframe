"""Measure polynom-FE parallel speedup (pair-loop joblib).

The polynom-FE block in MRMR.fit iterates over prospective_pairs and runs
``optimise_hermite_pair`` per pair. Pre-2026-05-18 the outer loop was
SERIAL and ignored ``n_jobs``. Post-fix it dispatches via
joblib.Parallel(backend="threading").

Bench scales the problem to expose the parallelism benefit:
- n=10_000 rows
- 8 numeric features -> 28 prospective pairs (with all-pairs gate open)
- Per pair: fe_smart_polynom_iters=2, fe_smart_polynom_optimization_steps=50
  (~50 Optuna trials per restart, 100 trials per pair)
- Total: 2_800 Optuna trials across 28 pairs

Run: python profiling/bench_polynom_fe_parallel.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def _build_problem(n: int, n_feats: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(n_feats):
        cols[f"x{i}"] = rng.normal(size=n).astype(np.float64)
    # Multiple interaction pairs so polynom-FE has work.
    z = (
        cols["x0"] + cols["x1"]
        + 2.0 * cols["x0"] * cols["x1"]
        + 0.7 * cols["x2"] * cols["x3"]
        + 0.5 * cols["x4"] * cols["x5"]
        + rng.normal(0, 0.3, n)
    )
    y = (z > np.median(z)).astype(np.int64)
    return pd.DataFrame(cols), y


def _run_mrmr(*, X, y, n_jobs: int) -> tuple[float, int]:
    from mlframe.feature_selection.filters.mrmr import MRMR

    t0 = time.perf_counter()
    m = MRMR(
        fe_smart_polynom_iters=2,
        fe_smart_polynom_optimization_steps=50,
        fe_min_polynom_degree=1,
        fe_max_polynom_degree=3,
        fe_max_pair_features=30,
        fe_min_pair_mi=-1.0,
        fe_min_pair_mi_prevalence=0.0,
        fe_min_engineered_mi_prevalence=0.0,
        fe_min_nonzero_confidence=0.0,
        min_nonzero_confidence=0.0,
        n_jobs=n_jobs,
        mrmr_skip_when_prior_was_identity=False,
        verbose=0,
    )
    m.fit(X, y)
    elapsed = time.perf_counter() - t0
    n_hf = len(getattr(m, "_hermite_features_", None) or [])
    return elapsed, n_hf


def main() -> int:
    print("=" * 70)
    print("Polynom-FE parallel speedup (pair-loop joblib threading)")
    print("=" * 70)

    X, y = _build_problem(n=10_000, n_feats=8)
    print(f"Problem: n={len(X)}, n_feats={X.shape[1]} "
          f"(~{X.shape[1] * (X.shape[1]-1) // 2} prospective pairs)")
    print(f"Per pair: 2 Optuna restarts x 50 trials")
    print()

    # Warm up (CMA-ES JIT, numba kernels).
    _run_mrmr(X=X, y=y, n_jobs=1)

    n_runs = 2
    results = {}
    for n_jobs in (1, 2, 4, 8):
        times = []
        n_hf_seen = 0
        for _ in range(n_runs):
            t, n_hf = _run_mrmr(X=X, y=y, n_jobs=n_jobs)
            times.append(t)
            n_hf_seen = n_hf
        results[n_jobs] = float(np.median(times))
        print(f"  n_jobs={n_jobs}: median {results[n_jobs]:.2f}s  "
              f"hermite_features={n_hf_seen}  raw_times={times}")
    print()
    serial = results[1]
    print("Speedup vs n_jobs=1:")
    for n_jobs in (2, 4, 8):
        print(f"  n_jobs={n_jobs}: {serial / max(results[n_jobs], 1e-9):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
