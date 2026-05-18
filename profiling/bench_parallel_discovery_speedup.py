"""HIGH#6 2026-05-18: measure ``discovery_n_jobs`` parallel speedup.

The T1#1 commit shipped real parallel composite-candidate evaluation and
the README claimed "5-10x speedup via joblib threading". That was
inherited from the original placeholder docstring and never measured.

This benchmark times CompositeTargetDiscovery.fit at n_jobs=1, 2, 4, 8
on a representative workload (3 bases x 10 transforms x synthetic
n=8000 rows) and reports the actual speedup. The README claim is
qualified to match the measurement.

Run: python profiling/bench_parallel_discovery_speedup.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd


def _build_problem(n: int = 200_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    base1 = rng.normal(50.0, 10.0, n)
    base2 = rng.normal(20.0, 5.0, n)
    base3 = rng.exponential(3.0, n)
    cols = {f"f{i}": rng.normal(size=n) for i in range(6)}
    y = (
        1.5 * base1 + 0.8 * base2 + 0.3 * np.log1p(base3)
        + 0.2 * cols["f0"] - 0.1 * cols["f1"]
        + rng.normal(0.0, 1.0, n)
    )
    cols["base1"] = base1
    cols["base2"] = base2
    cols["base3"] = base3
    cols["y"] = y
    return pd.DataFrame(cols), y


def _time(n_jobs: int, transforms: list[str]) -> float:
    from mlframe.training.composite_discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    df, y = _build_problem()
    n = len(df)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        mi_sample_n=50_000,
        composite_skip_when_raw_dominates_ratio=0.0,
        transforms=transforms,
        mi_nbins=8, mi_estimator="bin",
        top_k_after_mi=4,
        eps_mi_gain=-1.0,
        random_state=42,
        discovery_n_jobs=n_jobs,
        mi_gain_bootstrap_n=0,
        detect_linear_residual_alpha_drift=False,
        base_candidates=["base1", "base2", "base3"],
    )
    disc = CompositeTargetDiscovery(config=cfg)
    feature_cols = [c for c in df.columns if c != "y"]
    train_idx = np.arange(int(0.8 * n))
    t0 = time.perf_counter()
    disc.fit(df=df, target_col="y", feature_cols=feature_cols, train_idx=train_idx)
    return time.perf_counter() - t0


def main() -> int:
    transforms = [
        "linear_residual", "diff", "ratio", "logratio",
        "monotonic_residual", "quantile_residual",
        "ewma_residual", "cbrt_y", "log_y", "yeo_johnson_y",
    ]
    print("=" * 70)
    print("HIGH#6 ``discovery_n_jobs`` parallel speedup measurement")
    print("=" * 70)
    print(f"Problem: 3 bases x {len(transforms)} transforms on n=200_000")
    print(f"CPU count: {os.cpu_count()}")
    print()

    # Warm up (JIT, imports).
    _time(n_jobs=1, transforms=transforms)

    n_runs = 3
    results = {}
    for n_jobs in (1, 2, 4, 8):
        times = [_time(n_jobs=n_jobs, transforms=transforms) for _ in range(n_runs)]
        results[n_jobs] = float(np.median(times))
        print(f"  n_jobs={n_jobs}: median {results[n_jobs]:.3f}s  (runs: {times})")

    print()
    serial = results[1]
    print("Speedup vs n_jobs=1:")
    for n_jobs in (2, 4, 8):
        speedup = serial / max(results[n_jobs], 1e-9)
        print(f"  n_jobs={n_jobs}: {speedup:.2f}x")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
