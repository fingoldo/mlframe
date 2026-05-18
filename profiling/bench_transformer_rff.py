"""Bench harness for ``compute_rff_features`` - measures CPU njit vs GPU cupy across a sweep of (N, d, n_features).

Methodology (per Hardware #30 critique):
- 3 warmup iterations discarded so we measure steady-state, not JIT compile + cupy first-call overhead.
- 7 measured iterations; report median + IQR (interquartile range).
- ``cp.cuda.Stream.null.synchronize()`` before each timestamp on the GPU path so we measure real wall clock, not async dispatch.
- cProfile attribution overhead is NOT measured here - the numba bodies are invisible to cProfile (per the project's numba-coverage-blind reference); for hotspot
  attribution inside the kernels use py-spy in a separate run.

Outputs a markdown table with rows ``(N, d, n_features, cpu_ms, gpu_ms, speedup)``. The crossover line (N*d where GPU starts winning) goes into the RFF default
``gpu_threshold`` calibration; the per-shape median is the source of truth for that constant.

Run with: ``D:/ProgramData/anaconda3/python.exe profiling/bench_transformer_rff.py``
Optional args: ``--sizes 1000,10000,100000`` to override the N sweep, ``--ds 64,1000,10000`` for the d sweep.
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import numpy as np

from mlframe.feature_engineering.transformer import compute_rff_features


def _time_one(fn: Callable[[], None], *, warmup: int, measured: int) -> tuple[float, float]:
    """Return ``(median_ms, iqr_ms)`` for the wall-clock of ``fn``. Discards ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(measured):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    median = statistics.median(times)
    q1 = times[n // 4]
    q3 = times[(3 * n) // 4]
    return median, q3 - q1


def _gpu_sync():
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,10000,100000", help="Comma-separated N values")
    parser.add_argument("--ds", type=str, default="64,1000,10000", help="Comma-separated d values")
    parser.add_argument("--n_features", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=7)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    ds = [int(d) for d in args.ds.split(",")]

    print(f"# RFF bench (n_features={args.n_features}, warmup={args.warmup}, measured={args.measured})\n")
    print("| N | d | cpu_ms median (IQR) | gpu_ms median (IQR) | gpu_speedup | crossover_N*d |")
    print("|---|---|---|---|---|---|")

    rng = np.random.default_rng(args.seed)
    for n in sizes:
        for d in ds:
            X = rng.standard_normal((n, d)).astype(np.float32)

            def call_cpu():
                compute_rff_features(X, seed=args.seed, n_features=args.n_features, use_gpu=False)
            def call_gpu():
                _gpu_sync()
                compute_rff_features(X, seed=args.seed, n_features=args.n_features, use_gpu=True)
                _gpu_sync()

            cpu_med, cpu_iqr = _time_one(call_cpu, warmup=args.warmup, measured=args.measured)
            try:
                gpu_med, gpu_iqr = _time_one(call_gpu, warmup=args.warmup, measured=args.measured)
                speedup = cpu_med / max(gpu_med, 1e-9)
                print(f"| {n} | {d} | {cpu_med:.1f} ({cpu_iqr:.1f}) | {gpu_med:.1f} ({gpu_iqr:.1f}) | {speedup:.2f}x | {n*d} |")
            except Exception as exc:
                print(f"| {n} | {d} | {cpu_med:.1f} ({cpu_iqr:.1f}) | GPU N/A ({type(exc).__name__}) | - | {n*d} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
