"""Wave 65 (2026-05-20): RFF matmul CPU-vs-GPU crossover bench.

Closes the wave-23 deferral on `random_features.py:_should_use_gpu_rff`. The
auto-dispatch consults `kernel_tuning_cache.lookup("rff_matmul", work=...)` for
an HW-tuned crossover; this script populates that cache entry by timing
CPU/GPU on a sweep of (N, d) shapes.

Run once on dev hardware (or per-machine in production):
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_engineering._benchmarks.bench_rff_matmul

Result: writes a `work_threshold` entry to `kernel_tuning_cache` keyed by
hw_fingerprint(). The auto-dispatch picks it up next call. Falls back to the
prior placeholder (5_000_000 * 256) when the cache miss happens.

The bench measures actual numpy GEMM (CPU) vs cupy GEMM (GPU) for the RFF
projection `X @ projection_matrix.T` on a sweep of work=N*d. Picks the
smallest work value at which GPU wins by >=20%.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _bench_cpu(n: int, d: int, n_features: int, n_reps: int = 3) -> float:
    """Time CPU numpy GEMM in seconds (median of n_reps)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d), dtype=np.float32)
    W = rng.standard_normal((n_features, d), dtype=np.float32)
    # Warm-up.
    _ = X @ W.T
    timings = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = X @ W.T
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings))


def _bench_gpu(n: int, d: int, n_features: int, n_reps: int = 3) -> Optional[float]:
    """Time GPU cupy GEMM in seconds (median of n_reps). Returns None if no GPU."""
    try:
        import cupy as cp
    except ImportError:
        return None
    try:
        rng = cp.random.default_rng(0)
        X_g = rng.standard_normal((n, d), dtype=cp.float32)
        W_g = rng.standard_normal((n_features, d), dtype=cp.float32)
        # Warm-up + sync.
        _ = X_g @ W_g.T
        cp.cuda.Stream.null.synchronize()
        timings = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = X_g @ W_g.T
            cp.cuda.Stream.null.synchronize()
            timings.append(time.perf_counter() - t0)
        return float(np.median(timings))
    except Exception as e:
        logger.warning("GPU bench failed: %s", e)
        return None


def calibrate(n_features: int = 256, n_reps: int = 3) -> Tuple[Optional[int], List[dict]]:
    """Sweep (N, d) shapes, return (work_threshold, sweep_log).

    work_threshold = smallest N*d at which GPU is >=20% faster than CPU. None
    when no GPU is available or GPU never wins on the sweep grid.
    """
    sweep_shapes: List[Tuple[int, int]] = [
        (1000, 16), (1000, 64), (1000, 256),
        (10_000, 16), (10_000, 64), (10_000, 256),
        (100_000, 16), (100_000, 64), (100_000, 256),
        (500_000, 64), (500_000, 256),
        (1_000_000, 64), (1_000_000, 256),
        (5_000_000, 256),
    ]
    sweep_log: List[dict] = []
    threshold: Optional[int] = None
    for n, d in sweep_shapes:
        work = n * d
        cpu_s = _bench_cpu(n, d, n_features, n_reps=n_reps)
        gpu_s = _bench_gpu(n, d, n_features, n_reps=n_reps)
        speedup = (cpu_s / gpu_s) if gpu_s and gpu_s > 0 else None
        gpu_wins = speedup is not None and speedup >= 1.2
        sweep_log.append({
            "n": n, "d": d, "work": work,
            "cpu_s": cpu_s, "gpu_s": gpu_s,
            "speedup": speedup, "gpu_wins": gpu_wins,
        })
        if gpu_wins and threshold is None:
            threshold = work
    return threshold, sweep_log


def main() -> int:
    """CLI entry: run calibration, write to kernel_tuning_cache, print summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    threshold, sweep = calibrate()
    print(f"\n{'n':>10} {'d':>5} {'work':>12} {'cpu_s':>10} {'gpu_s':>10} {'speedup':>8} {'gpu_wins':>9}")
    for row in sweep:
        gs = f"{row['gpu_s']:>10.4f}" if row["gpu_s"] is not None else f"{'(no GPU)':>10}"
        sp = f"{row['speedup']:>8.2f}x" if row["speedup"] is not None else f"{'-':>9}"
        print(
            f"{row['n']:>10} {row['d']:>5} {row['work']:>12} "
            f"{row['cpu_s']:>10.4f} {gs} {sp} {str(row['gpu_wins']):>9}"
        )
    if threshold is None:
        print("\nNo GPU crossover found on this hardware (CPU faster across the sweep). "
              "Placeholder fallback (5_000_000 * 256) will remain in effect.")
        return 1
    print(f"\nGPU crossover threshold: work >= {threshold} (writing to kernel_tuning_cache)")
    try:
        from pyutilz.system.kernel_tuning_cache import KernelTuningCache
        cache = KernelTuningCache.load_or_create()
        cache.store("rff_matmul", {"work_threshold": int(threshold), "calibrated_at": time.time()})
        print("Cache updated. random_features._should_use_gpu_rff will use this value next call.")
        return 0
    except Exception as e:
        print(f"Could not write to kernel_tuning_cache: {e}")
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
