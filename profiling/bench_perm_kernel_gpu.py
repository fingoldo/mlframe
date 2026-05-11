"""A/B/C bench: CPU numba prange vs cupy GPU for
``_count_nfailed_joint_indep_*`` across a shape sweep.

Wave 13c. Verifies the crossover threshold
``_GPU_PERM_KERNEL_THRESHOLD_NXP`` is correctly calibrated:
- Below the threshold (small N, low n_perms), CPU should win
- Above the threshold (large N, high n_perms), GPU should win

Usage:
    python -m mlframe.profiling.bench_perm_kernel_gpu
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from mlframe.feature_selection.filters.cat_interactions import (
    _count_nfailed_joint_indep_prange,
    _count_nfailed_joint_indep_cupy,
)


def _make_inputs(n: int, K_pair: int = 20, K_x: int = 8, K_y: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    classes_pair = rng.integers(0, K_pair, n).astype(np.int32)
    classes_x1 = rng.integers(0, K_x, n).astype(np.int32)
    classes_x2 = rng.integers(0, K_x, n).astype(np.int32)
    classes_y = rng.integers(0, K_y, n).astype(np.int32)
    freqs_pair = np.bincount(classes_pair, minlength=K_pair).astype(np.float64) / n
    freqs_x1 = np.bincount(classes_x1, minlength=K_x).astype(np.float64) / n
    freqs_x2 = np.bincount(classes_x2, minlength=K_x).astype(np.float64) / n
    freqs_y = np.bincount(classes_y, minlength=K_y).astype(np.float64) / n
    return (classes_pair, freqs_pair, classes_x1, freqs_x1,
            classes_x2, freqs_x2, classes_y, freqs_y)


def bench_once(n: int, n_perms: int, n_repeat: int = 3) -> tuple:
    args = _make_inputs(n)
    # Warm both kernels (compile / GPU init)
    _count_nfailed_joint_indep_prange(*args, 0.0, n_perms, 7, np.int32)
    try:
        import cupy as _cp  # noqa: F401
        _count_nfailed_joint_indep_cupy(*args, 0.0, n_perms, 7)
        gpu_available = True
    except Exception as e:
        gpu_available = False
        print(f"  GPU bench skipped ({e})")

    cpu_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _count_nfailed_joint_indep_prange(*args, 0.0, n_perms, 7, np.int32)
        cpu_times.append(time.perf_counter() - t0)

    gpu_times = []
    if gpu_available:
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _count_nfailed_joint_indep_cupy(*args, 0.0, n_perms, 7)
            gpu_times.append(time.perf_counter() - t0)

    cpu_m = statistics.mean(cpu_times) * 1000
    cpu_s = (statistics.stdev(cpu_times) * 1000) if len(cpu_times) > 1 else 0.0
    gpu_m = (statistics.mean(gpu_times) * 1000) if gpu_times else float('nan')
    gpu_s = (statistics.stdev(gpu_times) * 1000) if len(gpu_times) > 1 else 0.0
    return cpu_m, cpu_s, gpu_m, gpu_s


def main() -> None:
    print(f"# _count_nfailed_joint_indep_* (CPU numba prange vs cupy GPU)\n")
    print(f"  {'N':>10}  {'n_perms':>8}  {'CPU ms':>14}  {'GPU ms':>14}  {'winner':>7}")
    shapes = [
        (100_000, 3),
        (100_000, 50),
        (1_000_000, 3),
        (1_000_000, 50),
        (1_000_000, 100),
        (5_000_000, 10),
    ]
    for n, n_perms in shapes:
        cpu_m, cpu_s, gpu_m, gpu_s = bench_once(n, n_perms)
        if np.isnan(gpu_m):
            winner = "CPU only"
        elif cpu_m < gpu_m:
            winner = "CPU"
        else:
            winner = "GPU"
        gpu_str = f"{gpu_m:>5.1f} +/- {gpu_s:>4.1f}" if not np.isnan(gpu_m) else "    (n/a)"
        print(
            f"  {n:>10_}  {n_perms:>8d}  "
            f"{cpu_m:>5.1f} +/- {cpu_s:>4.1f}  "
            f"{gpu_str}  {winner:>7}"
        )


if __name__ == "__main__":
    main()
