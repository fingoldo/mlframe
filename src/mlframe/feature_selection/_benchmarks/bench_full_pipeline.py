"""End-to-end bench of the MRMR GPU pipeline + per-host kernel_tuning_cache.

Runs MRMR.fit at several (n_rows, n_features) sizes, in two phases:

1. **Cold phase**: empty cache. Each fit pays the first-call CuPy NVRTC
   compile + numba JIT cache miss + the kernel_tuning_cache auto-tune
   sweep (~30s, one-time). The cache is built incrementally.
2. **Warm phase**: same fits, second pass. Cache is hit; all JITs warm;
   dispatch is O(1) lookup; wall time drops to the steady-state.

Output: per-size table of cold vs warm wall, speedup, plus the final
``KernelTuningCache`` dump for the live host. Useful as a sanity check
that the full session work (Layer 1/2/3 + .get() coalesce + shared-mem
kernel + cache dispatch + multi-GPU pin + provenance) actually lands
on a real workload.

Run::

    PYTHONPATH=src D:/ProgramData/anaconda3/python.exe \\
        -m mlframe.feature_selection._benchmarks.bench_full_pipeline
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

_SIZES: tuple[tuple[int, int], ...] = (
    (200_000, 12),
    (500_000, 20),
    (1_000_000, 30),
)


def _build(n_rows: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 5, size=n_rows).astype(np.int32) for _ in range(n_features)]
    X = pd.DataFrame(np.column_stack(cols), columns=[f"f{i}" for i in range(n_features)])
    y_raw = (X["f0"] + X["f1"] + 2 * X["f0"] * X["f1"] + rng.integers(0, 2, size=n_rows)) % 3
    return X, pd.Series(y_raw.astype(np.int32), name="y")


def _one_fit(X, y, *, fe_npermutations: int) -> float:
    from mlframe.feature_selection.filters.mrmr import MRMR
    sel = MRMR(
        fe_max_steps=1, fe_ntop_features=10,
        fe_npermutations=fe_npermutations,
        random_seed=11, verbose=0,
    )
    t0 = time.perf_counter()
    sel.fit(X, y)
    return time.perf_counter() - t0


def main() -> None:
    print("=== mlframe GPU pipeline end-to-end bench ===")
    # Sanity: where is the kernel_tuning_cache?
    try:
        from pyutilz.performance.kernel_tuning.cache import cache_path, hw_fingerprint
        print(f"hw_fingerprint: {hw_fingerprint()}")
        print(f"cache_path:     {cache_path()}")
        print(f"cache present:  {os.path.isfile(cache_path())}")
    except ImportError:
        print("pyutilz.performance.kernel_tuning.cache unavailable")

    # Prewarm once for the cold phase (numba + cupy).
    try:
        from mlframe.feature_selection.filters._prewarm import (
            prewarm_fs_numba_cache, prewarm_fs_cupy_kernels,
        )
        print("\n--- prewarm (one-time numba + cupy + auto-tune sweep) ---")
        t0 = time.perf_counter()
        prewarm_fs_numba_cache(verbose=False)
        prewarm_fs_cupy_kernels(verbose=False)
        print(f"prewarm wall: {time.perf_counter() - t0:.2f}s")
    except ImportError:
        print("prewarm unavailable")

    # Cold phase. Per Critic 3 B1: each fit wrapped in try/except so an
    # OOM at one size doesn't kill the whole bench.
    print("\n--- cold phase (cache may be partially built) ---")
    cold = []
    for n_rows, n_features in _SIZES:
        try:
            X, y = _build(n_rows, n_features, seed=11)
            wall = _one_fit(X, y, fe_npermutations=50)
            cold.append(wall)
            print(f"  n={n_rows:>10_} k={n_features:>3} fit_wall={wall:>6.2f}s")
        except Exception as exc:
            cold.append(None)
            print(f"  n={n_rows:>10_} k={n_features:>3} SKIPPED ({type(exc).__name__}: {exc})")

    # Warm phase: same fits, cache fully warm.
    print("\n--- warm phase (all caches hit) ---")
    warm = []
    for n_rows, n_features in _SIZES:
        try:
            X, y = _build(n_rows, n_features, seed=11)
            wall = _one_fit(X, y, fe_npermutations=50)
            warm.append(wall)
            print(f"  n={n_rows:>10_} k={n_features:>3} fit_wall={wall:>6.2f}s")
        except Exception as exc:
            warm.append(None)
            print(f"  n={n_rows:>10_} k={n_features:>3} SKIPPED ({type(exc).__name__}: {exc})")

    # Summary. Handle None entries (OOM-skipped) gracefully.
    print("\n=== summary ===")
    print(f"  {'size':<20} {'cold':>8} {'warm':>8} {'speedup':>8}")
    for (n_rows, n_features), cold_w, warm_w in zip(_SIZES, cold, warm):
        if cold_w is None or warm_w is None:
            print(f"  {n_rows:>10_} x {n_features:<6} {'skipped':>8} {'skipped':>8} {'-':>8}")
            continue
        spd = cold_w / max(warm_w, 1e-9)
        print(f"  {n_rows:>10_} x {n_features:<6} {cold_w:>7.2f}s {warm_w:>7.2f}s {spd:>7.2f}x")

    # Dump the cache for the report.
    try:
        from pyutilz.performance.kernel_tuning.cache import cache_path
        path = cache_path()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("\n=== KernelTuningCache contents ===")
            print(json.dumps(data, indent=2)[:1200] + " ...")
    except Exception as e:
        print(f"cache dump failed: {e}")


if __name__ == "__main__":
    main()
