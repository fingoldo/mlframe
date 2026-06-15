"""iter140 microbench: plug-in MI classif njit-vs-cuda crossover on the running host.

Measures ``_plugin_mi_classif_njit`` vs ``_plugin_mi_classif_cuda`` (k=1) and the batch
variants (k>1) across n, to find the real GPU crossover on THIS GPU and compare it to the
``plugin_mi_classif_dispatch`` hardcoded fallback (75k single / 10k batch, measured on a
GTX 1050 Ti cc 6.1, 2026-05-20). Run in its own subprocess so a native cupy crash cannot
take down the calling session.

Usage:  python -m mlframe.feature_selection._benchmarks.bench_mi_classif_crossover_iter140
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _best(fn, *args, iters=9, warmup=2):
    for _ in range(warmup):
        fn(*args)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000.0)


def main():
    import cupy as cp

    from mlframe.feature_selection.filters.hermite_fe import (
        _plugin_mi_classif_njit,
        _plugin_mi_classif_cuda,
        _plugin_mi_classif_batch_njit,
        _plugin_mi_classif_batch_cuda,
    )

    rng = np.random.default_rng(11)
    n_axis = (5_000, 10_000, 20_000, 35_000, 50_000, 75_000, 100_000, 250_000, 500_000, 1_000_000)

    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print("\n=== k=1 (single column) ===")
    print(f"{'n':>10} {'njit_ms':>10} {'cuda_ms':>10} {'winner':>8} {'speedup':>8}")
    for n in n_axis:
        x = rng.normal(size=n)
        y = rng.integers(0, 3, size=n).astype(np.int64)
        m_n = _best(_plugin_mi_classif_njit, x, y, 20)
        m_c = _best(_plugin_mi_classif_cuda, x, y, 20)
        cp.get_default_memory_pool().free_all_blocks()
        win = "cuda" if m_c < m_n else "njit"
        sp = m_n / m_c if m_c > 0 else 0.0
        print(f"{n:>10} {m_n:>10.3f} {m_c:>10.3f} {win:>8} {sp:>7.2f}x")

    for k in (5, 20):
        print(f"\n=== k={k} (batch) ===")
        print(f"{'n':>10} {'njit_ms':>10} {'cuda_ms':>10} {'winner':>8} {'speedup':>8}")
        for n in n_axis:
            X = rng.normal(size=(n, k))
            y = rng.integers(0, 3, size=n).astype(np.int64)
            m_n = _best(_plugin_mi_classif_batch_njit, X, y, 20)
            m_c = _best(_plugin_mi_classif_batch_cuda, X, y, 20)
            cp.get_default_memory_pool().free_all_blocks()
            win = "cuda" if m_c < m_n else "njit"
            sp = m_n / m_c if m_c > 0 else 0.0
            print(f"{n:>10} {m_n:>10.3f} {m_c:>10.3f} {win:>8} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
