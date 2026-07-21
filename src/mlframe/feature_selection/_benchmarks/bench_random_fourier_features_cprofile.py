"""cProfile harness for ``filters._random_fourier_features_fe.random_fourier_features``
(mrmr_audit_2026-07-20 fe_expansion.md "Random Fourier Features").

Run: ``python -m mlframe.feature_selection._benchmarks.bench_random_fourier_features_cprofile``

Cost is a single (n, p) @ (p, m) GEMM plus elementwise cos -- expect ``np.dot``/matmul to dominate
cumtime, not a numba/cupy kernel (this is pure-numpy by design; GPU residency is HIGH via a single
cupy matmul + elementwise cos, not yet ported).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._random_fourier_features_fe import random_fourier_features


def _make_data(n_rows: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, p))


def _run(n_rows: int, p: int, m: int) -> None:
    X = _make_data(n_rows, p, seed=0)
    random_fourier_features(X, m=m, random_state=0)


if __name__ == "__main__":
    for n_rows, p, m in [(2_000, 10, 64), (20_000, 20, 128), (100_000, 30, 256)]:
        t0 = time.perf_counter()
        _run(n_rows, p, m)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} p={p:>3} m={m:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 30, 256)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())
