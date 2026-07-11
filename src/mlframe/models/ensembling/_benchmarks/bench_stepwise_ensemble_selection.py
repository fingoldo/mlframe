"""cProfile harness for ``models.ensembling.selection.stepwise_ensemble_selection``.

Run: ``python -m mlframe.models.ensembling._benchmarks.bench_stepwise_ensemble_selection``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.models.ensembling.selection import stepwise_ensemble_selection


def _make_matrix(m: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.int64)
    signal = 2 * y - 1
    noise_levels = np.linspace(0.4, 1.6, m)
    preds = np.empty((m, n))
    for i, nl in enumerate(noise_levels):
        preds[i] = np.clip(0.5 + 0.4 * signal + rng.normal(0, nl, n), 0, 1)
    return preds, y


def _run(m: int, n: int) -> None:
    preds, y = _make_matrix(m, n, seed=0)
    stepwise_ensemble_selection(preds, y, max_picks=m)


if __name__ == "__main__":
    for m, n in [(10, 5000), (10, 100000), (30, 100000)]:
        t0 = time.perf_counter()
        _run(m, n)
        wall = time.perf_counter() - t0
        print(f"m={m:>3} n={n:>7} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(30, 100000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    buf_tot = StringIO()
    stats_tot = pstats.Stats(profiler, stream=buf_tot).sort_stats("tottime")
    stats_tot.print_stats(15)
    print(buf_tot.getvalue())
