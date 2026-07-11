"""cProfile harness for ``models.lgbm_defaults.default_lgbm_params``.

Run: ``python -m mlframe.models._benchmarks.bench_lgbm_defaults_dart_heuristic``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

from mlframe.models.lgbm_defaults import default_lgbm_params


def _run(n_calls: int) -> None:
    for _ in range(n_calls):
        default_lgbm_params(objective="regression")
        default_lgbm_params(objective="regression", n_features=500)
        default_lgbm_params(objective="regression", n_features=500, n_estimators=200)


if __name__ == "__main__":
    for n_calls in [1000, 100000, 1000000]:
        t0 = time.perf_counter()
        _run(n_calls)
        wall = time.perf_counter() - t0
        print(f"n_calls={n_calls:>8} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(10)
    print(buf.getvalue())
