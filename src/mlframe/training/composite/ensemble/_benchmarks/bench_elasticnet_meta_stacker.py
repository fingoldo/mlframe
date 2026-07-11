"""cProfile harness for ``training.composite.ensemble._stackers.fit_elasticnet_meta_stacker``.

Run: ``python -m mlframe.training.composite.ensemble._benchmarks.bench_elasticnet_meta_stacker``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite.ensemble._stackers import fit_elasticnet_meta_stacker


def _make_oof(n_rows: int, n_components: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_rows)
    b = rng.normal(size=n_rows)
    y = 0.7 * a + 0.3 * b + 0.05 * rng.normal(size=n_rows)
    extra = [rng.normal(size=n_rows) for _ in range(n_components - 2)]
    X = np.column_stack([a, b, *extra])
    return X, y


def _run(n_rows: int, n_components: int) -> None:
    X, y = _make_oof(n_rows, n_components, seed=0)
    fit_elasticnet_meta_stacker(X, y, n_components)


if __name__ == "__main__":
    for n_rows, n_components in [(2000, 10), (20000, 10), (20000, 50)]:
        t0 = time.perf_counter()
        _run(n_rows, n_components)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6} n_components={n_components:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
