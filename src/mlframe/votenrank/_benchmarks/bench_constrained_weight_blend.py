"""cProfile harness for ``votenrank.constrained_weight_blend``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_constrained_weight_blend``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _run(n_samples: int, n_models: int, n_restarts: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(n_samples)
    preds = [y_true + rng.uniform(0.2, 2.0) * rng.standard_normal(n_samples) for _ in range(n_models)]
    constrained_weight_blend(preds, y_true, _rmse, n_restarts=n_restarts, random_state=0)


if __name__ == "__main__":
    for n_samples, n_models, n_restarts in [(5_000, 20, 5), (100_000, 50, 5)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models, n_restarts)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>9,} n_models={n_models:>3} n_restarts={n_restarts:>2} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 20, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
