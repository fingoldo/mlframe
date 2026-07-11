"""cProfile harness for ``inference.recursive_forecast.recursive_multi_step_forecast``.

Run: ``python -m mlframe.inference._benchmarks.bench_recursive_forecast``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.inference.recursive_forecast import recursive_multi_step_forecast


def _run(n_series: int, n_steps: int) -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(5000, 1))
    y_train = 0.8 * X_train[:, 0] + rng.normal(scale=0.5, size=5000)
    model = Ridge(alpha=0.1).fit(X_train, y_train)

    features = pd.DataFrame({"lag_1": rng.normal(size=n_series)})
    recursive_multi_step_forecast(model, features, n_steps, "lag_1", lambda f, p, s: f)


if __name__ == "__main__":
    for n_series, n_steps in [(5000, 10), (50000, 10), (50000, 50)]:
        t0 = time.perf_counter()
        _run(n_series, n_steps)
        wall = time.perf_counter() - t0
        print(f"n_series={n_series:>6} n_steps={n_steps:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
