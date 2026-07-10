"""cProfile harness for ``signal.gp_smoothing.compute_gp_smoothed_features``.

Run: ``python -m mlframe.signal._benchmarks.bench_gp_smoothing``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.signal.gp_smoothing import compute_gp_smoothed_features


def _make_dataset(n_objects: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for obj in range(n_objects):
        peak_t = rng.uniform(5, 15)
        n_obs = rng.integers(3, 8)
        t_obs = np.sort(rng.uniform(0, 20, n_obs))
        y_obs = np.exp(-((t_obs - peak_t) ** 2) / 8.0) + rng.normal(scale=0.2, size=n_obs)
        for t, y in zip(t_obs, y_obs):
            rows.append({"obj": obj, "t": t, "y": y})
    return pd.DataFrame(rows)


def _run(n_objects: int) -> None:
    df = _make_dataset(n_objects, seed=0)
    query_times = np.linspace(0, 20, 8)
    compute_gp_smoothed_features(df, "obj", "t", "y", query_times, alpha=0.05)


if __name__ == "__main__":
    for n_objects in [50, 200, 500]:
        t0 = time.perf_counter()
        _run(n_objects)
        wall = time.perf_counter() - t0
        print(f"n_objects={n_objects:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
