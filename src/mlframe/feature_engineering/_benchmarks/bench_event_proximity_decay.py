"""cProfile harness for ``feature_engineering.event_proximity_decay.event_proximity_decay_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_event_proximity_decay``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.event_proximity_decay import event_proximity_decay_features


def _run(n_days: int, n_events: int) -> None:
    dates = pd.Series(np.arange(n_days))
    rng = np.random.default_rng(0)
    event_days = rng.choice(n_days, n_events, replace=False).tolist()
    event_proximity_decay_features(dates, event_dates=event_days, cap=30)


if __name__ == "__main__":
    for n_days, n_events in [(10000, 20), (100000, 20), (100000, 200)]:
        t0 = time.perf_counter()
        _run(n_days, n_events)
        wall = time.perf_counter() - t0
        print(f"n_days={n_days:>7} n_events={n_events:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
