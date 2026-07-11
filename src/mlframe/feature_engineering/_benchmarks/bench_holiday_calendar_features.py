"""cProfile harness for ``feature_engineering.holiday_calendar_features.holiday_calendar_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_holiday_calendar_features``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import pandas as pd

from mlframe.feature_engineering.holiday_calendar_features import holiday_calendar_features


def _run(n_years: int, n_calls: int) -> None:
    dates = pd.Series(pd.date_range("2000-01-01", periods=365 * n_years, freq="D"))
    for _ in range(n_calls):
        holiday_calendar_features(dates, country="US")


if __name__ == "__main__":
    for n_years, n_calls in [(1, 20), (25, 20), (25, 100)]:
        t0 = time.perf_counter()
        _run(n_years, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_years={n_years:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(25, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
