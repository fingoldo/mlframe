"""cProfile harness for ``feature_engineering.holiday_calendar_features.holiday_calendar_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_holiday_calendar_features``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.holiday_calendar_features import holiday_calendar_features
from mlframe.feature_engineering.holiday_locale_target_encoding import holiday_name_target_encode_cross_locale


def _run(n_years: int, n_calls: int) -> None:
    dates = pd.Series(pd.date_range("2000-01-01", periods=365 * n_years, freq="D"))
    for _ in range(n_calls):
        holiday_calendar_features(dates, country="US")


def _run_with_name(n_years: int, n_calls: int) -> None:
    dates = pd.Series(pd.date_range("2000-01-01", periods=365 * n_years, freq="D"))
    for _ in range(n_calls):
        holiday_calendar_features(dates, country="US", include_nearest_name=True)


def _run_multi_country(n_years: int, n_calls: int, n_countries: int) -> None:
    # Opt-in multi-region blend path: each extra country in the list rebuilds/looks-up its own calendar and
    # OR-reduces into the combined `any_*` columns -- cost should scale roughly linearly in n_countries since
    # the underlying per-country calendar build is itself lru_cache'd (see `_cached_holiday_dates`).
    dates = pd.Series(pd.date_range("2000-01-01", periods=365 * n_years, freq="D"))
    countries = ["US", "CA", "GB", "DE", "AU"][:n_countries]
    for _ in range(n_calls):
        holiday_calendar_features(dates, country="US", countries=countries)


def _run_cross_locale(n_years: int, n_calls: int, cross_locale_shrinkage: float | None) -> None:
    # Two-country panel: rows tagged US/CA in equal proportion, holiday names drawn from the real US calendar
    # (matches the shape of the actual biz_value scenario -- one rich locale's history feeding a sparser one).
    dates = pd.Series(pd.date_range("2000-01-01", periods=365 * n_years, freq="D"))
    feats = holiday_calendar_features(dates, country="US", include_nearest_name=True, name_window_days=0)
    names = feats["holiday_nearest_holiday_name"].to_numpy()
    rng = np.random.default_rng(0)
    countries = np.where(rng.random(len(names)) < 0.5, "US", "CA")
    y = rng.normal(100.0, 5.0, size=len(names))
    order = np.arange(len(names))
    for _ in range(n_calls):
        holiday_name_target_encode_cross_locale(
            names, countries, y, order=order, smoothing=1.0, cross_locale_shrinkage=cross_locale_shrinkage
        )


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

    for n_years, n_calls in [(1, 20), (25, 20), (25, 100)]:
        t0 = time.perf_counter()
        _run_with_name(n_years, n_calls)
        wall = time.perf_counter() - t0
        print(f"[include_nearest_name] n_years={n_years:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_with_name(25, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    for n_years, n_calls, n_countries in [(1, 20, 2), (25, 20, 2), (25, 100, 2), (25, 100, 5)]:
        t0 = time.perf_counter()
        _run_multi_country(n_years, n_calls, n_countries)
        wall = time.perf_counter() - t0
        print(f"[countries={n_countries}] n_years={n_years:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_multi_country(25, 100, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    for cross_locale_shrinkage, label in [(None, "same-country-only"), (5.0, "cross-locale")]:
        for n_years, n_calls in [(1, 20), (25, 20), (25, 100)]:
            t0 = time.perf_counter()
            _run_cross_locale(n_years, n_calls, cross_locale_shrinkage)
            wall = time.perf_counter() - t0
            print(f"[{label}] n_years={n_years:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_cross_locale(25, 100, 5.0)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
