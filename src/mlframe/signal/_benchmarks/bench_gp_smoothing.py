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


def _run(df) -> None:
    query_times = np.linspace(0, 20, 8)
    compute_gp_smoothed_features(df, "obj", "t", "y", query_times, alpha=0.05)


def _run_ensemble(df, ensemble_mode: str) -> None:
    query_times = np.linspace(0, 20, 8)
    compute_gp_smoothed_features(df, "obj", "t", "y", query_times, alpha=0.05, length_scales=[0.5, 2.0, 8.0], ensemble_mode=ensemble_mode)


if __name__ == "__main__":
    print("--- single fixed length_scale (default path) ---")
    for n_objects in [50, 200, 500]:
        df = _make_dataset(n_objects, seed=0)  # built outside the timer: the wall is the smoothing alone (audit 30-F4)
        t0 = time.perf_counter()
        _run(df)
        wall = time.perf_counter() - t0
        print(f"n_objects={n_objects:>5} -> {wall * 1000:9.2f} ms")

    print("--- multi-length-scale ensemble, cv_best (opt-in, extra LOO-CV fits) ---")
    for n_objects in [50, 200, 500]:
        df = _make_dataset(n_objects, seed=0)
        t0 = time.perf_counter()
        _run_ensemble(df, "cv_best")
        wall = time.perf_counter() - t0
        print(f"n_objects={n_objects:>5} -> {wall * 1000:9.2f} ms")

    print("--- multi-length-scale ensemble, cv_blend (opt-in, extra LOO-CV + full fit per candidate) ---")
    for n_objects in [50, 200, 500]:
        df = _make_dataset(n_objects, seed=0)
        t0 = time.perf_counter()
        _run_ensemble(df, "cv_blend")
        wall = time.perf_counter() - t0
        print(f"n_objects={n_objects:>5} -> {wall * 1000:9.2f} ms")

    df_profile = _make_dataset(500, seed=0)
    profiler = cProfile.Profile()
    profiler.enable()
    _run(df_profile)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("=== single fixed length_scale profile ===")
    print(buf.getvalue())

    profiler_ensemble = cProfile.Profile()
    profiler_ensemble.enable()
    _run_ensemble(df_profile, "cv_best")
    profiler_ensemble.disable()
    buf_ensemble = StringIO()
    stats_ensemble = pstats.Stats(profiler_ensemble, stream=buf_ensemble).sort_stats("cumulative")
    stats_ensemble.print_stats(15)
    print("=== multi-length-scale ensemble (cv_best) profile ===")
    print(buf_ensemble.getvalue())
