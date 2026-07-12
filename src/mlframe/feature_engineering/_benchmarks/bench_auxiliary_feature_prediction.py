"""cProfile harness for ``feature_engineering.compute_auxiliary_feature_prediction_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_auxiliary_feature_prediction``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from mlframe.feature_engineering import compute_auxiliary_feature_prediction_features


def _make_dataset(n: int, d: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    cols = {"ext_source": z + rng.normal(scale=1.8, size=n)}
    cols.update({f"f{i}": z + rng.normal(scale=1.2, size=n) for i in range(d)})
    return pd.DataFrame(cols)


def _run(n: int, d: int, n_uncertainty_repeats: int = 1) -> None:
    X = _make_dataset(n, d, seed=0)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    compute_auxiliary_feature_prediction_features(X, ["ext_source"], splitter=kf, seed=0, n_uncertainty_repeats=n_uncertainty_repeats)


if __name__ == "__main__":
    for n, d in [(500, 10), (2_000, 10), (5_000, 20)]:
        t0 = time.perf_counter()
        _run(n, d)
        wall = time.perf_counter() - t0
        print(f"n={n:>6,} d={d:>3} -> {wall * 1000:9.2f} ms")

    print("-- with n_uncertainty_repeats=8 (opt-in confidence column) --")
    for n, d in [(500, 10), (2_000, 10), (5_000, 20)]:
        t0 = time.perf_counter()
        _run(n, d, n_uncertainty_repeats=8)
        wall = time.perf_counter() - t0
        print(f"n={n:>6,} d={d:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_unc = cProfile.Profile()
    profiler_unc.enable()
    _run(5_000, 20, n_uncertainty_repeats=8)
    profiler_unc.disable()
    buf_unc = StringIO()
    stats_unc = pstats.Stats(profiler_unc, stream=buf_unc).sort_stats("cumulative")
    stats_unc.print_stats(15)
    print("-- uncertainty path profile --")
    print(buf_unc.getvalue())
