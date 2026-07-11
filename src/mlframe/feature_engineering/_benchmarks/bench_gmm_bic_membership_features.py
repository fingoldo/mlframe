"""cProfile harness for ``feature_engineering.gmm_bic_membership_features.gmm_bic_membership_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_gmm_bic_membership_features``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.gmm_bic_membership_features import gmm_bic_membership_features


def _make_dataset(n_rows: int, n_features: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_rows, n_features)), columns=[f"f{i}" for i in range(n_features)])


def _run(n_rows: int, n_features: int) -> None:
    df = _make_dataset(n_rows, n_features, seed=0)
    gmm_bic_membership_features(df, n_components_range=(2, 3, 4, 5, 6))


if __name__ == "__main__":
    for n_rows, n_features in [(1000, 5), (5000, 5), (5000, 15)]:
        t0 = time.perf_counter()
        _run(n_rows, n_features)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>5} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5000, 15)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
