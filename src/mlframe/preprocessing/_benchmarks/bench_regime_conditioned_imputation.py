"""cProfile harness for ``preprocessing.regime_conditioned_median_fill``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_regime_conditioned_imputation``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.regime_conditioned_imputation import regime_conditioned_median_fill


def _make_data(n_rows: int, n_features: int, n_regimes: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_features)}
    for c in cols:
        nan_mask = rng.random(n_rows) < 0.2
        cols[c][nan_mask] = np.nan
    return pd.DataFrame({"regime": rng.integers(0, n_regimes, n_rows), **cols})


def _run(n_rows: int, n_features: int, n_regimes: int) -> None:
    df = _make_data(n_rows, n_features, n_regimes, seed=0)
    regime_conditioned_median_fill(df, regime_col="regime")


if __name__ == "__main__":
    for n_rows, n_features, n_regimes in [(20_000, 10, 5), (500_000, 20, 10)]:
        t0 = time.perf_counter()
        _run(n_rows, n_features, n_regimes)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_features={n_features:>3} n_regimes={n_regimes:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, 10, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
