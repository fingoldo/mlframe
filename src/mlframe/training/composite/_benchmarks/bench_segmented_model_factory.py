"""cProfile harness for ``training.composite.SegmentedModelFactory``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_segmented_model_factory``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import SegmentedModelFactory


def _make_dataset(n_segments: int, n_per_segment: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_segments):
        w = rng.normal(size=2)
        x1 = rng.normal(size=n_per_segment)
        x2 = rng.normal(size=n_per_segment)
        y = x1 * w[0] + x2 * w[1] + rng.normal(scale=0.3, size=n_per_segment)
        for i in range(n_per_segment):
            rows.append({"segment": s, "x1": x1[i], "x2": x2[i], "y": y[i]})
    return pd.DataFrame(rows)


def _run(n_segments: int, n_per_segment: int) -> None:
    df = _make_dataset(n_segments, n_per_segment, seed=0)
    factory = SegmentedModelFactory(estimator_factory=lambda: LinearRegression(), segment_keys=["segment"])
    factory.fit(df[["segment", "x1", "x2"]], df["y"])
    factory.predict(df[["segment", "x1", "x2"]])


if __name__ == "__main__":
    for n_segments, n_per_segment in [(20, 50), (100, 50), (100, 200)]:
        t0 = time.perf_counter()
        _run(n_segments, n_per_segment)
        wall = time.perf_counter() - t0
        print(f"n_segments={n_segments:>4} n_per_segment={n_per_segment:>4} (n_rows={n_segments*n_per_segment:>7,}) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
