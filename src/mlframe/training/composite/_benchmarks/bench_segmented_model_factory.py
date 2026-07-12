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


def _make_regional_dataset(n_regions: int, n_large: int, n_tiny_per_region: int, n_tiny: int, seed: int) -> pd.DataFrame:
    """Mirrors ``test_biz_val_segmented_model_factory``'s hierarchical-shrinkage fixture: each region has
    one large segment plus several tiny ones, so ``shrinkage_min_rows``/``shrinkage_parent_keys`` has both
    parent-model fitting (one per region) and per-row blending work to profile, not just the plain path."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_regions):
        w = rng.normal(size=2)
        x1 = rng.normal(size=n_large)
        x2 = rng.normal(size=n_large)
        y = x1 * w[0] + x2 * w[1] + rng.normal(scale=0.3, size=n_large)
        for i in range(n_large):
            rows.append({"region": r, "segment": f"{r}_BIG", "x1": x1[i], "x2": x2[i], "y": y[i]})
        for t in range(n_tiny_per_region):
            w_tiny = w + rng.normal(scale=0.1, size=2)
            x1 = rng.normal(size=n_tiny)
            x2 = rng.normal(size=n_tiny)
            y = x1 * w_tiny[0] + x2 * w_tiny[1] + rng.normal(scale=0.3, size=n_tiny)
            for i in range(n_tiny):
                rows.append({"region": r, "segment": f"{r}_TINY{t}", "x1": x1[i], "x2": x2[i], "y": y[i]})
    return pd.DataFrame(rows)


def _run_shrinkage(n_regions: int, n_large: int, n_tiny_per_region: int, n_tiny: int) -> None:
    df = _make_regional_dataset(n_regions, n_large, n_tiny_per_region, n_tiny, seed=0)
    factory = SegmentedModelFactory(
        estimator_factory=lambda: LinearRegression(),
        segment_keys=["region", "segment"],
        min_segment_rows=2,
        shrinkage_min_rows=30,
        shrinkage_parent_keys=["region"],
        shrinkage_k=10.0,
    )
    factory.fit(df[["region", "segment", "x1", "x2"]], df["y"])
    factory.predict(df[["region", "segment", "x1", "x2"]])


if __name__ == "__main__":
    for n_segments, n_per_segment in [(20, 50), (100, 50), (100, 200)]:
        t0 = time.perf_counter()
        _run(n_segments, n_per_segment)
        wall = time.perf_counter() - t0
        print(f"n_segments={n_segments:>4} n_per_segment={n_per_segment:>4} (n_rows={n_segments*n_per_segment:>7,}) -> {wall * 1000:9.2f} ms")

    for n_regions, n_large, n_tiny_per_region, n_tiny in [(5, 50, 4, 5), (20, 50, 4, 5), (20, 200, 8, 5)]:
        t0 = time.perf_counter()
        _run_shrinkage(n_regions, n_large, n_tiny_per_region, n_tiny)
        wall = time.perf_counter() - t0
        n_rows = n_regions * (n_large + n_tiny_per_region * n_tiny)
        print(f"[shrinkage] n_regions={n_regions:>4} n_large={n_large:>4} n_tiny_per_region={n_tiny_per_region:>3} (n_rows={n_rows:>7,}) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- plain per-segment path ---")
    print(buf.getvalue())

    profiler_shrinkage = cProfile.Profile()
    profiler_shrinkage.enable()
    _run_shrinkage(20, 200, 8, 5)
    profiler_shrinkage.disable()
    buf_shrinkage = StringIO()
    stats_shrinkage = pstats.Stats(profiler_shrinkage, stream=buf_shrinkage).sort_stats("cumulative")
    stats_shrinkage.print_stats(15)
    print("--- hierarchical-shrinkage fallback path ---")
    print(buf_shrinkage.getvalue())
