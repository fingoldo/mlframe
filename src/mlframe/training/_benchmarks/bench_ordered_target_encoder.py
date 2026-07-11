"""cProfile harness for ``training.feature_handling.ordered_target_encode``.

Run: ``python -m mlframe.training._benchmarks.bench_ordered_target_encoder``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode, ordered_target_encode_batch


def _run(n_rows: int, n_cats: int) -> None:
    rng = np.random.default_rng(0)
    cats = rng.integers(0, n_cats, n_rows)
    y = rng.integers(0, 2, n_rows).astype(np.float64)
    ordered_target_encode(cats, y, smoothing=1.0)


def _make_batch_inputs(n_rows: int, n_cats: int, n_cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    y = rng.integers(0, 2, n_rows).astype(np.float64)
    columns = {f"c{i}": rng.integers(0, n_cats, n_rows) for i in range(n_cols)}
    return columns, y, order


def _run_separate_calls(n_rows: int, n_cats: int, n_cols: int) -> None:
    columns, y, order = _make_batch_inputs(n_rows, n_cats, n_cols)
    for cats in columns.values():
        ordered_target_encode(cats, y, order=order, smoothing=1.0)


def _run_batch(n_rows: int, n_cats: int, n_cols: int) -> None:
    columns, y, order = _make_batch_inputs(n_rows, n_cats, n_cols)
    ordered_target_encode_batch(columns, y, order=order, smoothing=1.0)


if __name__ == "__main__":
    for n_rows, n_cats in [(50_000, 500), (1_000_000, 5_000)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cats)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_cats={n_cats:>6,} -> {wall * 1000:9.2f} ms")

    print("\nordered_target_encode_batch vs N separate ordered_target_encode calls (shared y/order):")
    for n_rows, n_cats, n_cols in [(40_000, 2_000, 25), (200_000, 5_000, 25)]:
        # warm up both paths once before timing.
        _run_separate_calls(n_rows, n_cats, 2)
        _run_batch(n_rows, n_cats, 2)

        t0 = time.perf_counter()
        _run_separate_calls(n_rows, n_cats, n_cols)
        separate_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_batch(n_rows, n_cats, n_cols)
        batch_wall = time.perf_counter() - t0

        print(
            f"n_rows={n_rows:>9,} n_cats={n_cats:>6,} n_cols={n_cols:>3} -> "
            f"separate={separate_wall * 1000:9.2f}ms batch={batch_wall * 1000:9.2f}ms speedup={separate_wall / batch_wall:5.2f}x"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 5_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    columns, y, order = _make_batch_inputs(200_000, 5_000, 25)
    profiler = cProfile.Profile()
    profiler.enable()
    ordered_target_encode_batch(columns, y, order=order, smoothing=1.0)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("ordered_target_encode_batch profile (200k rows x 25 columns):")
    print(buf.getvalue())
