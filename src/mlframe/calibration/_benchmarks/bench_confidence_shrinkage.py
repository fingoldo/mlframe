"""cProfile harness for ``calibration.compute_oof_confidence`` / ``apply_confidence_shrinkage``.

Run: ``python -m mlframe.calibration._benchmarks.bench_confidence_shrinkage``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.confidence_shrinkage import apply_confidence_shrinkage, compute_oof_confidence


def _run(n_samples: int, n_outputs: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    preds = {f"out_{i}": rng.uniform(0, 1, n_samples) for i in range(n_outputs)}
    labels = {f"out_{i}": rng.integers(0, 2, n_samples) for i in range(n_outputs)}
    for _ in range(n_calls):
        confidences = {name: compute_oof_confidence(preds[name], labels[name]) for name in preds}
        apply_confidence_shrinkage(preds, confidences)


def _run_per_segment(n_samples: int, n_outputs: int, n_segments: int, n_calls: int) -> None:
    """Same shape as ``_run``, but exercising the opt-in per-segment path end to end: per-output
    ``segment_ids`` -> per-segment confidence dict -> per-segment shrinkage weight per row.
    """
    rng = np.random.default_rng(0)
    preds = {f"out_{i}": rng.uniform(0, 1, n_samples) for i in range(n_outputs)}
    labels = {f"out_{i}": rng.integers(0, 2, n_samples) for i in range(n_outputs)}
    segment_ids = rng.integers(0, n_segments, n_samples)
    segments = {name: segment_ids for name in preds}
    for _ in range(n_calls):
        confidences = {name: compute_oof_confidence(preds[name], labels[name], segment_ids=segment_ids) for name in preds}
        apply_confidence_shrinkage(preds, confidences, segments=segments)


if __name__ == "__main__":
    for n_samples, n_outputs, n_calls in [(10_000, 50, 20), (1_000_000, 100, 5)]:
        t0 = time.perf_counter()
        _run(n_samples, n_outputs, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>9,} n_outputs={n_outputs:>4} n_calls={n_calls:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call")

    for n_samples, n_outputs, n_segments, n_calls in [(10_000, 50, 10, 20), (1_000_000, 100, 20, 5)]:
        t0 = time.perf_counter()
        _run_per_segment(n_samples, n_outputs, n_segments, n_calls)
        wall = time.perf_counter() - t0
        print(
            f"[per-segment] n_samples={n_samples:>9,} n_outputs={n_outputs:>4} n_segments={n_segments:>3} n_calls={n_calls:>3} "
            f"-> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 100, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_per_segment(1_000_000, 100, 20, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[per-segment]")
    print(buf.getvalue())
