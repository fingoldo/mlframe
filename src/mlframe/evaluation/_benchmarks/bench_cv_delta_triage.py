"""cProfile harness for ``evaluation.triage_cv_delta``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_cv_delta_triage``

The function is a thin wrapper around ``cv_score_equivalence_band`` (already benched/optimized in
``bench_noise_band.py``) plus a couple of scalar comparisons -- O(n_folds) per call, called at selection-loop
cadence (thousands of times, n_folds typically 3-10). Profiled to confirm no added hotspot beyond the band.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.evaluation.cv_delta_triage import triage_cv_delta


def _run(n_calls: int = 20_000, n_folds: int = 5) -> None:
    rng = np.random.default_rng(0)
    for _ in range(n_calls):
        baseline = 0.85 + 0.01 * rng.standard_normal(n_folds)
        candidate = baseline + 0.003
        triage_cv_delta(baseline, candidate, change_source="feature_engineering")
        triage_cv_delta(baseline, candidate, change_source="hyperparameter")


if __name__ == "__main__":
    n_calls = 20_000
    t0 = time.perf_counter()
    _run(n_calls=n_calls)
    wall = time.perf_counter() - t0
    print(f"wall for {n_calls} paired calls: {wall * 1000:.2f} ms ({wall / n_calls * 1e6:.2f} us/call pair)")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(n_calls=n_calls)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
