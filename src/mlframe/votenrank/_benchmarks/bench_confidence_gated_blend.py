"""cProfile harness for ``votenrank.confidence_gated_blend``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_confidence_gated_blend``

Backend comparison at n=1,000,000 (host input), 2026-07-10: numpy 20.7ms, njit 4.4ms, njit_parallel 3.6ms,
cupy resident 0.8ms, cupy end-to-end (H2D+compute+D2H) 8.5ms -- njit_parallel wins for host input at this
size; cupy only wins when the caller already holds GPU-resident arrays. Dispatch is routed through KTC (see
``_confidence_gated_blend_ktc_dispatch.py``); rejected the naive "always cupy" choice since e2e transfer cost
loses to njit_parallel for the common host-array calling convention.

NOTE: the FIRST several calls at a brand-new shape in a fresh process pay a one-time cost while the KTC
``async_sweep`` background thread measures all backends (incl. compiling njit and probing cupy) concurrently
with the foreground loop -- confirmed via a warmed-cache/separate-process re-run at n=10,000 that steady-state
per-call cost is sub-millisecond, matching njit_parallel. This is expected ``async_sweep=True`` behavior, not
a regression.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.confidence_gated_blend import confidence_gated_blend


def _run(n_samples: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    ensemble_pred = rng.uniform(0, 1, n_samples)
    auxiliary_pred = rng.uniform(0, 1, n_samples)
    confidence = rng.uniform(0, 1, n_samples)
    for _ in range(n_calls):
        confidence_gated_blend(ensemble_pred, auxiliary_pred, confidence, confidence_threshold=0.6, gated_weight=0.5)


if __name__ == "__main__":
    for n_samples, n_calls in [(10_000, 500), (1_000_000, 50)]:
        t0 = time.perf_counter()
        _run(n_samples, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>9,} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
