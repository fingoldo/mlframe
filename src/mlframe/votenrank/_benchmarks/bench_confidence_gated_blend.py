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


def _run_calibrated(n_samples: int, n_calibration: int, n_calls: int) -> None:
    """Profile the ``per_sample_gate_calibration=True`` path: fits an isotonic regressor every call.

    The calibrator is fit fresh on every call here (worst case for a caller that doesn't cache/reuse it
    across calls) to surface its true per-call cost -- callers with a stable calibration set should fit once
    and swap to a cheaper apply-only path if this ever shows up hot in a real profile.
    """
    rng = np.random.default_rng(1)
    ensemble_pred = rng.uniform(0, 1, n_samples)
    auxiliary_pred = rng.uniform(0, 1, n_samples)
    confidence = rng.uniform(0, 1, n_samples)
    calibration_confidence = rng.uniform(0, 1, n_calibration)
    calibration_reliability = np.clip(calibration_confidence + rng.normal(0, 0.1, n_calibration), 0.0, 1.0)
    for _ in range(n_calls):
        confidence_gated_blend(
            ensemble_pred,
            auxiliary_pred,
            confidence,
            confidence_threshold=0.6,
            gated_weight=0.5,
            per_sample_gate_calibration=True,
            calibration_confidence=calibration_confidence,
            calibration_reliability=calibration_reliability,
        )


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

    print("--- per_sample_gate_calibration=True path (fresh isotonic fit every call) ---")
    for n_samples, n_calibration, n_calls in [(10_000, 2_000, 200), (1_000_000, 2_000, 20)]:
        t0 = time.perf_counter()
        _run_calibrated(n_samples, n_calibration, n_calls)
        wall = time.perf_counter() - t0
        print(
            f"n_samples={n_samples:>9,} n_calibration={n_calibration:>6,} n_calls={n_calls:>4} -> "
            f"{wall * 1000:9.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call"
        )

    profiler_cal = cProfile.Profile()
    profiler_cal.enable()
    _run_calibrated(1_000_000, 2_000, 20)
    profiler_cal.disable()
    buf_cal = StringIO()
    stats_cal = pstats.Stats(profiler_cal, stream=buf_cal).sort_stats("cumulative")
    stats_cal.print_stats(15)
    print(buf_cal.getvalue())
