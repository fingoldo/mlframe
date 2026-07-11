"""cProfile harness for ``votenrank.hill_climb_ensemble``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_hill_climb``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.hill_climb import hill_climb_ensemble


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _run(n_samples: int, n_models: int, max_iterations: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(n_samples)
    preds = [y_true + rng.uniform(0.2, 2.0) * rng.standard_normal(n_samples) for _ in range(n_models)]
    for _ in range(n_calls):
        hill_climb_ensemble(preds, y_true, _rmse, maximize=False, max_iterations=max_iterations)


def _run_bagged(n_samples: int, n_models: int, max_iterations: int, n_calls: int, n_bags: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(n_samples)
    preds = [y_true + rng.uniform(0.2, 2.0) * rng.standard_normal(n_samples) for _ in range(n_models)]
    for _ in range(n_calls):
        hill_climb_ensemble(
            preds,
            y_true,
            _rmse,
            maximize=False,
            max_iterations=max_iterations,
            n_bags=n_bags,
            randomize_start=True,
            randomize_order=True,
            random_state=0,
        )


if __name__ == "__main__":
    for n_samples, n_models, max_iterations, n_calls in [(5_000, 20, 60, 20), (100_000, 50, 60, 3)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models, max_iterations, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n_samples:>9,} models={n_models:>3} iters<={max_iterations:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call")

    # bagged/randomized-restart path (opt-in via n_bags>1) -- roughly n_bags x the cost of a single path,
    # since it runs the same greedy loop n_bags independent times before averaging the weight vectors.
    for n_samples, n_models, max_iterations, n_calls in [(5_000, 20, 60, 5), (100_000, 50, 60, 1)]:
        n_bags_val = 10
        t0 = time.perf_counter()
        _run_bagged(n_samples, n_models, max_iterations, n_calls, n_bags_val)
        wall = time.perf_counter() - t0
        print(
            f"[bagged n_bags={n_bags_val}] n={n_samples:>9,} models={n_models:>3} iters<={max_iterations:>3} "
            f"-> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 50, 60, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_bagged = cProfile.Profile()
    profiler_bagged.enable()
    _run_bagged(100_000, 50, 60, 1, n_bags=10)
    profiler_bagged.disable()
    buf_bagged = StringIO()
    stats_bagged = pstats.Stats(profiler_bagged, stream=buf_bagged).sort_stats("cumulative")
    stats_bagged.print_stats(15)
    print("[bagged path]")
    print(buf_bagged.getvalue())

    # Rejected optimization attempt: batching `trial_sum = running_sum + preds[j]` across all candidates into
    # one `(n_models, n_samples)` vectorized numpy op (instead of n_models separate per-candidate array adds)
    # measured 0.38x (SLOWER, not faster) at n_samples=100_000, n_models=50 -- materializing the full candidate
    # matrix every iteration costs more in memory traffic than it saves in per-call numpy dispatch overhead.
    # The O(iterations * n_models * n_samples) cost is inherent to evaluating an arbitrary caller-supplied
    # metric_fn against every candidate at every greedy step; left as the simple per-candidate loop.
