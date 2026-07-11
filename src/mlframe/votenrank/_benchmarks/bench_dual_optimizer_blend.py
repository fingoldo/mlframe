"""cProfile harness for ``votenrank.dual_optimizer_blend.dual_optimizer_weight_blend``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_dual_optimizer_blend``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.dual_optimizer_blend import dual_optimizer_weight_blend


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_dataset(n_samples: int, n_models: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n_samples) * 3.0
    preds = [y_true + 0.5 * rng.standard_normal(n_samples) for _ in range(n_models)]
    return y_true, preds


def _run(n_samples: int, n_models: int, n_optuna_trials: int, include_coord_descent: bool = False, n_coord_descent_iters: int = 300) -> None:
    y_true, preds = _make_dataset(n_samples, n_models, seed=0)
    dual_optimizer_weight_blend(
        preds,
        y_true,
        _rmse,
        n_restarts=5,
        n_optuna_trials=n_optuna_trials,
        random_state=0,
        include_coord_descent=include_coord_descent,
        n_coord_descent_iters=n_coord_descent_iters,
    )


if __name__ == "__main__":
    for n_samples, n_models, n_trials in [(2000, 5, 50), (2000, 5, 200), (20000, 10, 100)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models, n_trials)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>6} n_models={n_models:>3} n_trials={n_trials:>4} -> {wall * 1000:9.2f} ms")

    # 3-optimizer path (opt-in `include_coord_descent`) -- the added coordinate-descent search cost.
    for n_samples, n_models, n_trials, n_coord_iters in [(2000, 5, 50, 300), (2000, 5, 200, 300), (20000, 10, 100, 300)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models, n_trials, include_coord_descent=True, n_coord_descent_iters=n_coord_iters)
        wall = time.perf_counter() - t0
        print(f"[+coord_descent] n_samples={n_samples:>6} n_models={n_models:>3} n_trials={n_trials:>4} n_coord_iters={n_coord_iters:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20000, 10, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("2-optimizer profile:")
    print(buf.getvalue())

    profiler3 = cProfile.Profile()
    profiler3.enable()
    _run(20000, 10, 100, include_coord_descent=True, n_coord_descent_iters=300)
    profiler3.disable()
    buf3 = StringIO()
    stats3 = pstats.Stats(profiler3, stream=buf3).sort_stats("cumulative")
    stats3.print_stats(20)
    print("3-optimizer profile (with coordinate descent):")
    print(buf3.getvalue())
