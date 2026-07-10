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


def _run(n_samples: int, n_models: int, n_optuna_trials: int) -> None:
    y_true, preds = _make_dataset(n_samples, n_models, seed=0)
    dual_optimizer_weight_blend(preds, y_true, _rmse, n_restarts=5, n_optuna_trials=n_optuna_trials, random_state=0)


if __name__ == "__main__":
    for n_samples, n_models, n_trials in [(2000, 5, 50), (2000, 5, 200), (20000, 10, 100)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models, n_trials)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>6} n_models={n_models:>3} n_trials={n_trials:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20000, 10, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
