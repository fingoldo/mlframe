"""cProfile harness for ``votenrank.correlation_diversity_ablation.diversity_ablation_report``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_correlation_diversity_ablation``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.correlation_diversity_ablation import diversity_ablation_report


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_dataset(n_samples: int, n_models: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n_samples)
    oof_preds = {}
    for i in range(n_models):
        noise_scale = 0.2 if i < n_models // 2 else 1.5
        oof_preds[f"m{i}"] = y_true + noise_scale * rng.standard_normal(n_samples)
    individual_scores = {name: -_rmse(y_true, pred) for name, pred in oof_preds.items()}
    return y_true, oof_preds, individual_scores


def _run(n_samples: int, n_models: int) -> None:
    y_true, oof_preds, individual_scores = _make_dataset(n_samples, n_models, seed=0)
    diversity_ablation_report(oof_preds, individual_scores, y_true, _rmse)


if __name__ == "__main__":
    for n_samples, n_models in [(2000, 10), (20000, 10), (20000, 50)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>6} n_models={n_models:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
