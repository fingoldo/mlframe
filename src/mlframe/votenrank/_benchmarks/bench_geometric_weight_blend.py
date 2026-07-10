"""cProfile harness for ``votenrank.geometric_weight_blend.geometric_weight_blend``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_geometric_weight_blend``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.metrics import log_loss

from mlframe.votenrank.geometric_weight_blend import geometric_weight_blend


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _log_loss(y_true, y_pred):
    return float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7)))


def _make_dataset(n_samples: int, n_models: int, seed: int):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(n_models, n_samples))
    y = (rng.uniform(size=n_samples) < _sigmoid(xs.sum(axis=0))).astype(int)
    preds = [_sigmoid(x) for x in xs]
    return y, preds


def _run(n_samples: int, n_models: int) -> None:
    y, preds = _make_dataset(n_samples, n_models, seed=0)
    geometric_weight_blend(preds, y, _log_loss, n_restarts=5, random_state=0)


if __name__ == "__main__":
    for n_samples, n_models in [(2000, 5), (20000, 5), (20000, 20)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>6} n_models={n_models:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20000, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
