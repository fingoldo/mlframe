"""cProfile harness for ``forward_select`` with ``initial_selected`` (stacking-core-augmented forward selection).

Run: ``python -m mlframe.feature_selection._benchmarks.bench_forward_select_initial_selected``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.feature_selection.forward_select import forward_select


def _make_dataset(n: int, n_raw: int, seed: int):
    rng = np.random.default_rng(seed)
    true_signal = rng.normal(size=n)
    base_pred = true_signal + rng.normal(scale=0.5, size=n)
    X = pd.DataFrame({"base_pred": base_pred, **{f"raw{i}": rng.normal(size=n) for i in range(n_raw)}})
    return X, true_signal


def _run(n: int, n_raw: int) -> None:
    X, y = _make_dataset(n, n_raw, seed=0)
    raw_candidates = [c for c in X.columns if c != "base_pred"]
    forward_select(X, y, lambda: Ridge(alpha=1.0), scoring="neg_mean_squared_error", cv=3, candidate_features=raw_candidates, initial_selected=["base_pred"], max_features=5)


if __name__ == "__main__":
    for n, n_raw in [(500, 10), (500, 30), (2000, 30)]:
        t0 = time.perf_counter()
        _run(n, n_raw)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_raw={n_raw:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 30)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
