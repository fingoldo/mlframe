"""cProfile harness for ``training.neural.field_grouped_mlp.FieldGroupedMLPRegressor``.

Run: ``python -m mlframe.training.neural._benchmarks.bench_field_grouped_mlp``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.neural.field_grouped_mlp import FieldGroupedMLPRegressor


def _make_data(n: int, field_size: int, n_fields: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_features = field_size * n_fields
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    field_groups = {f"F{f}": list(range(f * field_size, (f + 1) * field_size)) for f in range(n_fields)}
    return X, y, field_groups


def _run_fit(n: int, field_size: int, n_fields: int, n_epochs: int) -> None:
    X, y, field_groups = _make_data(n, field_size, n_fields)
    FieldGroupedMLPRegressor(field_groups=field_groups, n_epochs=n_epochs, random_state=0).fit(X, y)


if __name__ == "__main__":
    for n, field_size, n_fields, n_epochs in [(150, 5, 3, 100), (300, 10, 5, 150), (300, 10, 5, 300)]:
        t0 = time.perf_counter()
        _run_fit(n, field_size, n_fields, n_epochs)
        wall = time.perf_counter() - t0
        print(f"n={n:>4} field_size={field_size:>3} n_fields={n_fields:>2} n_epochs={n_epochs:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_fit(300, 10, 5, 300)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
