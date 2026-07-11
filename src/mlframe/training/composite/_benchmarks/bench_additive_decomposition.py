"""cProfile harness for ``training.composite.additive_decomposition.AdditiveDecompositionRegressor``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_additive_decomposition``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite.additive_decomposition import AdditiveDecompositionRegressor


def _make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = x1 + rng.normal(scale=0.05, size=n)
    c1 = 2.0 * x1
    c2 = -1.5 * x2
    y = c1 + c2 + rng.normal(scale=0.02, size=n)
    X = np.column_stack([x1, x2]).astype(np.float32)
    return X, y.astype(np.float32), c1, c2


def _run(n: int, n_epochs: int, component_constraints=None) -> None:
    X, y, c1, c2 = _make_data(n, seed=0)
    model = AdditiveDecompositionRegressor(component_names=("c1", "c2"), hidden_sizes=(16, 8), n_epochs=n_epochs, component_constraints=component_constraints, random_state=0)
    model.fit(X, y, component_targets={"c1": c1, "c2": c2})
    model.predict(X)


if __name__ == "__main__":
    for n, n_epochs in [(500, 300), (5000, 300), (5000, 1000)]:
        t0 = time.perf_counter()
        _run(n, n_epochs)
        wall = time.perf_counter() - t0
        print(f"n={n:>6} n_epochs={n_epochs:>5} constraints=None -> {wall * 1000:9.2f} ms")

    # Confirm the softplus constraint path adds no meaningful overhead (one extra elementwise op per
    # constrained head per epoch -- expected to be noise-level against the dominant autograd/Adam cost).
    for n, n_epochs in [(5000, 1000)]:
        t0 = time.perf_counter()
        _run(n, n_epochs, component_constraints={"c1": "non_negative", "c2": "non_negative"})
        wall = time.perf_counter() - t0
        print(f"n={n:>6} n_epochs={n_epochs:>5} constraints=both_non_negative -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5000, 1000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
