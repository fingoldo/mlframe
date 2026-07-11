"""cProfile harness for ``training.neural.fixed_sparse_linear.FixedSparseLinear``.

Run: ``python -m mlframe.training.neural._benchmarks.bench_fixed_sparse_linear``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import torch

from mlframe.training.neural.fixed_sparse_linear import FixedSparseLinear


def _run(in_features: int, out_features: int, batch_size: int, n_forward: int) -> None:
    layer = FixedSparseLinear(in_features, out_features, sparsity=0.9, random_state=0)
    x = torch.randn(batch_size, in_features)
    for _ in range(n_forward):
        layer(x)


if __name__ == "__main__":
    for in_f, out_f, batch, n in [(100, 128, 64, 200), (500, 512, 64, 200), (500, 512, 256, 200)]:
        t0 = time.perf_counter()
        _run(in_f, out_f, batch, n)
        wall = time.perf_counter() - t0
        print(f"in={in_f:>4} out={out_f:>4} batch={batch:>4} n_forward={n:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500, 512, 256, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
