"""cProfile harness for ``filters._orthogonal_tail_dependence_fe.tail_dependence_score`` (Layer 73,
mrmr_audit_2026-07-20 fe_expansion.md).

Run: ``python -m mlframe.feature_selection._benchmarks.bench_tail_dependence_cprofile``

Cost is dominated by ``n_perm`` independent shuffle-and-compare passes over the rank-uniformized
pair (the permutation-null floor); each pass is O(n), so total cost is O(n_perm * n) plus the O(n
log n) rank transform paid once.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._orthogonal_tail_dependence_fe import tail_dependence_score


def _make_data(n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n_rows)
    v = rng.uniform(0.0, 1.0, n_rows)
    return u, v


def _run(n_rows: int, n_perm: int) -> None:
    u, v = _make_data(n_rows, seed=0)
    tail_dependence_score(u, v, q=0.95, tail="upper", n_perm=n_perm, random_state=0)


if __name__ == "__main__":
    for n_rows, n_perm in [(2_000, 100), (20_000, 100), (100_000, 200)]:
        t0 = time.perf_counter()
        _run(n_rows, n_perm)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_perm={n_perm:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 200)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(25)
    print(stream.getvalue())
