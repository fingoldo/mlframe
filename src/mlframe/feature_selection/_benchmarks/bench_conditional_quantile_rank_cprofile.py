"""cProfile harness for ``filters._conditional_quantile_rank_fe.conditional_quantile_rank_fe``
(mrmr_audit_2026-07-20 fe_expansion.md "Extend conditional-dispersion to the full conditional
quantile").

Run: ``python -m mlframe.feature_selection._benchmarks.bench_conditional_quantile_rank_cprofile``

Cost is a Python loop over the distinct conditioning bins, each doing one ``np.sort`` (fit) plus
one ``np.searchsorted`` (apply) -- expect the sort to dominate per-bin, and the loop itself to
dominate wall time once the bin count is large (matches the module's own note that an exact
per-bin sort has no cupy segmented-sort primitive; a coarser cumulative-bincount approximation
would be the GPU-resident alternative).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._conditional_quantile_rank_fe import conditional_quantile_rank_fe


def _make_data(n_rows: int, n_bins: int, seed: int):
    rng = np.random.default_rng(seed)
    x_i = rng.standard_normal(n_rows)
    bins = rng.integers(0, n_bins, n_rows)
    return x_i, bins


def _run(n_rows: int, n_bins: int) -> None:
    x_i, bins = _make_data(n_rows, n_bins, seed=0)
    conditional_quantile_rank_fe(x_i, bins)


if __name__ == "__main__":
    for n_rows, n_bins in [(2_000, 10), (20_000, 30), (100_000, 50)]:
        t0 = time.perf_counter()
        _run(n_rows, n_bins)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_bins={n_bins:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 50)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())
