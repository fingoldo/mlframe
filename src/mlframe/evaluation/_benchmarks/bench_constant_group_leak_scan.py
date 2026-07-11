"""cProfile harness for ``evaluation.constant_group_target_scan``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_constant_group_leak_scan``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan


def _run(n_rows: int, n_cols: int, n_groups: int, combo_max_size: int = 1, combo_max_cols: int = 8) -> None:
    rng = np.random.default_rng(0)
    cols = {f"col_{i}": rng.integers(0, n_groups, n_rows) for i in range(n_cols)}
    df = pd.DataFrame(cols)
    y = rng.integers(0, 2, n_rows).astype(float)
    constant_group_target_scan(
        df, y, candidate_cols=list(cols.keys()), min_group_size=20, combo_max_size=combo_max_size, combo_max_cols=combo_max_cols
    )


if __name__ == "__main__":
    for n_rows, n_cols, n_groups in [(20_000, 10, 100), (200_000, 20, 500)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols, n_groups)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_cols={n_cols:>3} n_groups={n_groups:>4} -> {wall * 1000:9.2f} ms")

    # combo mode is O(k^depth) in the number of columns considered -- kept small (8 columns, depth 2 ->
    # C(8,2)=28 extra groupbys) so the benchmark stays fast; combo_max_cols is exactly the knob that bounds this.
    for n_rows, n_cols, n_groups, combo_max_size, combo_max_cols in [
        (20_000, 10, 100, 2, 8),
        (200_000, 20, 500, 2, 8),
    ]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols, n_groups, combo_max_size=combo_max_size, combo_max_cols=combo_max_cols)
        wall = time.perf_counter() - t0
        print(
            f"[combo] n_rows={n_rows:>9,} n_cols={n_cols:>3} n_groups={n_groups:>4} "
            f"combo_max_size={combo_max_size} combo_max_cols={combo_max_cols} -> {wall * 1000:9.2f} ms"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, 10, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, 10, 100, combo_max_size=2, combo_max_cols=8)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[combo profile]")
    print(buf.getvalue())
