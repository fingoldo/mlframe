"""cProfile harness for ``apply_ma_crossover_composite_fe``.

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_ma_crossover_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._ma_crossover_composite_fe import apply_ma_crossover_composite_fe


def _run(n_rows: int, n_groups: int, n_cols: int):
    rng = np.random.default_rng(0)
    group_ids = rng.integers(0, n_groups, n_rows)
    ts = np.arange(n_rows).astype(np.float64)
    data = {f"val{i}": rng.normal(size=n_rows).cumsum() for i in range(n_cols)}
    df = pd.DataFrame(data)
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=list(data.keys()), ma_crossover_windows=[3, 5, 10])
    apply_ma_crossover_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(n_rows), None, None, metadata={}, verbose=0)


if __name__ == "__main__":
    for n_rows, n_groups, n_cols in [(50_000, 500, 3), (300_000, 2_000, 5)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_rows, n_groups, n_cols)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(f"=== n_rows={n_rows} n_groups={n_groups} n_cols={n_cols} ===")
        print(s.getvalue())
