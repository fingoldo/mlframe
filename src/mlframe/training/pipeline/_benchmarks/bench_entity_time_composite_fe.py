"""cProfile harness for ``apply_entity_time_composite_fe`` (state_duration + recency_aggregation).

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_entity_time_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._entity_time_composite_fe import apply_entity_time_composite_fe


def _run(n_rows: int, n_groups: int):
    rng = np.random.default_rng(0)
    group_ids = rng.integers(0, n_groups, n_rows)
    ts = np.arange(n_rows).astype(np.float64)
    df = pd.DataFrame(
        {
            "state_col": rng.random(n_rows) < 0.3,
            "val_col": rng.normal(size=n_rows).astype(np.float32),
        }
    )
    cfg = PreprocessingExtensionsConfig(state_duration_columns=["state_col"], recency_aggregation_columns=["val_col"])
    apply_entity_time_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(n_rows), None, None, verbose=0)


if __name__ == "__main__":
    for n_rows, n_groups in [(50_000, 500), (500_000, 5_000)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_rows, n_groups)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(f"=== n_rows={n_rows} n_groups={n_groups} ===")
        print(s.getvalue())
