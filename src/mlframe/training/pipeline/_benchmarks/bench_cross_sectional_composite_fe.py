"""cProfile harness for ``apply_cross_sectional_composite_fe``.

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_cross_sectional_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._cross_sectional_composite_fe import apply_cross_sectional_composite_fe


def _run(n_rows: int, n_snapshots: int, n_feature_cols: int, k: int):
    rng = np.random.default_rng(0)
    snap = rng.integers(0, n_snapshots, n_rows)
    data = {"time_id": snap}
    for i in range(n_feature_cols):
        data[f"f{i}"] = rng.normal(size=n_rows).astype(np.float32)
    df = pd.DataFrame(data)
    cfg = PreprocessingExtensionsConfig(
        cross_sectional_neighbors_snapshot_col="time_id",
        cross_sectional_neighbors_feature_cols=[f"f{i}" for i in range(n_feature_cols)],
        cross_sectional_neighbors_k=k,
    )
    apply_cross_sectional_composite_fe(df, None, None, cfg, {}, verbose=0)


if __name__ == "__main__":
    for n_rows, n_snapshots, n_feature_cols, k in [(50_000, 500, 5, 10), (300_000, 2_000, 8, 20)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_rows, n_snapshots, n_feature_cols, k)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(f"=== n_rows={n_rows} n_snapshots={n_snapshots} n_feature_cols={n_feature_cols} k={k} ===")
        print(s.getvalue())
