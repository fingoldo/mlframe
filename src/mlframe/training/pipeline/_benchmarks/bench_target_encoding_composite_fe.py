"""cProfile harness for ``apply_target_encoding_composite_fe``.

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_target_encoding_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._target_encoding_composite_fe import apply_target_encoding_composite_fe


def _run(n_rows: int, n_entities: int):
    rng = np.random.default_rng(0)
    group_ids = rng.integers(0, n_entities, n_rows)
    ts = np.arange(n_rows).astype(np.float64)
    cat_col = rng.choice(list("ABCDE"), n_rows)
    y = rng.normal(size=n_rows)
    df = pd.DataFrame({"cat_col": cat_col})
    n_train = int(n_rows * 0.7)
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"])
    apply_target_encoding_composite_fe(
        df.iloc[:n_train].reset_index(drop=True), None, df.iloc[n_train:].reset_index(drop=True),
        cfg, group_ids, ts, y[:n_train], np.arange(n_train), None, np.arange(n_train, n_rows), metadata={}, verbose=0,
    )


if __name__ == "__main__":
    for n_rows, n_entities in [(50_000, 5_000), (300_000, 20_000)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_rows, n_entities)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(f"=== n_rows={n_rows} n_entities={n_entities} ===")
        print(s.getvalue())
