"""cProfile harness for ``apply_categorical_composite_fe`` (powerset + auto-group concat).

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_categorical_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._categorical_composite_fe import apply_categorical_composite_fe


def _make_frame(n_rows: int, n_cat_cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"cat{i}": rng.choice(list("ABCDEFGH"), n_rows) for i in range(n_cat_cols)}
    data["num0"] = rng.normal(size=n_rows).astype(np.float32)
    return pd.DataFrame(data)


def _run(n_rows: int, n_cat_cols: int, max_order: int):
    df = _make_frame(n_rows, n_cat_cols)
    y = np.random.default_rng(1).integers(0, 2, n_rows)
    cfg = PreprocessingExtensionsConfig(
        categorical_powerset_concat_enabled=True, categorical_group_concat_auto_enabled=True,
        categorical_powerset_concat_max_order=max_order, categorical_composite_max_source_columns=n_cat_cols,
    )
    n_train = int(n_rows * 0.7)
    apply_categorical_composite_fe(df.iloc[:n_train], df.iloc[n_train:], None, cfg, y[:n_train], {}, verbose=0)


if __name__ == "__main__":
    for n_rows, n_cat_cols, max_order in [(5_000, 4, 2), (50_000, 4, 2), (50_000, 6, 3)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_rows, n_cat_cols, max_order)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(25)
        print(f"=== n_rows={n_rows} n_cat_cols={n_cat_cols} max_order={max_order} ===")
        print(s.getvalue())
