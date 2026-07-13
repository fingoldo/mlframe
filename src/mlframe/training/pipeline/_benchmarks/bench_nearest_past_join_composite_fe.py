"""cProfile harness for ``apply_nearest_past_join_composite_fe``.

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_nearest_past_join_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._nearest_past_join_composite_fe import apply_nearest_past_join_composite_fe


def _run(n_entities: int, right_rows_per_entity: int, n_left: int):
    rng = np.random.default_rng(0)
    right = pd.DataFrame(
        {
            "entity": np.repeat(np.arange(n_entities), right_rows_per_entity),
            "t": np.tile(np.arange(right_rows_per_entity), n_entities),
            "known_value": rng.normal(size=n_entities * right_rows_per_entity),
        }
    )
    left = pd.DataFrame(
        {
            "entity": rng.integers(0, n_entities, n_left),
            "t": rng.integers(0, right_rows_per_entity, n_left).astype(float) + 0.5,
        }
    )
    cfg = PreprocessingExtensionsConfig(nearest_past_join_on="t", nearest_past_join_by=["entity"], nearest_past_join_value_cols=["known_value"])
    apply_nearest_past_join_composite_fe(left, None, None, cfg, right, metadata={}, verbose=0)


if __name__ == "__main__":
    for n_entities, right_rows_per_entity, n_left in [(5_000, 20, 50_000), (20_000, 30, 300_000)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_entities, right_rows_per_entity, n_left)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(f"=== n_entities={n_entities} right_rows_per_entity={right_rows_per_entity} n_left={n_left} ===")
        print(s.getvalue())
