"""cProfile harness for ``apply_latent_interaction_svd_composite_fe``.

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_latent_interaction_svd_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._latent_interaction_svd_composite_fe import apply_latent_interaction_svd_composite_fe


def _run(n_customers: int, n_items: int, n_events: int, n_rows: int):
    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        {
            "customer_id": rng.integers(0, n_customers, n_events),
            "item_id": rng.integers(0, n_items, n_events),
            "qty": rng.integers(1, 5, n_events),
        }
    )
    group_ids = rng.integers(0, n_customers, n_rows)
    df = pd.DataFrame({"dummy": np.arange(n_rows)})
    cfg = PreprocessingExtensionsConfig(
        latent_interaction_svd_row_entity="customer_id", latent_interaction_svd_col_entity="item_id",
        latent_interaction_svd_weight_col="qty", latent_interaction_svd_n_components=10,
    )
    apply_latent_interaction_svd_composite_fe(df, None, None, cfg, events, group_ids, np.arange(n_rows), None, None, metadata={}, verbose=0)


if __name__ == "__main__":
    for n_customers, n_items, n_events, n_rows in [(5_000, 500, 50_000, 20_000), (50_000, 2_000, 500_000, 100_000)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_customers, n_items, n_events, n_rows)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(f"=== n_customers={n_customers} n_items={n_items} n_events={n_events} n_rows={n_rows} ===")
        print(s.getvalue())
