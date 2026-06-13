"""Bench: reuse the per-group mean/std series in ``generate_composite_group_agg_features``
instead of re-running ``grouped.agg("mean"|"std")`` inside the stat loop.

Run: ``python -m mlframe.feature_selection.filters._benchmarks.bench_composite_group_agg_mean_std_reuse``

The z-within / ratio residuals always materialise ``grouped.mean()`` and ``grouped.std(ddof=1)``.
The default stats include ``mean`` and ``std``, so the prior stat loop ran the identical cython
groupby a second time per (key-set, num_col). Reusing the already-computed series removes those two
O(n) aggregations; ``grouped.mean() == grouped.agg("mean")`` and ``grouped.std(ddof=1) ==
grouped.agg("std")`` bit-for-bit, so the encoding is unchanged.

The win scales with the groupby-agg fraction of wall: it is largest when there are many num_cols and
the composite key is low-cardinality (cheap ``np.unique``, so the removed aggregations dominate).

Measured (store-py 3.14, CPU, warm best-of-6, this dev box):
  - n=60k, 3 key-sets x 3 num_cols (np.unique/argsort on the object key dominates wall): ~1.0-1.08x
    -- the removed aggs are a small slice when the object-key sort is the bottleneck.
  - n=150k, 1 low-card key-set (8x6) x 12 num_cols (groupby aggs dominate): old ~430ms -> new ~385ms
    = 1.04-1.12x, new consistently <= old across interleaved A/B runs.
Output BIT-IDENTICAL (0.0 max-abs diff across 5 adversarial-magnitude trials incl mean/std/count/
min/max/median); regression sensor: tests/feature_selection/test_biz_value_mrmr_grouped_cat_fe/
test_composite_group_key.py::TestMeanStdReuseBitIdentical.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._composite_group_agg_fe import (
    generate_composite_group_agg_features,
)


def _bench(X, gsets, nums, reps: int = 6) -> float:
    for _ in range(2):
        generate_composite_group_agg_features(X, gsets, nums)
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        generate_composite_group_agg_features(X, gsets, nums)
        best = min(best, time.perf_counter() - t)
    return best * 1000.0


def main() -> None:
    rng = np.random.default_rng(0)

    n = 150_000
    cols = {"g1": rng.integers(0, 8, n), "g2": rng.integers(0, 6, n)}
    for k in range(12):
        cols[f"v{k}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    gsets = [("g1", "g2")]
    nums = [f"v{k}" for k in range(12)]
    print(f"groupby-dominated (n={n:_}, 1 low-card key x 12 num_cols): {_bench(X, gsets, nums):.1f} ms")


if __name__ == "__main__":
    main()
