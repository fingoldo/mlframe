"""cProfile the per-category order-statistic target-encoding path at a production shape (n=100k, 500 categories).

Run: ``python -m mlframe.feature_selection._benchmarks.bench_te_order_stats``

The order-stat path (``_target_encoding_order_stats.per_category_order_stats``) is one ``np.lexsort`` per fold (the
dominant term) plus a single ``@numba.njit`` segment sweep. Measured on the dev box (n=100k, K=500, all 7 order stats,
warm JIT): the full 5-fold fit is ~0.30s wall, of which the 6 ``np.lexsort`` calls (5 folds + full-data) dominate the
``per_category_order_stats`` tottime and the njit segment kernel is a small fraction -- lexsort is numpy's C sort and is
not beatable in Python. No actionable speedup: the sort is inherent (order stats need y sorted within each category) and
already vectorised in one pass; the njit kernel avoids any per-category Python call. An argsort on category codes alone
(then a per-segment sort) was NOT faster -- a single lexsort beats N small sorts.
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._target_encoding_fe import kfold_target_encode_fit

_STATS = ("median", "trimmed_mean", "q10", "q90", "iqr", "min", "max")


def _make_data(n=100_000, k=500, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, k, n)
    centers = rng.normal(0, 3, k)
    y = centers[cat] + rng.standard_t(2.0, size=n) * 2.0
    return pd.DataFrame({"c": cat}), y


def main():
    df, y = _make_data()
    # Warm numba JIT (first call compiles the segment kernel).
    kfold_target_encode_fit(df.iloc[:2000], y[:2000], ["c"], stats=_STATS, n_folds=5)

    t0 = time.perf_counter()
    for _ in range(3):
        kfold_target_encode_fit(df, y, ["c"], stats=_STATS, n_folds=5)
    wall = (time.perf_counter() - t0) / 3
    print(f"n={len(df)} K=500 stats={_STATS}: {wall * 1000:.1f} ms / fit (5-fold, warm)")

    pr = cProfile.Profile()
    pr.enable()
    kfold_target_encode_fit(df, y, ["c"], stats=_STATS, n_folds=5)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
