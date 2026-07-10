"""cProfile harness for ``evaluation.flag_subgroup_only_feature_overfit_risk``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_subgroup_feature_overfit_risk``

A thin wrapper around the already-benched ``subpopulation_ratio_drift_check`` (see ``bench_subpopulation_drift.py``)
plus O(1) dict lookups; expect its own added cost to be negligible relative to the wrapped call.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.subgroup_feature_overfit_risk import flag_subgroup_only_feature_overfit_risk


def _run(n_rows: int) -> None:
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=n_rows, p=[0.9, 0.1])})
    test_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=n_rows, p=[0.99, 0.01])})
    flag_subgroup_only_feature_overfit_risk(train_df, test_df, subgroup_col="loan_type", feature_subgroup_value="revolving", cv_delta=0.006)


if __name__ == "__main__":
    for n_rows in (10_000, 1_000_000):
        t0 = time.perf_counter()
        _run(n_rows)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
