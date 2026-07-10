"""cProfile harness for ``feature_engineering.relational_dfs``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_relational_dfs``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.relational_dfs import ChildTableSpec, compute_relational_features, stack_relational_features


def _make_dataset(n_parents: int, n_children_per_parent: int, seed: int):
    rng = np.random.default_rng(seed)
    parent = pd.DataFrame({"cust_id": np.arange(n_parents), "cutoff": rng.uniform(5, 15, n_parents)})
    n_child = n_parents * n_children_per_parent
    child = pd.DataFrame(
        {
            "cust_id": rng.integers(0, n_parents, n_child),
            "ts": rng.uniform(0, 20, n_child),
            "amount": rng.normal(size=n_child),
        }
    )
    return parent, child


def _run(n_parents: int, n_children_per_parent: int) -> None:
    parent, child = _make_dataset(n_parents, n_children_per_parent, seed=0)
    spec = ChildTableSpec(child_df=child, foreign_key_col="cust_id", time_col="ts", value_cols={"amount": ["sum", "mean", "count"]}, prefix="txn")
    compute_relational_features(parent, "cust_id", "cutoff", [spec])


if __name__ == "__main__":
    for n_parents, n_children_per_parent in [(500, 10), (2000, 10), (2000, 50)]:
        t0 = time.perf_counter()
        _run(n_parents, n_children_per_parent)
        wall = time.perf_counter() - t0
        print(f"n_parents={n_parents:>5} children/parent={n_children_per_parent:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
