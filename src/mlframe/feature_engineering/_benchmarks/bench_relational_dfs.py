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

from mlframe.feature_engineering.relational_dfs import ChildTableSpec, RelationalHop, compute_relational_features, stack_relational_chain, stack_relational_features


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


def _make_chain_dataset(n_parents: int, n_hop1_per_parent: int, n_hop2_per_hop1: int, n_leaf_per_hop2: int, seed: int):
    # depth-4 chain (3 hops + leaf): parent -> hop1 -> hop2 -> leaf, mirroring a device/session/event hierarchy.
    rng = np.random.default_rng(seed)
    parent = pd.DataFrame({"account_id": np.arange(n_parents), "cutoff": np.full(n_parents, 3000.0)})

    n_hop1 = n_parents * n_hop1_per_parent
    hop1 = pd.DataFrame({"hop1_id": np.arange(n_hop1), "account_id": rng.integers(0, n_parents, n_hop1), "hop1_ts": np.full(n_hop1, 2000.0)})

    n_hop2 = n_hop1 * n_hop2_per_hop1
    hop2 = pd.DataFrame(
        {
            "hop2_id": np.arange(n_hop2),
            "hop1_id": rng.integers(0, n_hop1, n_hop2),
            "hop2_ts": np.full(n_hop2, 1000.0),
            "hop2_val": rng.normal(size=n_hop2),
        }
    )

    n_leaf = n_hop2 * n_leaf_per_hop2
    leaf = pd.DataFrame({"hop2_id": rng.integers(0, n_hop2, n_leaf), "leaf_ts": rng.uniform(0, 1000, n_leaf), "leaf_val": rng.normal(size=n_leaf)})
    return parent, hop1, hop2, leaf


def _run_chain(n_parents: int, n_hop1_per_parent: int, n_hop2_per_hop1: int, n_leaf_per_hop2: int) -> None:
    parent, hop1, hop2, leaf = _make_chain_dataset(n_parents, n_hop1_per_parent, n_hop2_per_hop1, n_leaf_per_hop2, seed=0)
    leaf_spec = ChildTableSpec(child_df=leaf, foreign_key_col="hop2_id", time_col="leaf_ts", value_cols={"leaf_val": ["sum", "mean"]}, prefix="leaf")
    stack_relational_chain(
        parent_df=parent,
        parent_id_col="account_id",
        cutoff_col="cutoff",
        hops=[
            RelationalHop(df=hop1, id_col="hop1_id", time_col="hop1_ts", foreign_key_col="account_id", value_cols={}),
            RelationalHop(df=hop2, id_col="hop2_id", time_col="hop2_ts", foreign_key_col="hop1_id", value_cols={"hop2_val": ["sum", "mean"]}),
        ],
        leaf_specs=[leaf_spec],
        prefix="l3",
    )


if __name__ == "__main__":
    for n_parents, n_children_per_parent in [(500, 10), (2000, 10), (2000, 50)]:
        t0 = time.perf_counter()
        _run(n_parents, n_children_per_parent)
        wall = time.perf_counter() - t0
        print(f"compute_relational_features: n_parents={n_parents:>5} children/parent={n_children_per_parent:>3} -> {wall * 1000:9.2f} ms")

    for n_parents, n_hop1, n_hop2, n_leaf in [(500, 3, 3, 5), (2000, 3, 3, 5), (2000, 4, 4, 8)]:
        t0 = time.perf_counter()
        _run_chain(n_parents, n_hop1, n_hop2, n_leaf)
        wall = time.perf_counter() - t0
        print(f"stack_relational_chain (depth-3): n_parents={n_parents:>5} hop1/parent={n_hop1} hop2/hop1={n_hop2} leaf/hop2={n_leaf} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("=== compute_relational_features cProfile ===")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_chain(2000, 4, 4, 8)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("=== stack_relational_chain (depth-3) cProfile ===")
    print(buf.getvalue())
