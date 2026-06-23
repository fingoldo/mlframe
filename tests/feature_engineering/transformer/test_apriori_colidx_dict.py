"""Regression test for the apriori_itemsets column-index dict optimization.

The lift-scoring + top-K emit loops in ``compute_apriori_itemsets_features``
resolve each frequent itemset's column positions. The OLD code used
``col_names.index(it)`` (O(M) linear scan per item); the optimization builds a
single ``{name: idx}`` dict and does O(1) lookups. The indices are identical by
construction, so this test pins:

1. The dict resolution returns exactly the same indices as ``list.index`` for a
   realistic ``f{j}_b{b}`` column-name layout (the optimization's invariant).
2. The full feature function still produces finite, correctly-shaped output and
   is deterministic across two runs (selection/output stability).

bench: src/mlframe/feature_engineering/_benchmarks/bench_apriori_colidx_dict.py
"""
from __future__ import annotations

import numpy as np
import pytest


def test_colidx_dict_matches_list_index():
    """O(1) dict lookup must equal O(M) list.index for every column name."""
    d, n_bins = 30, 5
    col_names = [f"f{j}_b{b}" for j in range(d) for b in range(n_bins)]
    col_index = {name: i for i, name in enumerate(col_names)}
    rng = np.random.default_rng(0)
    for _ in range(500):
        L = int(rng.integers(1, 4))
        names = [col_names[i] for i in rng.choice(len(col_names), size=L, replace=False)]
        assert [col_index[it] for it in names] == [col_names.index(it) for it in names]


def test_apriori_features_deterministic_and_finite():
    """Full feature function: deterministic across runs + finite, correctly shaped."""
    pytest.importorskip("mlxtend")
    from sklearn.model_selection import KFold

    from mlframe.feature_engineering.transformer.apriori_itemsets import (
        compute_apriori_itemsets_features,
    )

    rng = np.random.default_rng(42)
    n, d = 800, 30
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (rng.random(n) < 0.05).astype(np.float32)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)

    out1 = compute_apriori_itemsets_features(X, y, None, splitter=spl, seed=123, task="binary").to_numpy()
    out2 = compute_apriori_itemsets_features(X, y, None, splitter=spl, seed=123, task="binary").to_numpy()

    top_k = 8
    assert out1.shape == (n, top_k + 2)
    assert np.all(np.isfinite(out1))
    # Same seed + splitter -> bit-identical output (dict resolution is deterministic).
    assert np.array_equal(out1, out2)
