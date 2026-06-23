"""Bench: apriori_itemsets lift-scoring + top-K column-index lookup.

Hotspot (apriori_itemsets.py _process): for every frequent itemset the OLD code
called ``col_names.index(it)`` — an O(M) linear scan through the ``d*n_bins``
column-name list — per item, both in the lift loop (every frequent itemset) and
again in the top-K emit loop. With M = d*n_bins column names and F frequent
itemsets of length up to ``max_len``, that is O(F * max_len * M) python-level
string comparisons, all of which a single ``{name: idx}`` dict built once turns
into O(F * max_len) O(1) lookups. Bit-identical by construction (same indices).

This isolates ONLY the index-resolution work (list.index vs dict) at a realistic
(F, M) drawn from a real fpgrowth run, best-of-N, warm.

Run:
    python -m mlframe.feature_engineering._benchmarks.bench_apriori_colidx_dict
"""
from __future__ import annotations

import time

import numpy as np


def _make_realistic_itemsets(d: int, n_bins: int, n_frequent: int, max_len: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    col_names = [f"f{j}_b{b}" for j in range(d) for b in range(n_bins)]
    M = len(col_names)
    itemsets = []
    for _ in range(n_frequent):
        L = int(rng.integers(1, max_len + 1))
        idx = rng.choice(M, size=L, replace=False)
        itemsets.append([col_names[i] for i in idx])
    return col_names, itemsets


def _old_resolve(col_names, itemsets):
    out = []
    for itemset in itemsets:
        cols_idx = [col_names.index(it) for it in itemset]
        out.append(cols_idx)
    return out


def _new_resolve(col_names, itemsets):
    col_index = {name: i for i, name in enumerate(col_names)}
    out = []
    for itemset in itemsets:
        cols_idx = [col_index[it] for it in itemset]
        out.append(cols_idx)
    return out


def _best_of(fn, args, n=7):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    # Realistic mammography-ish FE shape: d=30 features, 5 bins -> M=150 col names.
    # fpgrowth at min_support=0.05/max_len=3 on rare-positive data routinely
    # yields hundreds of frequent itemsets.
    for d, n_bins, F, max_len in [(30, 5, 200, 3), (60, 5, 800, 3), (100, 5, 2000, 3)]:
        col_names, itemsets = _make_realistic_itemsets(d, n_bins, F, max_len)
        # identity gate
        old = _old_resolve(col_names, itemsets)
        new = _new_resolve(col_names, itemsets)
        assert old == new, "index resolution diverged!"
        # warm
        _best_of(_old_resolve, (col_names, itemsets), n=2)
        _best_of(_new_resolve, (col_names, itemsets), n=2)
        t_old = _best_of(_old_resolve, (col_names, itemsets))
        t_new = _best_of(_new_resolve, (col_names, itemsets))
        M = len(col_names)
        print(f"M={M:4d} F={F:5d} max_len={max_len} | OLD={t_old*1e3:8.3f}ms NEW={t_new*1e3:8.3f}ms speedup={t_old/t_new:6.2f}x identity=OK")


if __name__ == "__main__":
    main()
