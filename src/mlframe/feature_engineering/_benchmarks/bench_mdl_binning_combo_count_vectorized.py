"""Bench: MDL-binning pairwise combo-count — vectorised np.unique+searchsorted vs Counter.get() loop.

The OLD path built a ``collections.Counter`` over the train bin-combo codes and then ran a
per-query-row Python list comprehension ``[counts.get(int(c), 0) for c in query_combo]`` plus a
``len(set(query_combo))`` pass. Both scale linearly in the number of query rows in pure Python.

The NEW path uses ``np.unique(train_combo, return_counts=True)`` + ``np.searchsorted`` to look up
every query row's count in one vectorised C call. Counts are integers, so the result is
bit-identical to the dict path (verified here and in the regression test).

Run:
    CUDA_VISIBLE_DEVICES="" python bench_mdl_binning_combo_count_vectorized.py
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np


def _old(train_combo: np.ndarray, query_combo: np.ndarray):
    combo_counts = Counter(train_combo)
    out = np.array([combo_counts.get(int(c), 0) for c in query_combo], dtype=np.float32)
    unique_combos = float(len(set(query_combo)))
    return out, unique_combos


def _new(train_combo: np.ndarray, query_combo: np.ndarray):
    uniq_combo, uniq_counts = np.unique(train_combo, return_counts=True)
    pos = np.searchsorted(uniq_combo, query_combo)
    pos_clipped = np.clip(pos, 0, uniq_combo.shape[0] - 1)
    matched = uniq_combo[pos_clipped] == query_combo
    out = np.where(matched, uniq_counts[pos_clipped], 0).astype(np.float32)
    unique_combos = float(np.unique(query_combo).shape[0])
    return out, unique_combos


def _best_of(fn, *args, n: int = 7) -> float:
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    for n_train, n_query in [(10_000, 5_000), (80_000, 20_000), (80_000, 40_000)]:
        tb = rng.integers(0, 9, n_train) * 100 + rng.integers(0, 9, n_train)
        qb = rng.integers(0, 9, n_query) * 100 + rng.integers(0, 9, n_query)
        out_old, uc_old = _old(tb, qb)
        out_new, uc_new = _new(tb, qb)
        assert np.array_equal(out_old, out_new), "combo counts diverged"  # nosec B101 - internal invariant check in src/mlframe/feature_engineering/_benchmarks, not reachable with untrusted input
        assert uc_old == uc_new, (uc_old, uc_new)  # nosec B101 - internal invariant check in src/mlframe/feature_engineering/_benchmarks, not reachable with untrusted input
        b_old = _best_of(_old, tb, qb)
        b_new = _best_of(_new, tb, qb)
        print(f"n_train={n_train} n_query={n_query}: OLD {b_old * 1000:.2f}ms  " f"NEW {b_new * 1000:.2f}ms  speedup {b_old / b_new:.1f}x  identical=True")


if __name__ == "__main__":
    main()
